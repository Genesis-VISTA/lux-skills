"""
moe_align_block_size on ROCm (MI250X) at prefill sizes: two Triton kernels
replace the _moe_C op. The expert assignment is the same: identical
num_tokens_post_padded, expert_ids, per-expert token sets and padding.

THE PROBLEM

moe_align_block_size runs before every routed-expert GEMM pair (TritonExperts,
fused_moe). The _moe_C op does the whole count / prefix / padding fill in one
1024-thread workgroup, then count_and_sort places every token with a global
atomic per element on 256 cursors. With T * topk = 49152 / 98304 entries
(8192 / 16384-token chunk, top-6) that is 35 / 67 us of device time per MoE
layer on one GCD, the single workgroup accounting for two thirds of it.

THE FIX

For topk_ids.numel() >= _TRITON_ALIGN_MIN_NUMEL (T >= 4096 at top-6) with
no expert_map:

1. _moe_align_count (one program per 1024 entries): tl.histogram of the
   block, one vector atomic_add of the 256 counts into counts[group, :],
   group = block % 8.
2. _moe_align_place (one program per 1024 entries, at least enough programs
   to cover expert_ids in 16-block slices): every program rebuilds the
   padded per-expert offsets from counts (256-wide cumsum), places its
   entries at start[e] + (earlier groups' count of e) + an atomic slot in
   cursor[group, e], writes expert_ids for its slice by binary search over
   the padded ends (-1 past num_tokens_post_padded, as the op does) and the
   sentinel numel into every padding slot up to the buffer end. It also
   zeroes the other half of the double-buffered counts / cursors workspace,
   which the next call on the same stream uses.

The 8 groups split each expert's cursor 8 ways, which cuts the atomic
contention that bounds count_and_sort (one group: 27 us at T=16384).

Token order inside an expert's range is atomic order, as it already is for
count_and_sort. The GEMMs compute each row independently of the other rows of
its block (the K reduction is per row), so c2 / c3 do not depend on it.

Measured on one MI250X GCD (Tioga), Inkling-Small-bf16 TP8 per-rank shapes
(256 routed experts, top-6), captured layer 11 routing, served env, block
size from the served t8kg configs (64), device time per
_prepare_expert_assignment from torch.profiler, installed patched vLLM with
the switch flipped in-process (job f61xcZqQV4FR):

| T | _moe_C op | this patch |
|---|---|---|
| 4096 | 21.9 us | 11.6 us |
| 8192 | 35.0 us | 13.8 us |
| 12288 | 51.5 us | 14.6 us |
| 16384 | 66.9 us | 16.1 us |

num_tokens_post_padded, expert_ids (whole buffer) and the padding / tail of
sorted_token_ids are torch.equal to the op's (block size 16 / 32 / 64 / 128,
pad_sorted_ids, int32 and int64 ids), the valid entries hold the same token
set per expert, and _g1act -> _w2_astat through _prepare_expert_assignment
with real layer 11 / 20 weights gives torch.equal c2 and c3 at
T=8192 / 12288 / 16384.

Host cost: the call sits on the host path of every MoE layer. After the first
call per (device, stream, experts, block size, ids dtype) both kernels are
launched through their compiled launchers (no argument binding), and the
workspace is cached per (device, stream) and zeroed on device as above (no
memset). Python per moe_align_block_size call, perf_counter median (job
f61xmJUdFuhH, T=4096..16384): op 36-38 us, this patch 28-30 us (with
torch.zeros + kernel[grid] launches it was 134-136 us). Device time with the
in-kernel zeroing: 10.5 / 11.8 / 13.6 / 15.0 us at T=4096 / 8192 / 12288 /
16384.

Below T=4096 the device saving is under 10 us, so smaller batches (decode)
keep the op, as does any call during CUDA graph capture (the workspace
double-buffering is host state).

INKLING_MOE_ALIGN_TRITON=0 keeps the _moe_C op everywhere.
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/model_executor/layers/fused_moe/moe_align_block_size.py"

OLD_IMPORTS = '''from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up
'''

NEW_IMPORTS = '''from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

# Zenith patch rocm_moe_align_triton.py: Triton moe_align_block_size on ROCm
# for prefill-sized topk_ids (MI250X Inkling-Small: 35 -> 14 us at T=8192,
# 67 -> 16 us at T=16384 per layer). Same expert assignment as the _moe_C
# op; token order inside an expert is atomic order, as in the op. Host cost
# below the op's (direct launches, no memset). INKLING_MOE_ALIGN_TRITON=0
# disables.
import os as _align_os  # noqa: E402

_TRITON_ALIGN = (
    _align_os.environ.get("INKLING_MOE_ALIGN_TRITON", "1") != "0"
    and torch.version.hip is not None
    and hasattr(torch._C, "_cuda_getCurrentRawStream")
)
_TRITON_ALIGN_MIN_NUMEL = 4096 * 6
_TRITON_ALIGN_BLOCK = 1024
_TRITON_ALIGN_NG = 8  # cursor groups
_TRITON_ALIGN_BJ = 16
_TRITON_ALIGN_ZB = 256
_align_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
# (device, stream, E) -> double-buffered workspace;
# (device, stream, E, block_size, dtype) -> direct launchers
_TRITON_ALIGN_STATE: dict = {}


@triton.jit(do_not_specialize=["n"])
def _moe_align_count(
    ids_ptr, cnt_ptr, n, E: tl.constexpr, BLOCK: tl.constexpr, NG: tl.constexpr
):
    b = tl.program_id(0)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    e = tl.load(ids_ptr + offs, mask=m, other=0).to(tl.int32)
    h = tl.histogram(e, E)
    # masked lanes were counted in bin 0
    h -= tl.where(tl.arange(0, E) == 0, BLOCK - tl.sum(m.to(tl.int32), axis=0), 0)
    tl.atomic_add(cnt_ptr + (b % NG) * E + tl.arange(0, E), h, mask=h > 0, sem="relaxed")


@triton.jit(do_not_specialize=["n", "NB", "max_pad", "max_blocks"])
def _moe_align_place(
    ids_ptr, cnt_ptr, cur_ptr, sorted_ptr, eid_ptr, ntpp_ptr, z_ptr, n, NB, max_pad,
    max_blocks, E: tl.constexpr, LOG2E: tl.constexpr, BLOCK: tl.constexpr,
    NG: tl.constexpr, BM: tl.constexpr, BJ: tl.constexpr, ZN: tl.constexpr,
    ZB: tl.constexpr,
):
    b = tl.program_id(0)
    ar = tl.arange(0, E)
    g = b % NG
    rows = tl.arange(0, NG)
    cg = tl.load(cnt_ptr + rows[:, None] * E + ar[None, :])
    cnt = tl.sum(cg, axis=0)
    padded = (cnt + BM - 1) // BM * BM
    pend = tl.cumsum(padded, axis=0)
    start = pend - padded
    total = tl.sum(padded, axis=0)
    if b == 0:
        tl.store(ntpp_ptr, total)
    if b < NB:
        gbase = start + tl.sum(tl.where(rows[:, None] < g, cg, 0), axis=0)
        offs = b * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        e = tl.load(ids_ptr + offs, mask=m, other=0).to(tl.int32)
        slot = tl.atomic_add(cur_ptr + g * E + e, 1, mask=m, sem="relaxed")
        tl.store(sorted_ptr + tl.gather(gbase, e, 0) + slot, offs, mask=m)
    # expert_ids for blocks [b * BJ, b * BJ + BJ): the expert whose padded
    # range holds the block start (binary search over the padded ends)
    j = b * BJ + tl.arange(0, BJ)
    pos = j * BM
    jm = j < max_blocks
    inb = pos < total
    ex = tl.zeros((BJ,), dtype=tl.int32)
    for k in tl.static_range(LOG2E - 1, -1, -1):
        c = ex + (1 << k)
        ex = tl.where(tl.gather(pend, c - 1, 0) <= pos, c, ex)
    tl.store(eid_ptr + j, tl.where(inb, ex, -1), mask=jm)
    vend = tl.gather(start + cnt, tl.where(inb, ex, 0), 0)
    p = pos[:, None] + tl.arange(0, BM)[None, :]
    pad = (p >= vend[:, None]) | (~inb[:, None])
    tl.store(sorted_ptr + p, n, mask=pad & (p < max_pad) & jm[:, None])
    # zero the other workspace half (the next call's counts and cursors)
    for off in range(b * ZB, ZN, tl.num_programs(0) * ZB):
        zo = off + tl.arange(0, ZB)
        tl.store(z_ptr + zo, tl.zeros((ZB,), dtype=tl.int32), mask=zo < ZN)


def _use_triton_align(topk_ids, block_size, num_experts) -> bool:
    return (
        _TRITON_ALIGN
        and _TRITON_ALIGN_MIN_NUMEL <= topk_ids.numel() < 2**30
        and topk_ids.is_cuda
        and topk_ids.is_contiguous()
        and topk_ids.dtype in (torch.int32, torch.int64)
        # the cached launchers are specialized on 16-byte aligned pointers
        and topk_ids.data_ptr() % 16 == 0
        and 16 <= num_experts <= 1024
        and num_experts & (num_experts - 1) == 0
        and 16 <= block_size <= 256
        and block_size & (block_size - 1) == 0
        and not torch.cuda.is_current_stream_capturing()
    )


class _AlignState:
    pass


def _triton_moe_align_first(topk_ids, block_size, num_experts, sorted_ids, expert_ids,
                            num_tokens_post_pad, dev, stream, nb, grid) -> None:
    """kernel[grid] launches; caches the workspace and the compiled launchers."""
    ng, zn = _TRITON_ALIGN_NG, 2 * _TRITON_ALIGN_NG * num_experts
    ws = _TRITON_ALIGN_STATE.get((dev, stream, num_experts))
    if ws is None:
        # two halves of counts[ng, E] + cursors[ng, E]
        ws = _AlignState()
        ws.buf = torch.zeros((2, zn), dtype=torch.int32, device=topk_ids.device)
        ws.ptr = (ws.buf[0].data_ptr(), ws.buf[1].data_ptr())
        ws.i = 0
        _TRITON_ALIGN_STATE[(dev, stream, num_experts)] = ws
    i = ws.i
    ws.i = 1 - i
    half = ws.buf[i]
    log2e = num_experts.bit_length() - 1
    ck = _moe_align_count[(nb,)](
        topk_ids, half, topk_ids.numel(), E=num_experts, BLOCK=_TRITON_ALIGN_BLOCK,
        NG=ng, num_warps=4,
    )
    pk = _moe_align_place[(grid,)](
        topk_ids, half, half[ng * num_experts:], sorted_ids, expert_ids,
        num_tokens_post_pad, ws.buf[1 - i], topk_ids.numel(), nb, sorted_ids.numel(),
        expert_ids.numel(), E=num_experts, LOG2E=log2e, BLOCK=_TRITON_ALIGN_BLOCK,
        NG=ng, BM=block_size, BJ=_TRITON_ALIGN_BJ, ZN=zn, ZB=_TRITON_ALIGN_ZB,
        num_warps=4,
    )
    try:
        st = _AlignState()
        st.ws, st.log2e, st.zn, st.cur_off = ws, log2e, zn, ng * num_experts * 4
        st.count, st.count_fn, st.count_md = ck.run, ck.function, ck.packed_metadata
        st.place, st.place_fn, st.place_md = pk.run, pk.function, pk.packed_metadata
    except AttributeError:
        return  # other Triton: keep launching through kernel[grid]
    _TRITON_ALIGN_STATE[(dev, stream, num_experts, block_size, topk_ids.dtype)] = st


def _triton_moe_align(topk_ids, block_size, num_experts, sorted_ids, expert_ids,
                      num_tokens_post_pad) -> None:
    n = topk_ids.numel()
    dev = topk_ids.get_device()
    stream = _align_raw_stream(dev)
    nb = (n + _TRITON_ALIGN_BLOCK - 1) // _TRITON_ALIGN_BLOCK
    grid = max(nb, (expert_ids.numel() + _TRITON_ALIGN_BJ - 1) // _TRITON_ALIGN_BJ)
    st = _TRITON_ALIGN_STATE.get((dev, stream, num_experts, block_size, topk_ids.dtype))
    if st is None:
        _triton_moe_align_first(
            topk_ids, block_size, num_experts, sorted_ids, expert_ids,
            num_tokens_post_pad, dev, stream, nb, grid,
        )
        return
    # direct launches with the arguments kernel[grid] would bind (constexprs
    # included, ignored by the launcher); same stream order as the op
    ws = st.ws
    i = ws.i
    ws.i = 1 - i
    w = ws.ptr[i]
    st.count(
        nb, 1, 1, stream, st.count_fn, st.count_md, None, None, None,
        topk_ids, w, n, num_experts, _TRITON_ALIGN_BLOCK, _TRITON_ALIGN_NG,
    )
    st.place(
        grid, 1, 1, stream, st.place_fn, st.place_md, None, None, None,
        topk_ids, w, w + st.cur_off, sorted_ids, expert_ids, num_tokens_post_pad,
        ws.ptr[1 - i], n, nb, sorted_ids.numel(), expert_ids.numel(), num_experts,
        st.log2e, _TRITON_ALIGN_BLOCK, _TRITON_ALIGN_NG, block_size, _TRITON_ALIGN_BJ,
        st.zn, _TRITON_ALIGN_ZB,
    )
'''

OLD_CALL = '''    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )
'''

NEW_CALL = '''    if expert_map is None and _use_triton_align(topk_ids, block_size, num_experts):
        _triton_moe_align(
            topk_ids, block_size, num_experts, sorted_ids, expert_ids,
            num_tokens_post_pad,
        )
    else:
        ops.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
        )
'''

MARKER = "Zenith patch rocm_moe_align_triton.py"
EDITS = (
    (OLD_IMPORTS, NEW_IMPORTS),
    (OLD_CALL, NEW_CALL),
)


def find_target():
    """Locate the file in the installed vLLM without importing vllm."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    path = os.path.join(root, TARGET)
    if not os.path.exists(path):
        sys.exit(f"expected {path} to exist; vLLM layout changed")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--revert", action="store_true",
                    help="restore the image's moe_align_block_size.py")
    ap.add_argument("--path", help="patch this file instead of the installed "
                    "vLLM's (e.g. a copy bind-mounted over the image file)")
    args = ap.parse_args()

    path = args.path or find_target()
    with open(path) as fh:
        src = fh.read()

    if args.revert:
        if MARKER not in src:
            print(f"not patched: {path}")
            return 0
        if any(src.count(new) != 1 for _, new in EDITS):
            print(f"ERROR: {path} carries a different version of this patch; "
                  "restore moe_align_block_size.py from the image", file=sys.stderr)
            return 1
        for old, new in EDITS:
            src = src.replace(new, old, 1)
        with open(path, "w") as fh:
            fh.write(src)
        print(f"reverted: {path}")
        return 0

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    missing = [i for i, (old, _) in enumerate(EDITS) if src.count(old) != 1]
    if missing:
        print(f"ERROR: patch target(s) {missing} not found exactly once in {path}",
              file=sys.stderr)
        print("moe_align_block_size() changed upstream; re-derive this patch",
              file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    for old, new in EDITS:
        src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"patched: {path}  (ROCm Triton moe_align_block_size at prefill sizes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
