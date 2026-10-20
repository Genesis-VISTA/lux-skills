#!/usr/bin/env python3
"""
Decode-step MoE glue for Inkling on MI250X (gfx90a): the routed/sink join done
by the routed moe_sum, and split gate-select outputs.

THE COST AS IT PRESENTS

At decode T (num_tokens <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD) each
MoE layer runs, besides the routed GEMMs and the sink branch on the aux
stream, a chain of small kernels (eager profile, T=512, us):
  gate select 13.6, two slice copies of its [T, K + S] outputs 6.6 + 6.4,
  moe_sum_vec 31.4, _add_to_slabs 12.3 (re-reads the [T, H] routed sum and
  the sink output to write the rank-major slabs for the sconv RS).

THE FIX (each switch default off = the served path)

- INKLING_DEC_MOE_JOIN=1: at decode T the sink branch is launched on the aux
  stream first and the routed moe_sum (the Triton kernel of
  rocm_triton_moe_sum.py) waits on its event and adds it in the same pass,
  writing the rank-major (tp, T, C) slabs (the xrsag direct buffer when that
  path is on). Same values as moe_sum + add_to_slabs: fp32 sum in top-k
  order rounded to bf16, then bf16(bf16(sum) + sink). Falls back to the
  served join if the moe_sum does not take the addend.
  INKLING_DEC_MOE_JOIN_SUM=BLOCK_M,BLOCK_N,num_warps (default 1,512,2) is the
  sum kernel tile below MOE_SUM_ADDEND_MIN_TOKENS rows (any tile computes the
  same per-element expression).
- INKLING_GATE_SELECT_SPLIT=1: the gate-select kernel stores the routed
  [T, K] ids / weights and the [T, S] sink gammas contiguously (the values
  of the [T, K + S] outputs), so forward drops the two .contiguous() copies.

Unit test (zv7 dec_zj_test, f6216N4FoH5h / f621NQQdV8k7), graph replay, us
per layer, served -> patched, every output torch.equal:
  join (moe_sum + add_to_slabs -> one sum):  T=64 6.2 -> 3.3, 128 10.3 -> 6.1,
    256 21.2 -> 16.3, 512 42.1 -> 30.4, 528 43.2 -> 31.7, 768 62.5 -> 44.7,
    1024 82.6 -> 59.0
  gate select + 2 copies -> split:  T=64 9.5 -> 5.1, 256 11.6 -> 6.6,
    512 12.0 -> 6.9, 1024 14.1 -> 9.0
Stage-1 20-layer graph fwd (dec_nat, TP8, same node vs base), ms:
  JOIN          T=256 -0.19..-0.22, T=512 -0.33..-0.44
  JOIN + SPLIT  T=256 -0.40,        T=512 -0.60
Tried and dropped (zv7 dec jobs 26-27): the Triton moe_align on a zeroed
workspace in graphs (slower than the _moe_C op at T <= 768) and the fused
GEMM1 + SiLU-and-mul at decode tiles (no overlay that fits LDS beats GEMM1 +
act_and_mul).

Apply after rocm_triton_moe_sum.py, inkling_gate_select_rows.py,
inkling_moe_sink_fused_sum.py and inkling_sconv_zero_copy_xrsag.py
(alphabetical order in docker/patches does this).

    python rocm_triton_moe_zdec_join.py [--check] [--revert] [--root DIR]
"""
import argparse
import importlib.util
import os
import sys

MARKER = "patch rocm_triton_moe_zdec_join.py"
D = "vllm/models/inkling/amd/"
FM = "vllm/model_executor/layers/fused_moe/"

# ---------------------------------------------------------------- moe.py
GS_SIG_OLD = '''    ids_ptr,  # [T, K + S] int32 out: selected expert ids
    weights_ptr,  # [T, K + S] fp32 out: renormalized weights
    route_scale,
'''
GS_SIG_NEW = '''    ids_ptr,  # [T, K + S] int32 out: selected expert ids
    weights_ptr,  # [T, K + S] fp32 out: renormalized weights
    gam_ptr,  # (patch rocm_triton_moe_zdec_join.py) [T, S] fp32 if SPLIT
    route_scale,
'''
GS_SIG2_OLD = '''    ROWS: tl.constexpr,  # tokens per program
):
'''
GS_SIG2_NEW = '''    ROWS: tl.constexpr,  # tokens per program
    SPLIT: tl.constexpr,
):
'''
GS_STORE_OLD = '''    out = rows[:, None] * A + offs_a[None, :]
    tl.store(ids_ptr + out, top_ids, mask=rmask[:, None])
    tl.store(weights_ptr + out, weights, mask=rmask[:, None])
'''
GS_STORE_NEW = '''    if SPLIT:
        # (patch rocm_triton_moe_zdec_join.py) routed ids / weights as
        # contiguous [T, K], sink gammas as [T, S]: the same values.
        kcol = offs_a[None, :] < K
        out_k = rows[:, None] * K + offs_a[None, :]
        tl.store(ids_ptr + out_k, top_ids, mask=rmask[:, None] & kcol)
        tl.store(weights_ptr + out_k, weights, mask=rmask[:, None] & kcol)
        out_s = rows[:, None] * S + (offs_a[None, :] - K)
        tl.store(gam_ptr + out_s, weights, mask=rmask[:, None] & (offs_a[None, :] >= K))
    else:
        out = rows[:, None] * A + offs_a[None, :]
        tl.store(ids_ptr + out, top_ids, mask=rmask[:, None])
        tl.store(weights_ptr + out, weights, mask=rmask[:, None])
'''
GS_LAUNCH_OLD = '''        topk_weights,
        route_scale,
        tokens,
'''
GS_LAUNCH_NEW = '''        topk_weights,
        topk_weights,  # (patch rocm_triton_moe_zdec_join.py) unused
        route_scale,
        tokens,
'''
GS_LAUNCH2_OLD = '''        ROWS=rows,
        num_warps=num_warps,
    )
    return topk_weights, topk_ids
'''
GS_LAUNCH2_NEW = '''        ROWS=rows,
        SPLIT=False,
        num_warps=num_warps,
    )
    return topk_weights, topk_ids


# (patch rocm_triton_moe_zdec_join.py) INKLING_GATE_SELECT_SPLIT=1: forward
# takes the routed slice and the sink gammas straight from the kernel.
_GATE_SELECT_SPLIT = __import__("os").environ.get("INKLING_GATE_SELECT_SPLIT", "0") == "1"


def inkling_gate_select_split(
    logits: torch.Tensor,
    n_gate_experts: int,
    n_routed_experts: int,
    topk: int,
    n_shared_experts: int,
    bias: torch.Tensor | None,
    route_scale: float,
    global_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """inkling_gate_select as contiguous (routed weights [T, K], routed ids
    [T, K], sink gammas [T, S])."""
    assert logits.dtype == torch.float32
    tokens = logits.shape[0]
    dev = logits.device
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=dev)
    weights = torch.empty((tokens, topk), dtype=torch.float32, device=dev)
    gammas = torch.empty((tokens, n_shared_experts), dtype=torch.float32, device=dev)
    if tokens == 0:
        return weights, ids, gammas
    rows, num_warps = next(
        (r, w) for t, r, w in _GATE_SELECT_LAUNCH if tokens <= t
    )
    _inkling_gate_select_rows_kernel[(triton.cdiv(tokens, rows),)](
        logits,
        bias if bias is not None else logits,
        global_scale if global_scale is not None else logits,
        ids,
        weights,
        gammas,
        route_scale,
        tokens,
        logits.stride(0),
        n_routed_experts,
        topk,
        n_shared_experts,
        HAS_BIAS=bias is not None,
        HAS_GSCALE=global_scale is not None,
        BLOCK_R=triton.next_power_of_2(n_routed_experts),
        ROWS=rows,
        SPLIT=True,
        num_warps=num_warps,
    )
    return weights, ids, gammas
'''
SEL_OLD = '''            self.route_scale,
            self.global_scale,
        )


# ---------------------------------------------------------------------------
# MoE layer
'''
SEL_NEW = '''            self.route_scale,
            self.global_scale,
        )

    def select_experts_split(
        self, gating_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(patch rocm_triton_moe_zdec_join.py) select_experts as contiguous
        (routed weights [T, K], routed ids [T, K], sink gammas [T, S])."""
        return inkling_gate_select_split(
            gating_output,
            self.n_total_experts,
            self.n_routed_experts,
            self.topk,
            self.n_shared_experts,
            self.bias,
            self.route_scale,
            self.global_scale,
        )


# ---------------------------------------------------------------------------
# MoE layer
'''
JOIN_OLD = '''def _inkling_moe_ep_size() -> int:
'''
JOIN_NEW = '''# (patch rocm_triton_moe_zdec_join.py) INKLING_DEC_MOE_JOIN=1: at decode T
# the routed moe_sum adds the aux-stream sink output (after its event) and
# writes the rank-major slabs.
_DEC_MOE_JOIN = __import__("os").environ.get("INKLING_DEC_MOE_JOIN", "0") == "1"


def _forward_sink_join_dec(
    layer, x: torch.Tensor, gammas: torch.Tensor, router_logits: torch.Tensor,
) -> torch.Tensor | None:
    """InklingMoE.forward's decode join (sink branch on the aux stream) with
    the sink add and the slab layout done by the routed moe_sum. Returns the
    (tp, T, C) slabs the served add_to_slabs returns, or None (not taken:
    nothing launched)."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.model_executor.layers.fused_moe.experts import triton_moe

    slot = getattr(triton_moe, "MOE_SUM_ADDEND", None)
    tp = get_tensor_model_parallel_world_size()
    t, h = x.shape
    if (
        slot is None
        or not getattr(triton_moe, "MOE_SUM_ADDEND_EVENT", False)
        or layer._sink_stream is None
        or BreakableCUDAGraphCapture.is_active()
        or not (_SCONV_ZC and tp > 1 and h % tp == 0)
        or x.dtype != torch.bfloat16
    ):
        return None
    ev0, ev1 = layer._sink_events
    ev0.record()
    with torch.cuda.stream(layer._sink_stream):
        ev0.wait()
        sink_out = layer.sink_experts(x, gammas)
        ev1.record()
    slabs = None
    try:
        from .ops import xrsag

        slabs = xrsag.direct_slabs(tp, t, h // tp, x.dtype)
    except ImportError:
        pass
    if slabs is None:
        slabs = x.new_empty((tp, t, h // tp))
    slot[0] = (sink_out, slabs, ev1)
    try:
        out = layer.experts(hidden_states=x, router_logits=router_logits)
    finally:
        fused = slot[0] is None
        slot[0] = None
    layer._routed_sel = None
    if fused:
        return slabs
    ev1.wait()
    if (
        out.dim() == 2
        and out.shape == sink_out.shape
        and out.stride(1) == 1
        and sink_out.stride(1) == 1
    ):
        return add_to_slabs(out, sink_out, tp, dst=slabs)
    return out.add_(sink_out)


def _inkling_moe_ep_size() -> int:
'''
FWD_OLD = '''        k = self.gate.topk
        weights, ids = self.gate.select_experts(router_logits)
        self._routed_sel = (
            router_logits,
            weights[:, :k].contiguous(),
            ids[:, :k].contiguous(),
        )
        gammas = weights[:, k:]
'''
FWD_NEW = '''        k = self.gate.topk
        if _GATE_SELECT_SPLIT:
            # (patch rocm_triton_moe_zdec_join.py) contiguous kernel outputs
            weights_k, ids_k, gammas = self.gate.select_experts_split(router_logits)
            self._routed_sel = (router_logits, weights_k, ids_k)
        else:
            weights, ids = self.gate.select_experts(router_logits)
            self._routed_sel = (
                router_logits,
                weights[:, :k].contiguous(),
                ids[:, :k].contiguous(),
            )
            gammas = weights[:, k:]
'''
FWD2_OLD = '''            return _forward_sink_fused_sum(self, x, gammas, router_logits, slot)

        out, sink_out = maybe_execute_in_parallel(
'''
FWD2_NEW = '''            return _forward_sink_fused_sum(self, x, gammas, router_logits, slot)
        if (
            _DEC_MOE_JOIN
            and num_tokens <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        ):
            # (patch rocm_triton_moe_zdec_join.py)
            joined = _forward_sink_join_dec(self, x, gammas, router_logits)
            if joined is not None:
                return joined

        out, sink_out = maybe_execute_in_parallel(
'''

# --------------------------------------------------------- triton_moe.py
SUM_OLD = '''        if _moe_sum_triton_ok(input, output):
            add, slabs = _moe_sum_take_addend(input, output)
            _moe_sum_triton(input, output, add, slabs)
            return
        ops.moe_sum(input, output)
'''
SUM_NEW = '''        if _moe_sum_triton_ok(input, output):
            add, slabs = _moe_sum_take_addend(input, output)
            # (patch rocm_triton_moe_zdec_join.py) below the size floor only
            # for a decode join request
            if add is not None or input.shape[0] >= MOE_SUM_ADDEND_MIN_TOKENS:
                _moe_sum_triton(input, output, add, slabs)
                return
        ops.moe_sum(input, output)
'''
SUMK_OLD = '''    grid = (triton.cdiv(m, _MOE_SUM_BLOCK_M), triton.cdiv(n, _MOE_SUM_BLOCK_N))
    _moe_sum_kernel[grid](
        input,
        out,
        add if add is not None else output,
        m,
        N=n,
        SEG=seg,
        TOPK=topk,
        BLOCK_M=_MOE_SUM_BLOCK_M,
        BLOCK_N=_MOE_SUM_BLOCK_N,
        HAS_ADD=add is not None,
        num_warps=_MOE_SUM_WARPS,
    )
'''
SUMK_NEW = '''    bm, bn, nw = _MOE_SUM_BLOCK_M, _MOE_SUM_BLOCK_N, _MOE_SUM_WARPS
    if m < MOE_SUM_ADDEND_MIN_TOKENS:
        # (patch rocm_triton_moe_zdec_join.py) decode join tile
        bm, bn, nw = _MOE_SUM_DEC_TILE
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _moe_sum_kernel[grid](
        input,
        out,
        add if add is not None else output,
        m,
        N=n,
        SEG=seg,
        TOPK=topk,
        BLOCK_M=bm,
        BLOCK_N=bn,
        HAS_ADD=add is not None,
        num_warps=nw,
    )
'''
TAKE_OLD = '''    req = MOE_SUM_ADDEND[0]
    if req is None:
        return None, None
    add, slabs = req
'''
TAKE_NEW = '''    req = MOE_SUM_ADDEND[0]
    if req is None:
        return None, None
    # (patch rocm_triton_moe_zdec_join.py) (addend, slabs, event): the addend
    # is produced on another stream; wait on the event once taken.
    add, slabs = req[0], req[1]
    event = req[2] if len(req) > 2 else None
    if event is None and input.shape[0] < MOE_SUM_ADDEND_MIN_TOKENS:
        return None, None
'''
TAKE2_OLD = '''    MOE_SUM_ADDEND[0] = None
    return add, slabs
'''
TAKE2_NEW = '''    MOE_SUM_ADDEND[0] = None
    if event is not None:
        torch.cuda.current_stream().wait_event(event)
    return add, slabs
'''
OK_OLD = '''        and input.dim() == 3
        and input.shape[0] >= MOE_SUM_ADDEND_MIN_TOKENS
'''
OK_NEW = '''        and input.dim() == 3
        and (
            input.shape[0] >= MOE_SUM_ADDEND_MIN_TOKENS
            # (patch rocm_triton_moe_zdec_join.py) decode join request
            or (MOE_SUM_ADDEND[0] is not None and len(MOE_SUM_ADDEND[0]) > 2)
        )
'''
CONST_OLD = '''_MOE_SUM_BLOCK_N = 512
_MOE_SUM_WARPS = 2
'''
CONST_NEW = '''_MOE_SUM_BLOCK_N = 512
_MOE_SUM_WARPS = 2
# (patch rocm_triton_moe_zdec_join.py) MOE_SUM_ADDEND may also hold
# (addend, slabs, event) at any M: InklingMoE's decode join
# (INKLING_DEC_MOE_JOIN); the sum then waits on the event first.
import os as _zdec_os  # noqa: E402

MOE_SUM_ADDEND_EVENT = True
_MOE_SUM_DEC_TILE = tuple(
    int(v) for v in _zdec_os.environ.get("INKLING_DEC_MOE_JOIN_SUM", "1,512,2").split(",")
)
'''
TARGETS = [
    (D + "moe.py", [
        ("gate select signature", GS_SIG_OLD, GS_SIG_NEW),
        ("gate select constexprs", GS_SIG2_OLD, GS_SIG2_NEW),
        ("gate select stores", GS_STORE_OLD, GS_STORE_NEW),
        ("gate select launch", GS_LAUNCH_OLD, GS_LAUNCH_NEW),
        ("gate select launch tail", GS_LAUNCH2_OLD, GS_LAUNCH2_NEW),
        ("InklingGate.select_experts_split", SEL_OLD, SEL_NEW),
        ("_forward_sink_join_dec", JOIN_OLD, JOIN_NEW),
        ("forward gate select", FWD_OLD, FWD_NEW),
        ("forward decode join", FWD2_OLD, FWD2_NEW),
    ]),
    (FM + "experts/triton_moe.py", [
        ("moe_sum", SUM_OLD, SUM_NEW),
        ("_moe_sum_triton tile", SUMK_OLD, SUMK_NEW),
        ("_moe_sum_take_addend", TAKE_OLD, TAKE_NEW),
        ("_moe_sum_take_addend event", TAKE2_OLD, TAKE2_NEW),
        ("_moe_sum_triton_ok", OK_OLD, OK_NEW),
        ("moe_sum constants", CONST_OLD, CONST_NEW),
    ]),
]


def find_root():
    """site-packages dir holding vllm/, without importing vllm."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--revert", action="store_true",
                    help="undo the patch")
    ap.add_argument("--root", help="patch the vllm/ tree under this dir "
                    "instead of the installed vLLM's")
    args = ap.parse_args()

    root = args.root or find_root()
    plans = []
    problems = []
    for rel, sites in TARGETS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            problems.append(f"{path} missing; vLLM layout changed")
            continue
        with open(path) as fh:
            src = fh.read()
        if (MARKER in src) != args.revert:
            print(f"already {'reverted' if args.revert else 'patched'}: {path}")
            continue
        patched = src
        for name, old, new in (reversed(sites) if args.revert else sites):
            a, b = (new, old) if args.revert else (old, new)
            n = patched.count(a)
            if n != 1:
                problems.append(f"{rel}: {name} (matched {n}x, expected 1)")
                continue
            patched = patched.replace(a, b, 1)
        plans.append((path, patched))

    if problems:
        print("ERROR: patch target not found:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("Inkling's moe.py / triton_moe.py changed (are "
              "rocm_triton_moe_sum.py, inkling_gate_select_rows.py, "
              "inkling_moe_sink_fused_sum.py and inkling_sconv_zero_copy_xrsag.py "
              "applied first?); re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        for path, _ in plans:
            print(f"would {'revert' if args.revert else 'patch'}: {path}")
        return 0
    for path, patched in plans:
        with open(path, "w") as fh:
            fh.write(patched)
        print(f"{'reverted' if args.revert else 'patched'}: {path}  "
              "(Inkling decode MoE glue, INKLING_DEC_MOE_JOIN / "
              "INKLING_GATE_SELECT_SPLIT)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
