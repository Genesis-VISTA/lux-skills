"""
Inkling router top-k (inkling_gate_select) on ROCm: a row-blocked Triton
kernel instead of one 4-warp program per token.

THE PROBLEM

InklingGate.select_experts runs _inkling_gate_select_kernel once per MoE
layer: sigmoid(+bias) over the routed logits, iterative top-6 argmax, the
two sink logits appended, log-sigmoid renorm. Each token is a 4-warp
program over next_pow2(258) = 512 lanes (254 of them padding), and every
pick does two block-wide reductions (argmax, then a masked max to fetch the
raw logit), each a cross-warp LDS round trip. On MI250X (one GCD, CUDA-graph
replay) it takes 18 us at T=512, 32 us at T=1024, 211 us at T=8192 and
415 us at T=16384 for a 34 B/token read: 40 MoE layers x 211 us = 8.4 ms
per 8k prefill chunk per PP stage.

THE FIX

Same arithmetic, different shape: score only the R routed lanes
(BLOCK_R = next_pow2(R) = 256), give each program ROWS whole tokens on
num_warps=1 so both reductions stay inside one wavefront, and fetch the
K + S raw logits of the picks with a single gather load (the same fp32
elements the masked max returned). The renorm runs on the same [K + S]
vector with the same ops. ROWS=1 up to 1024 tokens, ROWS=2 above, from a
sweep of ROWS x num_warps on MI250X: 6.6 us at T=512, 8.7 us at 1024,
37 us at 8192, 69 us at 16384.

Output is bit-identical (ids and weights, torch.equal) to the served kernel
for every variant swept, on 405k rows (captured served-model router logits
of layers 2/11/20 at T=4..8192, random and tie-heavy rows), and the whole
InklingMoE layer output is bit-identical (ndiff=0, L11 and L20, T=512..16384).
Whole-layer time (L11, eager, t8k config): -3 us at T=512, -19 us at 1024,
-174 us at 8192, -349 us at 16384.
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/moe.py"

OLD = '''@triton.jit(do_not_specialize=["T", "route_scale"])
def _inkling_gate_select_kernel(
    logits_ptr,  # [T, G] fp32 gate logits (stride_logits_0 may include pad)
    bias_ptr,  # [R] fp32 selection bias (or 0 ptr if HAS_BIAS=False)
    global_scale_ptr,  # [1] fp32 (or unused if HAS_GSCALE=False)
    ids_ptr,  # [T, K + S] int32 out: selected expert ids
    weights_ptr,  # [T, K + S] fp32 out: renormalized weights
    route_scale,
    T,
    G: tl.constexpr,  # total gate experts (routed + shared)
    stride_logits_0,
    R: tl.constexpr,  # routed experts
    K: tl.constexpr,  # top-k routed
    S: tl.constexpr,  # shared (sink) experts
    HAS_BIAS: tl.constexpr,
    HAS_GSCALE: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs = tl.arange(0, BLOCK_G)
    mask_r = offs < R
    logits = tl.load(
        logits_ptr + pid * stride_logits_0 + offs,
        mask=offs < G,
        other=float("-inf"),
    ).to(tl.float32)

    # Selection scores: sigmoid(routed logits) (+ bias), non-routed lanes -inf.
    sel = tl.where(mask_r, tl.sigmoid(logits), float("-inf"))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs, mask=mask_r, other=0.0).to(tl.float32)
        sel = tl.where(mask_r, sel + bias, float("-inf"))

    scale = route_scale
    if HAS_GSCALE:
        scale = scale * tl.load(global_scale_ptr).to(tl.float32)

    # Iterative top-K (K is small); argmax tie-breaks to the lowest index
    # (stable ordering).
    A: tl.constexpr = K + S
    offs_a = tl.arange(0, A)
    top_ids = tl.zeros([A], dtype=tl.int32)
    active = tl.zeros([A], dtype=tl.float32)
    for kk in tl.static_range(K):
        idx = tl.argmax(sel, axis=0).to(tl.int32)
        raw = tl.max(tl.where(offs == idx, logits, float("-inf")), axis=0)
        top_ids = tl.where(offs_a == kk, idx, top_ids)
        active = tl.where(offs_a == kk, raw, active)
        sel = tl.where(offs == idx, float("-inf"), sel)
    if S > 0:
        # Shared sink logits sit at the tail of the gate output; their expert
        # ids continue after the routed range (R + j).
        for jj in tl.static_range(S):
            raw = tl.max(tl.where(offs == R + jj, logits, float("-inf")), axis=0)
            top_ids = tl.where(offs_a == K + jj, tl.full([], R + jj, tl.int32), top_ids)
            active = tl.where(offs_a == K + jj, raw, active)

    # Log-sigmoid renormalization over the K + S active logits.
    abs_l = tl.abs(active)
    min_l = tl.minimum(active, 0.0)
    log_probs = min_l - tldevice.log1p(tldevice.exp(-abs_l))
    max_lp = tl.max(log_probs, axis=0)
    exp_shifted = tldevice.exp(log_probs - max_lp)
    sum_exp = tl.sum(exp_shifted, axis=0)
    weights = exp_shifted / sum_exp * scale

    tl.store(ids_ptr + pid * A + offs_a, top_ids)
    tl.store(weights_ptr + pid * A + offs_a, weights)


def inkling_gate_select(
    logits: torch.Tensor,  # [T, >=G] fp32 (rows may carry GEMM padding)
    n_gate_experts: int,
    n_routed_experts: int,
    topk: int,
    n_shared_experts: int,
    bias: torch.Tensor | None,
    route_scale: float,
    global_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid + bias + top-k + log-sigmoid renorm; returns (weights, ids)."""
    assert logits.dtype == torch.float32
    tokens = logits.shape[0]
    active = topk + n_shared_experts
    topk_ids = torch.empty((tokens, active), dtype=torch.int32, device=logits.device)
    topk_weights = torch.empty(
        (tokens, active), dtype=torch.float32, device=logits.device
    )
    if tokens == 0:
        return topk_weights, topk_ids
    _inkling_gate_select_kernel[(tokens,)](
        logits,
        bias if bias is not None else logits,
        global_scale if global_scale is not None else logits,
        topk_ids,
        topk_weights,
        route_scale,
        tokens,
        n_gate_experts,
        logits.stride(0),
        n_routed_experts,
        topk,
        n_shared_experts,
        HAS_BIAS=bias is not None,
        HAS_GSCALE=global_scale is not None,
        BLOCK_G=triton.next_power_of_2(n_gate_experts),
    )
    return topk_weights, topk_ids
'''

NEW = '''# Zenith patch inkling_gate_select_rows.py: the served kernel ran one
# 4-warp program per token over next_pow2(G)=512 lanes and did two
# block-wide reductions per pick (217 us at T=8192 on MI250X). This one
# scores only the R routed lanes, gives each wavefront whole rows so the
# argmax reductions stay in-wave, and fetches the K + S raw logits with one
# gather load. Same arithmetic per element and the same tie-break, so ids
# and weights are bit-identical to the served kernel.
@triton.jit(do_not_specialize=["T", "route_scale"])
def _inkling_gate_select_rows_kernel(
    logits_ptr,  # [T, G] fp32 gate logits (stride_logits_0 may include pad)
    bias_ptr,  # [R] fp32 selection bias (or 0 ptr if HAS_BIAS=False)
    global_scale_ptr,  # [1] fp32 (or unused if HAS_GSCALE=False)
    ids_ptr,  # [T, K + S] int32 out: selected expert ids
    weights_ptr,  # [T, K + S] fp32 out: renormalized weights
    route_scale,
    T,
    stride_logits_0,
    R: tl.constexpr,  # routed experts
    K: tl.constexpr,  # top-k routed
    S: tl.constexpr,  # shared (sink) experts
    HAS_BIAS: tl.constexpr,
    HAS_GSCALE: tl.constexpr,
    BLOCK_R: tl.constexpr,  # next_pow2(R)
    ROWS: tl.constexpr,  # tokens per program
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
    rmask = rows < T
    offs = tl.arange(0, BLOCK_R)
    mask_r = offs < R
    lrow = logits_ptr + rows[:, None] * stride_logits_0
    logits = tl.load(
        lrow + offs[None, :],
        mask=rmask[:, None] & mask_r[None, :],
        other=float("-inf"),
    ).to(tl.float32)

    # Selection scores: sigmoid(routed logits) (+ bias), padding lanes -inf.
    sel = tl.where(mask_r[None, :], tl.sigmoid(logits), float("-inf"))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs, mask=mask_r, other=0.0).to(tl.float32)
        sel = tl.where(mask_r[None, :], sel + bias[None, :], float("-inf"))

    scale = route_scale
    if HAS_GSCALE:
        scale = scale * tl.load(global_scale_ptr).to(tl.float32)

    # Iterative top-K (K is small); argmax tie-breaks to the lowest index
    # (stable ordering).
    A: tl.constexpr = K + S
    offs_a = tl.arange(0, A)
    top_ids = tl.zeros([ROWS, A], dtype=tl.int32)
    for kk in tl.static_range(K):
        idx = tl.argmax(sel, axis=1).to(tl.int32)
        top_ids = tl.where(offs_a[None, :] == kk, idx[:, None], top_ids)
        sel = tl.where(offs[None, :] == idx[:, None], float("-inf"), sel)
    if S > 0:
        # Shared sink logits sit at the tail of the gate output; their expert
        # ids continue after the routed range (R + j).
        for jj in tl.static_range(S):
            top_ids = tl.where(offs_a[None, :] == K + jj, R + jj, top_ids)
    active = tl.load(lrow + top_ids, mask=rmask[:, None], other=0.0).to(tl.float32)

    # Log-sigmoid renormalization over the K + S active logits.
    abs_l = tl.abs(active)
    min_l = tl.minimum(active, 0.0)
    log_probs = min_l - tldevice.log1p(tldevice.exp(-abs_l))
    max_lp = tl.max(log_probs, axis=1)
    exp_shifted = tldevice.exp(log_probs - max_lp[:, None])
    sum_exp = tl.sum(exp_shifted, axis=1)
    weights = exp_shifted / sum_exp[:, None] * scale

    out = rows[:, None] * A + offs_a[None, :]
    tl.store(ids_ptr + out, top_ids, mask=rmask[:, None])
    tl.store(weights_ptr + out, weights, mask=rmask[:, None])


# (max tokens, ROWS, num_warps), first match; MI250X sweep, every variant
# bit-identical.
_GATE_SELECT_LAUNCH = ((1024, 1, 1), (float("inf"), 2, 1))


def inkling_gate_select(
    logits: torch.Tensor,  # [T, >=G] fp32 (rows may carry GEMM padding)
    n_gate_experts: int,
    n_routed_experts: int,
    topk: int,
    n_shared_experts: int,
    bias: torch.Tensor | None,
    route_scale: float,
    global_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid + bias + top-k + log-sigmoid renorm; returns (weights, ids)."""
    assert logits.dtype == torch.float32
    tokens = logits.shape[0]
    active = topk + n_shared_experts
    topk_ids = torch.empty((tokens, active), dtype=torch.int32, device=logits.device)
    topk_weights = torch.empty(
        (tokens, active), dtype=torch.float32, device=logits.device
    )
    if tokens == 0:
        return topk_weights, topk_ids
    rows, num_warps = next(
        (r, w) for t, r, w in _GATE_SELECT_LAUNCH if tokens <= t
    )
    _inkling_gate_select_rows_kernel[(triton.cdiv(tokens, rows),)](
        logits,
        bias if bias is not None else logits,
        global_scale if global_scale is not None else logits,
        topk_ids,
        topk_weights,
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
        num_warps=num_warps,
    )
    return topk_weights, topk_ids
'''


MARKER = "Zenith patch inkling_gate_select_rows.py"


def find_target():
    """Locate moe.py in the installed vLLM without importing vllm (importing
    it touches the GPU at module scope)."""
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
                    help="restore the original inkling_gate_select")
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
        if NEW not in src:
            print(f"ERROR: {path} carries a different version of this patch; "
                  "restore moe.py from the image", file=sys.stderr)
            return 1
        with open(path, "w") as fh:
            fh.write(src.replace(NEW, OLD, 1))
        print(f"reverted: {path}")
        return 0

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    if OLD not in src:
        print(f"ERROR: patch target not found in {path}", file=sys.stderr)
        print("vLLM changed _inkling_gate_select_kernel / inkling_gate_select; "
              "re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (Inkling router top-k -> row-blocked Triton)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
