#!/usr/bin/env python3
"""
Bound the KV loop of Inkling's single-pass attention kernel to the keys its
queries can see: the sliding window on local layers, and the causal limit.

THE BUG AS IT PRESENTS

With MTP on, every decode step is a verify of 1 + num_spec query tokens, so
use_split_kv_decode (which needs max_query_len == 1) never applies and all 42
layers run the single-pass _inkling_rel_attention_kernel. For the 35
sliding-window layers (window 512) that kernel still walks the whole context:
an 8k-token request costs ~130 64-key tiles per layer when at most 9 hold a
visible key. The same kernel runs every prefill chunk, where each 4-query
program also walks keys past its own causal limit.

On gfx90a at TP8 the kernel launches one program per (query block, KV head,
request), and there is one KV head per rank. At small batch a handful of
programs walk the whole context serially, so local-layer latency grows with
context length rather than with the 512-token window.

WHY

The loop is `for k_start in range(0, kv_len, BLOCK_K)`, and tiles outside the
window or past the causal limit are only masked to -inf. They are still loaded
and multiplied, contributing exp2(-inf) = 0.

THE FIX

Start the loop at the first tile any query in the block can see
(prefix_len + q_block - window_left on window layers, else 0), rounded down to
a tile. Stop it after the last block query's own position
(prefix_len + min(q_block + BLOCK_Q, q_len)). The skipped tiles were fully
masked, so the online softmax is unchanged: every score in them was -inf,
which gives alpha = 1 and p = 0. The kernel's existing has_scores guard already
covers leading tiles that are entirely masked. No tiles are reordered; only
fully masked tiles are dropped.

EVIDENCE

Tioga, 1 GCD, Inkling-Small TP8 shapes (4 q heads, 1 KV head, D 128, page 32),
qlen 4 (MTP 3 verify), context 8192-9216. Graph replay, us per call (gbench
job f61hkVS7bnSs, attn_q4_before.log / attn_q4_after.log):

    layer            B=1    B=8    B=32   B=128
    local  before    605    721    744    996
    local  after      51     55     65     80
    global before    641    687    703    903
    global after     647    692    709    915   (no window; unchanged)

The output of every batch size, local and global, matches a dump taken
before the patch exactly (max|diff| 0). Global layers keep walking the whole
context. They need a split-KV path for qlen > 1, which this does not add.

End to end, 2-node TP8 x PP2 eager MTP 3 (f61hkVYagek7 -> f61iLb3sSZKd),
tok/s/user goes from 20.6 to 26.1 at c=1. The correctness gate passes 30/30,
and MTP acceptance is unchanged (0.385 -> 0.384 over ~1.26M drafted tokens).
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/models/inkling/amd/ops/fa4_rel_attention.py"

# the exact loop header we expect to find (upstream, unpatched)
OLD = '''    for k_start in range(0, kv_len, BLOCK_K):
        k_pos = k_start + tl.arange(0, BLOCK_K)
'''

NEW = '''    # Only walk tiles some query of this block can see: from the window start
    # of its first query (local layers) to the causal limit of its last one.
    # Every skipped tile was fully masked.
    # Zenith patch inkling_single_pass_kv_bounds.py.
    k_lo = 0
    if window_left >= 0:
        k_lo = tl.maximum(prefix_len + q_block - window_left, 0) // BLOCK_K * BLOCK_K
    k_hi = tl.minimum(kv_len, prefix_len + tl.minimum(q_block + BLOCK_Q, q_len))
    for k_start in range(k_lo, k_hi, BLOCK_K):
        k_pos = k_start + tl.arange(0, BLOCK_K)
'''

MARKER = "patch inkling_single_pass_kv_bounds.py"


def find_target():
    """Locate the attention module inside whichever site-packages vLLM lives in.

    Deliberately does NOT `import vllm` (see rocm_aiter_ray_env_refresh.py):
    importing it touches the GPU at module scope and fails in a build container.
    """
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
    ap.add_argument("--path", help="patch this file instead of the installed "
                    "vLLM's (e.g. a copy bind-mounted over the image file)")
    args = ap.parse_args()

    path = args.path or find_target()
    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    if OLD not in src:
        print(f"ERROR: patch target not found in {path}", file=sys.stderr)
        print("vLLM changed _inkling_rel_attention_kernel's KV loop; check "
              "whether it still starts at 0 and runs to kv_len, then re-derive "
              "or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (single-pass KV loop bounded to window/causal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
