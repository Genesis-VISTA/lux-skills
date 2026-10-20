"""
Modular fused MoE on ROCm: let TritonExperts write straight into the
caller's output buffer, as it already does on CUDA.

THE PROBLEM

FusedMoEKernel._fused_experts aliases fused_out to the caller's output
buffer when shape, dtype and device match, so that
TopKWeightAndReduceNoOP.apply can skip its copy. On ROCm the alias is only
taken when AITER fused MoE is enabled. With VLLM_ROCM_USE_AITER_MOE=0, the
TritonExperts path therefore:
1. writes moe_sum into a workspace buffer;
2. copies that buffer into output with output.copy_(fused_out).

The copy is a full [M, H] read and write per MoE layer: about 110 us at
M=8192, H=4096 bf16 on MI250X.

THE FIX

On ROCm, also take the alias when the experts object is exactly
TritonExperts. TritonExperts.apply writes output in one place only, the
final moe_sum, after it has read hidden_states and its workspaces, so the
alias is safe. The copy that is skipped is exact, so outputs are unchanged.
Other experts classes keep the existing rule.
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/model_executor/layers/fused_moe/modular_kernel.py"

OLD = '''        if current_platform.is_rocm():
            from vllm._aiter_ops import rocm_aiter_ops

            if use_output_alias and rocm_aiter_ops.is_fused_moe_enabled():
                fused_out = output_alias
'''

NEW = '''        if current_platform.is_rocm():
            from vllm._aiter_ops import rocm_aiter_ops

            # Zenith patch rocm_triton_moe_output_alias.py: TritonExperts
            # writes output only in its final moe_sum, so aliasing is safe
            # and skips the NoOP finalize's [M, H] copy (~110 us at M=8192).
            from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
                TritonExperts,
            )

            if use_output_alias and (
                rocm_aiter_ops.is_fused_moe_enabled()
                or type(self.fused_experts) is TritonExperts
            ):
                fused_out = output_alias
'''

MARKER = "Zenith patch rocm_triton_moe_output_alias.py"


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
                    help="restore the original alias rule")
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
                  "restore modular_kernel.py from the image", file=sys.stderr)
            return 1
        with open(path, "w") as fh:
            fh.write(src.replace(NEW, OLD, 1))
        print(f"reverted: {path}")
        return 0

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    if src.count(OLD) != 1:
        print(f"ERROR: patch target not found exactly once in {path}",
              file=sys.stderr)
        print("vLLM changed the ROCm output-alias rule in _fused_experts; "
              "check whether TritonExperts on ROCm still pays the NoOP "
              "finalize copy, then re-derive or retire this patch",
              file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (ROCm TritonExperts writes into the output buffer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
