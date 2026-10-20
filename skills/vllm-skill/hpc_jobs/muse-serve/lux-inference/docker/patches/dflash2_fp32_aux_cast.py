#!/usr/bin/env python3
"""
Cast the target's aux hidden states to the DFlash2 draft's dtype before its fc.

THE PROBLEM
Muse Spark keeps its residual stream in FP32 (vllm/models/muse_spark/amd/
model.py, residual_dtype = torch.float32), so the aux hidden states it hands a
speculator are FP32. The DFlash2 drafts for Muse (RadixArk/Muse-Spark-Open-
DFlash2, nvidia/Muse-Spark-Open-816B-A42B-DFlash2; architecture
DFlash2DraftModel -> qwen3_dflash2.DFlash2Qwen3ForCausalLM) are BF16, and
DFlashQwen3ForCausalLM.combine_hidden_states feeds the concatenated states
straight into the BF16 fc. Startup dies in the profile run:

    qwen3_dflash.py combine_hidden_states -> self.model.fc(hidden_states)
    RuntimeError: expected mat1 and mat2 to have the same dtype,
                  but got: float != c10::BFloat16

Muse's own DSpark draft (vllm/models/muse_spark/common/dspark.py) already
casts in its combine_hidden_states ("Target residuals are FP32; the training
collator casts features before FC"). DFlash2 has no such override.

Seen on lux (Muse-Spark-Open-816B-A42B-Quark-MXFP4, EP8, vLLM 0e939af8, jobs
350007 and 350008, 2026-10-18), with both DFlash2 drafts.

THE FIX
Give DFlash2Qwen3ForCausalLM the same override: cast to fc.weight.dtype, then
call the parent. For a BF16 target the cast is a no-op. Nothing else uses
DFlash2Qwen3ForCausalLM, so this is inert unless a DFlash2 draft is served.

Usage (inside the container):
    python3 dflash2_fp32_aux_cast.py [--check]
"""

import argparse
import importlib.util
import os
import sys

TARGET = "vllm/model_executor/models/qwen3_dflash2.py"

OLD = '''    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )
'''

NEW = OLD + '''
    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # zenith: an FP32-residual target (Muse Spark) hands over FP32 aux
        # states; cast before the draft's fc (dflash2_fp32_aux_cast).
        return super().combine_hidden_states(
            hidden_states.to(self.model.fc.weight.dtype)
        )
'''

MARKER = "dflash2_fp32_aux_cast"


def find_target():
    """Locate vLLM without importing it (no GPU in a build container)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    return os.path.join(root, TARGET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    args = ap.parse_args()

    path = find_target()
    if not os.path.exists(path):
        print(f"not needed: {path} does not exist (no DFlash2 draft model)")
        return 0
    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"already patched: {path}")
        return 0
    if "def combine_hidden_states" in src:
        print(f"not needed: {path} already overrides combine_hidden_states")
        return 0
    if OLD not in src:
        print(f"ERROR: patch target not found in {path}", file=sys.stderr)
        print("vLLM changed DFlash2Qwen3ForCausalLM; re-derive the patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (FP32 aux hidden states cast for the DFlash2 fc)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
