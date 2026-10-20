#!/usr/bin/env python3
"""
Dequantize only the routed experts in vLLM's OCP MX (MXFP4) emulation MoE, so
MXFP4 MoE models decode at a usable rate on gfx90a (MI250X).

THE PROBLEM
OCP_MXQuantizationEmulationTritonExperts.apply (the EMULATION backend, which
every MXFP4 MoE model gets on gfx90a: Muse Spark through muse_mxfp4_emulation,
Inkling-MXFP4 directly) dequantizes the packed weights of every local expert
to bf16 on every call:

    w1_dequant = self._dequantize_weights(w1, self.w1_scale_val, ...)
    w2_dequant = self._dequantize_weights(w2, self.w2_scale_val, ...)

For Muse Spark at TP8 (256 experts on its 4096-wide expert hidden,
moe_intermediate 4096) that writes 1.6 G bf16 values, 3.2 GB, per GCD per MoE
layer, and the Triton MoE reads them back, whatever the batch. At c=1 it is ~2.4 tok/s for TP8 x PP3 on
3 tioga nodes (job f639nzTjFcnK, 2026-10-16), and Inkling-MXFP4 TP8 x PP4 sat
at ~2.2 tok/s for the same reason (docs/MI250X_PORTABILITY.md).

THE FIX
Dequantize only the experts the batch routes to: gather their packed weights
and scales (torch.unique over topk_ids), dequantize those, and renumber
topk_ids onto the gathered experts. A decode step of one request touches 8 of
Muse's 256 experts per layer. The dequantized values are the same, and so is
the rest of the Triton MoE; only the expert count it sees changes (its tile
config lookup takes the gathered w1's shape). Skipped, i.e. the stock path:
under EP (expert_map), with LoRA, during graph capture (torch.unique syncs),
and when the batch routes to more than a quarter of the local experts: the
gather (PyTorch's indexing kernel, ~80 GB/s on MI250X) plus the dequant then
costs about what the full dequant does. Measured per Muse layer at TP8 on a
busy tioga GCD: full dequant 6.6 ms; gather + dequant 1.5 ms for 8 experts
(c=1), 2.3 ms for 31 (c=4), 6.2 ms for 100 (c=16), 15.5 ms for 225 (c=64).
The MoE output is bit-identical to the full path's (bf16 Triton MoE, 1-200
tokens).

Only gfx90a images are patched: the script is a no-op unless the image's
PYTORCH_ROCM_ARCH is gfx90a, so gfx950 (lux) and gfx942 images keep vLLM's
code byte for byte.

Usage (inside the container):
    python3 rocm_mxfp4_emulation_routed.py [--check]
"""

import argparse
import importlib.util
import os
import sys

TARGET = "vllm/model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py"
MARKER = "rocm_mxfp4_emulation_routed"

EDITS = [
    (
        "dequantize",
        '''        # Dequantize w1 and w2 from packed OCP MX format to bf16/fp16
        w1_dequant = self._dequantize_weights(
            w1, self.w1_scale_val, hidden_states.dtype
        )
        w2_dequant = self._dequantize_weights(
            w2, self.w2_scale_val, hidden_states.dtype
        )
''',
        '''        # Dequantize w1 and w2 from packed OCP MX format to bf16/fp16
        # zenith (rocm_mxfp4_emulation_routed): only the experts this batch
        # routes to, with topk_ids renumbered onto them. Dequantizing every
        # local expert writes the whole layer in bf16 on each call; past a
        # quarter of the experts the gather costs as much as it saves.
        w1_scale, w2_scale = self.w1_scale_val, self.w2_scale_val
        if (
            expert_map is None
            and getattr(self, "_lora_context", None) is None
            and not torch.cuda.is_current_stream_capturing()
        ):
            used, local_ids = torch.unique(topk_ids, return_inverse=True)
            if 0 < used.numel() <= w1.size(0) // 4:
                w1, w2 = w1[used], w2[used]
                w1_scale, w2_scale = w1_scale[used], w2_scale[used]
                topk_ids = local_ids.view_as(topk_ids).to(topk_ids.dtype)
                global_num_experts = used.numel()
        w1_dequant = self._dequantize_weights(w1, w1_scale, hidden_states.dtype)
        w2_dequant = self._dequantize_weights(w2, w2_scale, hidden_states.dtype)
''',
    ),
]


def vllm_root():
    """Locate vLLM without importing it (no GPU in a build container)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    args = ap.parse_args()

    # The image's arch, not the node's: this runs at build time too (no GPU).
    archs = {a.split(":")[0] for a in
             os.environ.get("PYTORCH_ROCM_ARCH", "").replace(",", ";").split(";") if a}
    if archs != {"gfx90a"}:
        print(f"not needed: image arch {sorted(archs) or 'unknown'} is not gfx90a")
        return 0
    path = os.path.join(vllm_root(), TARGET)
    if not os.path.exists(path):
        print(f"not needed: no {TARGET}")
        return 0
    with open(path) as fh:
        src = fh.read()
    if MARKER in src:
        print(f"already patched: {path}")
        return 0
    for name, old, _ in EDITS:
        if src.count(old) != 1:
            print(f"ERROR: anchor '{name}' found {src.count(old)} times in {path}",
                  file=sys.stderr)
            print("vLLM changed the emulation MoE; re-derive the patch", file=sys.stderr)
            return 1
    if args.check:
        print(f"would patch: {path}")
        return 0
    for _, old, new in EDITS:
        src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"patched: {path}  (MXFP4 emulation dequantizes the routed experts only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
