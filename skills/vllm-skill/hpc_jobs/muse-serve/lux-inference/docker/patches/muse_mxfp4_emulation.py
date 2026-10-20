#!/usr/bin/env python3
"""
Serve Muse Spark's Quark MXFP4 (W4A4) experts on GPUs without OCP MX hardware
(gfx90a / MI250X, gfx942) through vLLM's MXFP4 emulation backend.

THE PROBLEM
vllm/models/muse_spark/amd/muse_moe_runner.py::MuseSparkQuarkOCPMXFusedMoE
accepts only the AITER_MXFP4_MXFP4 backend:

    if self.mxfp4_backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
        raise NotImplementedError("MuseSpark1x MXFP4 on ROCm requires the "
                                  "AITER_MXFP4_MXFP4 backend; got EMULATION.")

AITER's MXFP4 MoE kernels are gfx950-only, so on tioga QuarkOCP_MX_MoEMethod
falls back to EMULATION and Muse dies at model construction. Muse also needs
experts that return one unweighted output per route (its post-expert RMSNorm
runs before the router weighting), which the stock emulation experts do not.

THE FIX
- Add MuseSparkMxfp4EmulationExperts: OCP_MXQuantizationEmulationTritonExperts
  (dequantize the packed weights per call, fake-quantize activations to MXFP4,
  run the bf16 Triton MoE) with the same route-output handoff and
  MuseSparkTopKWeightAndReduce finalize as MuseSparkTritonExperts, the BF16
  path that already serves layers 0 and 61 on any ROCm GPU.
- Select it when the oracle picked EMULATION; AITER_MXFP4_MXFP4 is unchanged
  and every other backend still raises.
- model.py writes the pre-expert RMSNorm once per route ([T * top_k, H]) for
  the AITER experts. The Triton experts take one row per token, so the
  expansion is skipped when the layer runs on EMULATION.

The numerics are the checkpoint's: MXFP4 weights and MXFP4-rounded
activations, computed in bf16.

Only gfx90a images are patched: the script is a no-op unless the image's
PYTORCH_ROCM_ARCH (build_image.sh sets it to the one target arch) is gfx90a,
so gfx950 (lux) and gfx942 images keep vLLM's Muse code byte for byte. It is
also a no-op on a vLLM without Muse support (the target files are absent).

Usage (inside the container):
    python3 muse_mxfp4_emulation.py [--check]
"""

import argparse
import importlib.util
import os
import sys

RUNNER = "vllm/models/muse_spark/amd/muse_moe_runner.py"
MODEL = "vllm/models/muse_spark/amd/model.py"
MARKER = "muse_mxfp4_emulation"

RUNNER_EDITS = [
    (
        "import",
        '''from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
''',
        '''from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
# zenith: MXFP4 emulation experts for GPUs without OCP MX (muse_mxfp4_emulation)
from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
    OCP_MXQuantizationEmulationTritonExperts,
)
''',
    ),
    (
        "experts class",
        '''class MuseSparkUnquantizedFusedMoE(UnquantizedFusedMoEMethod):
''',
        '''class MuseSparkMxfp4EmulationExperts(
    _RouteOutputHandoff, OCP_MXQuantizationEmulationTritonExperts
):
    """Emulated MXFP4 experts that expose unweighted GEMM2 output per route.

    zenith (muse_mxfp4_emulation): for GPUs without OCP MX hardware, where the
    AITER MXFP4 kernels do not run (gfx90a, gfx942).
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        if not isinstance(quant_config, MuseSparkQuarkOCPMXMoEQuantConfig):
            raise TypeError(
                "MuseSparkMxfp4EmulationExperts requires "
                "MuseSparkQuarkOCPMXMoEQuantConfig."
            )
        if quant_config.post_expert_norm_fn is None:
            raise ValueError("MuseSpark1x post-expert norm is not configured.")

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace13, workspace2, _ = super().workspace_shapes(
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )
        return workspace13, workspace2, (M, topk, K)

    def _apply_router_weight_in_gemm2(self, apply_router_weight_on_input: bool) -> bool:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "MuseSpark1x requires router weights after the post-expert norm."
            )
        return False

    def _finalize_expert_output(
        self,
        expert_output: torch.Tensor,
        output: torch.Tensor,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        # Under EP the GEMM skips non-local routes, leaving their rows unwritten.
        self._park_route_output(expert_output, expert_map)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        quant_config = self.quant_config
        assert isinstance(quant_config, MuseSparkQuarkOCPMXMoEQuantConfig)
        assert quant_config.post_expert_norm_fn is not None
        return MuseSparkTopKWeightAndReduce(
            quant_config.post_expert_norm_fn,
            pre_norm_all_reduce=self.moe_config.moe_parallel_config.tp_size > 1,
            take_route_output=self._take_route_output,
            take_route_expert_map=self._take_route_expert_map,
        )


class MuseSparkUnquantizedFusedMoE(UnquantizedFusedMoEMethod):
''',
    ),
    (
        "backend selection",
        '''        if self.mxfp4_backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
            raise NotImplementedError(
                "MuseSpark1x MXFP4 on ROCm requires the "
                "AITER_MXFP4_MXFP4 backend; got "
                f"{self.mxfp4_backend.value}."
            )
        self.experts_cls = MuseSparkAiterExperts
''',
        '''        if self.mxfp4_backend is Mxfp4MoeBackend.EMULATION:
            # zenith: no OCP MX hardware (muse_mxfp4_emulation)
            self.experts_cls = MuseSparkMxfp4EmulationExperts
            return
        if self.mxfp4_backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
            raise NotImplementedError(
                "MuseSpark1x MXFP4 on ROCm requires the "
                "AITER_MXFP4_MXFP4 backend; got "
                f"{self.mxfp4_backend.value}."
            )
        self.experts_cls = MuseSparkAiterExperts
''',
    ),
]

MODEL_EDITS = [
    (
        "expanded input",
        '''            and isinstance(
                self.experts.routed_experts.quant_method, MuseSparkQuarkOCPMXFusedMoE
            )
        )
''',
        '''            and isinstance(
                self.experts.routed_experts.quant_method, MuseSparkQuarkOCPMXFusedMoE
            )
            # zenith: the emulated experts take one row per token (muse_mxfp4_emulation)
            and self.experts.routed_experts.quant_method.mxfp4_backend.value
            != "EMULATION"
        )
''',
    ),
]


def vllm_root():
    """Locate vLLM without importing it (no GPU in a build container)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def patch_file(path, edits, check):
    with open(path) as fh:
        src = fh.read()
    if MARKER in src:
        print(f"already patched: {path}")
        return 0
    for name, old, _ in edits:
        if src.count(old) != 1:
            print(f"ERROR: anchor '{name}' found {src.count(old)} times in {path}",
                  file=sys.stderr)
            print("vLLM changed the Muse MoE code; re-derive the patch", file=sys.stderr)
            return 1
    if check:
        print(f"would patch: {path}")
        return 0
    for _, old, new in edits:
        src = src.replace(old, new, 1)
    with open(path, "w") as fh:
        fh.write(src)
    print(f"patched: {path}  (Muse MXFP4 on the emulation backend)")
    return 0


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
    root = vllm_root()
    runner, model = os.path.join(root, RUNNER), os.path.join(root, MODEL)
    if not (os.path.exists(runner) and os.path.exists(model)):
        print(f"not needed: no Muse Spark ROCm model in {root}")
        return 0
    rc = patch_file(runner, RUNNER_EDITS, args.check)
    return rc or patch_file(model, MODEL_EDITS, args.check)


if __name__ == "__main__":
    sys.exit(main())
