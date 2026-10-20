#!/usr/bin/env python3
"""
Tuned AITER FlyDSL a4w4 MoE configs for Muse Spark MXFP4 under EP8 on gfx950
(opt-in: ZENITH_MUSE_FLYDSL_TUNED=1).

THE PROBLEM
vLLM runs Muse Spark's Quark OCP MXFP4 experts (MuseSparkAiterExperts) through
aiter.fused_moe, which dispatches to the FlyDSL a4w4 two-stage kernels. AITER
picks the GEMM1/GEMM2 kernels from tuned_fmoe.csv, and it ships no rows for
Muse's shape: 256 experts, 32 local under EP8, model_dim = inter_dim = 4096,
and topk 2 (vLLM expands each token to 8 rows, each with one global expert and
the masked slot 256). Every bucket logs "no tuned FlyDSL config for (...)" and
falls back to a heuristic: t32x128x256 GEMM1 plus t32 atomic GEMM2 up to 16K
rows, and t64 above that.

THE FIX
bench/muse/flydsl_moe_tune.py swept the FlyDSL GEMM1 kernels (including the
_fp4 variants that quantize GEMM1's output in its epilogue) and the legacy and
v2-layout GEMM2 kernels that are valid under expert_mask, for every padded
token bucket the EP8 serve log looks up (8 .. 32768 rows, c=1 .. 4096 tokens).
It kept a candidate only if its error against an fp32 reference was no worse
than the heuristic's, and only buckets that won by 1% or more. The rows are
keyed exactly as AITER builds them (gfx950, cu_num 256), so other models and
other EP/TP layouts are unaffected.

This patch writes those rows to
    <aiter>/configs/model_configs/muse_ep8_flydsl_tuned_fmoe.csv
which AITER merges with configs/tuned_fmoe.csv when AITER_CONFIG_FMOE is unset
(aiter.jit.core: AITER_CONFIG.get_config_file). With the variable unset or 0
it removes that file, so the image default is the untuned heuristic. Setting
AITER_CONFIG_FMOE yourself bypasses the merge and this file.
ZENITH_MUSE_FLYDSL_TUNED_CSV=<path> installs a different CSV (same header).

Kernel choice changes the summation order of GEMM2 (atomic vs. per-tile), not
the quantization, so outputs differ from the heuristic at the bf16 rounding
level; gsm8k is in results/muse/flydsl_moe_mxfp4_2026-10-17/README.md.

Usage (inside the container):
    ZENITH_MUSE_FLYDSL_TUNED=1 python3 muse_flydsl_tuned_fmoe.py [--check]
"""

import argparse
import importlib.util
import os
import sys

NAME = "muse_ep8_flydsl_tuned_fmoe.csv"

# bench/muse/flydsl_moe_tune.py output (lux job 349991), MI355X, image musemain2
ROWS = """\
gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,xbf16,flat,tflops,bw,_tag
gfx950,256,8,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_kw4_fp4,0.21611,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32,0.21611,22.40,0,0,0,0,0,muse_ep8
gfx950,256,16,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_kw4_fp4,0.20171,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32,0.20171,24.61,0,0,0,0,0,muse_ep8
gfx950,256,32,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_kw2_fp4,0.21136,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32,0.21136,34.63,0,0,0,0,0,muse_ep8
gfx950,256,64,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_fp4,0.21312,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32,0.21312,52.75,0,0,0,0,0,muse_ep8
gfx950,256,128,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w4_fp4,0.21105,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32,0.21105,76.63,0,0,0,0,0,muse_ep8
gfx950,256,256,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w4_kw2_fp4,0.21101,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32,0.21101,100.55,0,0,0,0,0,muse_ep8
gfx950,256,512,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w4_fp4,0.21175,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32,0.21175,137.13,0,0,0,0,0,muse_ep8
gfx950,256,1024,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_fp4,0.21047,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32,0.21047,152.45,0,0,0,0,0,muse_ep8
gfx950,256,2048,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_fp4,0.21064,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32,0.21064,165.71,0,0,0,0,0,muse_ep8
gfx950,256,4096,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w4_fp4,0.20984,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32,0.20984,174.16,0,0,0,0,0,muse_ep8
gfx950,256,8192,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w3_fp4,0.21030,0.0,flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_atomic_nt_sbm64,0.21030,192.60,0,0,0,0,0,muse_ep8
gfx950,256,16384,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_fp4,0.21022,0.0,flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2_persist,0.21022,271.87,0,0,0,0,0,muse_ep8
gfx950,256,32768,4096,4096,32,2,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,0.0,flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_bnt0_fp4,0.21022,0.0,flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2_persist,0.21022,396.61,0,0,0,0,0,muse_ep8
"""


def target():
    spec = importlib.util.find_spec("aiter")  # no import: aiter builds its core module on import
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0], "configs", "model_configs", NAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report status without writing")
    args = ap.parse_args()

    path = target()
    if path is None or not os.path.isdir(os.path.dirname(path)):
        print("muse_flydsl_tuned_fmoe: no aiter/configs/model_configs, skipping")
        return 0
    want = os.environ.get("ZENITH_MUSE_FLYDSL_TUNED", "0") not in ("", "0")
    src = os.environ.get("ZENITH_MUSE_FLYDSL_TUNED_CSV", "")
    text = open(src).read() if (want and src) else ROWS
    have = open(path).read() if os.path.exists(path) else None

    if args.check:
        state = "absent" if have is None else ("current" if have == text else "different")
        print(f"muse_flydsl_tuned_fmoe: {path} {state}; ZENITH_MUSE_FLYDSL_TUNED={'1' if want else '0'}")
        return 0
    if not want:
        if have is not None:
            os.remove(path)
            print(f"muse_flydsl_tuned_fmoe: removed {path} (ZENITH_MUSE_FLYDSL_TUNED unset)")
        return 0
    if have != text:
        with open(path + ".tmp", "w") as f:
            f.write(text)
        os.replace(path + ".tmp", path)
    print(f"muse_flydsl_tuned_fmoe: installed {len(text.splitlines()) - 1} rows at {path}"
          + (f" from {src}" if src else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
