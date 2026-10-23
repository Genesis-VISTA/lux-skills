#!/bin/bash
# Hardware FLOP check on one Lux node: rocprofv3 MFMA counters around (1) known-FLOP workloads on 1 GPU (calib.py)
# and (2) one warm-up + one timed fwd+bwd step of each kernel on 8 GPUs. Compare with the `real` formula.
#   sbatch -A <account> -p batch -N1 --exclusive --gpus-per-node=8 -t 00:30:00 scripts/hwflops/run.sh
# Totals: SQ_INSTS_VALU_MFMA_MOPS_BF16 x 512 = FLOPs. Outputs in work/hwflops/out/. Run scripts/setup.sh first.
# Uses _rocm_sdk_core's rocprofv3: the PATH one (_rocm_sdk_devel) loads a second libLLVM next to torch's and aborts.
set -u
# sbatch runs a spooled copy of this file, so submit from the skill directory (or set SK).
SK=${SK:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}; [ -f $SK/env.lux.sh ] || SK=$PWD
source $SK/env.lux.sh >/dev/null
H=$SK/scripts/hwflops J=$SK/hpc_jobs/ringx-bench R=$RINGX_ROOT/ringX-attention P=$RINGX_ROOT/pydeps
OUT=$SK/work/hwflops/out; mkdir -p $OUT
export APPTAINERENV_PYTHONPATH=$R:$P:$J/shims APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 APPTAINERENV_NCCL_DEBUG=WARN
export APPTAINERENV_TRITON_CACHE_DIR=/tmp/$USER/hwf-triton
AX=(apptainer exec --rocm --bind /lustre/orion:/lustre/orion $SIF)
RP=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3
PMC=SQ_INSTS_VALU_MFMA_MOPS_BF16
for c in 0 1; do
  timeout 300 "${AX[@]}" $RP --pmc $PMC --output-format csv -d $OUT/calib$c -o calib -- python $H/calib.py $c > $OUT/calib$c.log 2>&1
  echo "calib causal=$c rc=$?"; grep CALIB $OUT/calib$c.log; "${AX[@]}" python $H/sumcsv.py $OUT/calib$c 2>/dev/null | head -4
done
prof() {  # name, then bench_ringx.py args
  local n=$1; shift
  timeout 600 "${AX[@]}" $RP --pmc $PMC --kernel-include-regex 'Fmha|ck_tile|Cijk|fmha|attn' \
    --output-format csv -d $OUT/$n -o $n -- torchrun --standalone --nnodes 1 --nproc_per_node 8 \
    $J/bench_ringx.py --num_iter 1 "$@" > $OUT/$n.log 2>&1
  echo "$n rc=$?"; "${AX[@]}" python $H/sumcsv.py $OUT/$n 2>/dev/null | head -4
}
for k in ringX2 ringX3 ringX4 zigzag; do prof causal-$k --kernel $k --causal --batch_size 1 --seq_length 16384; done
for k in ringX1 ring; do prof bidir-$k --kernel $k --batch_size 8 --seq_length 8192; done
