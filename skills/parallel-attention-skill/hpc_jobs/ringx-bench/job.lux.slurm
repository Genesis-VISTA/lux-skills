#!/bin/bash -l
# RingX context-parallel attention on Lux nodes (8x MI355X each) in the Primus image: correctness tests against
# single-GPU flash_attn_func, then a fwd+bwd kernel benchmark of the RingX kernels against the ring-flash-attention
# baselines, and the single-GPU flash_attn_func reference at the same per-GPU shapes. Results go to $VISTA_OUT.
# Carries no #SBATCH lines: resources come from cluster_defaults.json under VISTA (which writes the header), and
# from scripts/bench.sh by hand. setup_lux.sh has already cloned RingX and installed ring-flash-attn.
#
# Env:
#   RUN_DIR_Lux    this folder's files (bench_ringx.py, fa_ceiling.py, summarize.py, shims/, tests/); only read
#   VISTA_OUT      per-job output dir; VISTA creates it, by hand it defaults to $RINGX_RUNS_DIR/ringx.<jobid>
#   VISTA_JOB_DIR  shared state from setup_lux.sh (ringX-attention/, pydeps/); by hand RINGX_ROOT
#   SIF            the Primus image (cluster_defaults.json / env.lux.sh)
# Args ("$@", VISTA's script_args): KEY=VALUE pairs, see README.md.
set -u
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
: "${SIF:?SIF not set (cluster_defaults.json or env.lux.sh)}"
RINGX_ROOT=${RINGX_ROOT:-${VISTA_JOB_DIR:?set RINGX_ROOT or VISTA_JOB_DIR}}
VISTA_OUT=${VISTA_OUT:-${RINGX_RUNS_DIR:?set VISTA_OUT or RINGX_RUNS_DIR}/ringx.$SLURM_JOB_ID}
mkdir -p "$VISTA_OUT"
LOG=$VISTA_OUT/job.log
# Every failure leaves a summary.txt, so the reason is where VISTA (and the reader) looks first.
fail() { rc=$1; shift; echo "ERROR: $*" | tee -a "$LOG" >&2; echo "=== BENCH FAILED: $*" > "$VISTA_OUT/summary.txt"; exit "$rc"; }

MODE=all TESTS=ringX1,ringX2,ringX3,ringX4 CAUSAL_KERNELS=ringX2,ringX3,ringX4 BIDIR_KERNELS=ringX1
CAUSAL_POINTS=1:16384 BIDIR_POINTS=32:8192 BASELINES=1 CEILING=1 NUM_HEADS=32 HEAD_DIM=128 NUM_ITER=5 REPEATS=1
RUN_TIMEOUT=900
ACCEPTED="MODE= TESTS= CAUSAL_KERNELS= BIDIR_KERNELS= CAUSAL_POINTS= BIDIR_POINTS= BASELINES= CEILING= NUM_HEADS= HEAD_DIM= NUM_ITER= REPEATS= RUN_TIMEOUT="
for kv in "$@"; do
  case "$kv" in
    MODE=*|TESTS=*|CAUSAL_KERNELS=*|BIDIR_KERNELS=*|CAUSAL_POINTS=*|BIDIR_POINTS=*|BASELINES=*|CEILING=*|\
    NUM_HEADS=*|HEAD_DIM=*|NUM_ITER=*|REPEATS=*|RUN_TIMEOUT=*) declare "$kv" ;;
    *) fail 2 "unknown argument '$kv' (accepted: $ACCEPTED)" ;;
  esac
done
case "$MODE" in all|correctness|bench) ;; *) fail 2 "MODE is all, correctness or bench" ;; esac
case "$BASELINES$CEILING" in [01][01]) ;; *) fail 2 "BASELINES and CEILING are 0 or 1" ;; esac
for v in NUM_HEADS HEAD_DIM NUM_ITER REPEATS RUN_TIMEOUT; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || fail 2 "$v must be a positive integer"
done
[ $((HEAD_DIM % 8)) = 0 ] || fail 2 "HEAD_DIM must be a multiple of 8"
# Kernel names: the RingX ones the job knows about. ringX3b (multi-node only) and ringX1o_varlen are not timed here.
CAUSAL_OK=" ringX2 ringX2o ringX3 ringX4 ringX4o " BIDIR_OK=" ringX1 ringX1o "
for k in ${CAUSAL_KERNELS//,/ }; do [[ "$CAUSAL_OK" == *" $k "* ]] || fail 2 "CAUSAL_KERNELS: '$k' is not one of$CAUSAL_OK(comma-separated)"; done
for k in ${BIDIR_KERNELS//,/ }; do [[ "$BIDIR_OK" == *" $k "* ]] || fail 2 "BIDIR_KERNELS: '$k' is not one of$BIDIR_OK(comma-separated)"; done
for t in ${TESTS//,/ }; do case "$t" in ringX1|ringX2|ringX3|ringX4) ;; *) fail 2 "TESTS: '$t' is not ringX1, ringX2, ringX3 or ringX4" ;; esac; done
NGPN=${SLURM_GPUS_ON_NODE:-8} NNODES=${SLURM_JOB_NUM_NODES:-1}
[ "$NGPN" -ge 2 ] 2>/dev/null || fail 2 "this job sees ${SLURM_GPUS_ON_NODE:-no} GPUs; submit with gpus_per_node 8 (and exclusive_node_use)"
NGPU=$((NNODES * NGPN))   # the context-parallel group: every GPU of every node
# Points are BATCH:TOKENS_PER_GPU, comma-separated; tokens per GPU must split into 2 x GPUs zigzag chunks.
for p in ${CAUSAL_POINTS//,/ } ${BIDIR_POINTS//,/ }; do
  [[ "$p" =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]] || fail 2 "points are BATCH:TOKENS_PER_GPU, comma-separated, e.g. 1:16384,64:4096 (got '$p')"
  [ $(( ${p#*:} % (2 * NGPU) )) = 0 ] || fail 2 "tokens per GPU in '$p' must be a multiple of $((2 * NGPU)) (2 x GPUs)"
done
[ -n "$CAUSAL_POINTS$BIDIR_POINTS" ] || [ "$MODE" = correctness ] || fail 2 "CAUSAL_POINTS and BIDIR_POINTS are both empty: nothing to benchmark"

R=$RINGX_ROOT/ringX-attention P=$RINGX_ROOT/pydeps
[ -d "$R/ringX_attn" ] && [ -d "$P/ring_flash_attn" ] || fail 3 "no RingX clone or ring-flash-attn in $RINGX_ROOT: setup_lux.sh did not run (by hand: scripts/setup.sh)"
[ -r "$SIF" ] || fail 3 "the image $SIF is unreadable"

IFACE=${PRIVATE_IFACE:-bond0}
NODE_TMP=/tmp/$USER/ringx.$SLURM_JOB_ID
mkdir -p "$NODE_TMP/triton"
trap 'rm -rf "$NODE_TMP"' EXIT
if [ "$NNODES" -gt 1 ]; then
  srun -N"$NNODES" --ntasks-per-node=1 mkdir -p "$NODE_TMP/triton" || fail 3 "could not create $NODE_TMP on every node"
  trap 'srun -N"$NNODES" --ntasks-per-node=1 rm -rf "$NODE_TMP"; rm -rf "$NODE_TMP"' EXIT
fi
# Pass env via APPTAINERENV_* (NOT --env: it comma-splits values like NCCL_IB_HCA). Same fabric settings as
# primus-skill's primus-smoke job. On one node RCCL stays on xGMI; across nodes it uses the ionic NICs.
export APPTAINERENV_NCCL_SOCKET_IFNAME=$IFACE APPTAINERENV_GLOO_SOCKET_IFNAME=$IFACE
export APPTAINERENV_NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export APPTAINERENV_NCCL_IB_GID_INDEX=1 APPTAINERENV_USING_AINIC=1 APPTAINERENV_RCCL_USE_AMD_SMI_LIB=1
export APPTAINERENV_NCCL_DEBUG=WARN APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1   # the uploaded sources may be read-only
export APPTAINERENV_PYTHONPATH=$R:$P:$RUN_DIR_Lux/shims   # ringX_attn, ring_flash_attn, the import shims
export APPTAINERENV_TRITON_CACHE_DIR=$NODE_TMP/triton
AX=(apptainer exec --rocm --bind /lustre/orion:/lustre/orion --bind "$NODE_TMP:$NODE_TMP" --bind "$VISTA_OUT:$VISTA_OUT" "$SIF")
# One node: no srun, torchrun starts one rank per GPU inside one container; --standalone picks a free port.
# Several nodes: one srun task per node, each a torchrun with this node as master. Each launch gets its own port,
# so a launch never meets the previous one's socket. --cpus-per-task: otherwise Slurm pins all 8 ranks to one CPU.
# RUN_TIMEOUT bounds a hung launch.
MASTER_ADDR=$(ip -4 addr show "$IFACE" | grep -oP '(?<=inet )[0-9.]+' | head -1)
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
[ "$NNODES" = 1 ] || [ -n "$MASTER_ADDR" ] || fail 3 "no IPv4 address on $IFACE; set PRIVATE_IFACE"
run8() {  # $1 log, rest: script and args. Returns the launch's exit code.
  local log=$1; shift
  if [ "$NNODES" = 1 ]; then
    timeout -k 30 "$RUN_TIMEOUT" "${AX[@]}" torchrun --standalone --nnodes 1 --nproc_per_node "$NGPN" "$@" > "$log" 2>&1
  else
    PORT=$((PORT + 1))
    timeout -k 30 "$RUN_TIMEOUT" srun -N"$NNODES" -n"$NNODES" --ntasks-per-node=1 --cpus-per-task=128 \
      --gpus-per-node="$NGPN" --kill-on-bad-exit=1 "${AX[@]}" bash -c 'exec torchrun --nnodes "$1" \
      --nproc_per_node "$2" --node_rank "$SLURM_NODEID" --master_addr "$3" --master_port "$4" "${@:5}"' \
      _ "$NNODES" "$NGPN" "$MASTER_ADDR" "$PORT" "$@" > "$log" 2>&1
  fi
}

echo "job $SLURM_JOB_ID on $NNODES node(s) from $(hostname -s), $NGPU GPUs, mode=$MODE args: $*" | tee "$LOG"
"${AX[@]}" python -c '
import torch, flash_attn, ring_flash_attn, ringX_attn
rccl = ".".join(map(str, torch.cuda.nccl.version()))
print(f"torch {torch.__version__} (HIP {torch.version.hip}), flash_attn {flash_attn.__version__}, RCCL {rccl}, GPUs {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
' > "$VISTA_OUT/env.raw" 2>&1 || { cat "$VISTA_OUT/env.raw" >> "$LOG"; fail 3 "the image cannot import torch/flash_attn/ring_flash_attn/ringX_attn; see job.log"; }
{ echo "node $(hostname -s), job $SLURM_JOB_ID, $(date -u +%FT%TZ), $NNODES node(s), $NGPU GPUs"
  echo "RingX $(git -C "$R" rev-parse --short HEAD), ring-flash-attn $(sed -n 's/^Version: //p' "$P"/ring_flash_attn-*.dist-info/METADATA)"
  grep -v -e xalt -e WARN "$VISTA_OUT/env.raw" | tail -1
  echo "image $SIF"; } > "$VISTA_OUT/env.txt"
rm -f "$VISTA_OUT/env.raw"
cat "$VISTA_OUT/env.txt" | tee -a "$LOG"

# --- 1. correctness vs flash_attn_func: RingX's own tests (ringX4 from tests/, upstream has none) ---
# The tests put rank r on cuda:r, which exists only on one node: multi-node jobs time the kernels only.
if [ "$MODE" != bench ] && [ "$NNODES" -gt 1 ]; then
  echo "correctness tests skipped: they run on one node only (rank r uses cuda:r)" | tee -a "$LOG" "$VISTA_OUT/env.txt"
elif [ "$MODE" != bench ]; then
  mkdir -p "$VISTA_OUT/correctness"
  for t in ${TESTS//,/ }; do
    f=$R/test/test_${t}_attn_func.py; [ "$t" = ringX4 ] && f=$RUN_DIR_Lux/tests/test_ringX4_attn_func.py
    run8 "$VISTA_OUT/correctness/$t.log" "$f"; rc=$?
    echo "$rc" > "$VISTA_OUT/correctness/$t.rc"
    echo "$(date +%T) correctness $t rc=$rc" | tee -a "$LOG"
  done
fi

# --- 2. kernel benchmark, one launch per kernel and point ---
if [ "$MODE" != correctness ]; then
  mkdir -p "$VISTA_OUT/bench"; : > "$VISTA_OUT/runs.tsv"
  SPECS=
  bench() {  # $1 causal|bidir, $2 point, $3 kernels
    local mode=$1 b=${2%:*} s=${2#*:} k rep log rc flag=
    [ "$mode" = causal ] && flag=--causal
    SPECS="$SPECS $b,$s,$NUM_HEADS,$HEAD_DIM,$([ "$mode" = causal ] && echo 1 || echo 0)"
    for k in ${3//,/ }; do
      for rep in $(seq 1 "$REPEATS"); do
        log=$VISTA_OUT/bench/$mode-b$b-s$s-$k.$rep.log
        run8 "$log" "$RUN_DIR_Lux/bench_ringx.py" --kernel "$k" $flag --batch_size "$b" --seq_length "$s" \
          --num_heads "$NUM_HEADS" --head_dim "$HEAD_DIM" --num_iter "$NUM_ITER"; rc=$?
        printf '%s\t%s\t%s\t%s\t%s\n' "$mode:$b:$s" "$k" "$rep" "$rc" "$log" >> "$VISTA_OUT/runs.tsv"
        echo "$(date +%T) $mode b=$b s=$s $k run $rep rc=$rc $(grep -h -o '[0-9.]* sec$' "$log" | tail -1)" | tee -a "$LOG"
      done
    done
  }
  # Baselines: zigzag and stripe assert causal, so bi-directional points time plain ring only.
  for p in ${CAUSAL_POINTS//,/ }; do bench causal "$p" "$CAUSAL_KERNELS$([ "$BASELINES" = 1 ] && echo ,ring,zigzag,stripe)"; done
  for p in ${BIDIR_POINTS//,/ };  do bench bidir  "$p" "$BIDIR_KERNELS$([ "$BASELINES" = 1 ] && echo ,ring)"; done

  # --- 3. single-GPU flash_attn_func at the same per-GPU shapes ---
  if [ "$CEILING" = 1 ]; then
    HIP_VISIBLE_DEVICES=0 APPTAINERENV_HIP_VISIBLE_DEVICES=0 timeout -k 30 "$RUN_TIMEOUT" "${AX[@]}" \
      python "$RUN_DIR_Lux/fa_ceiling.py" $SPECS 2>&1 | grep -e '^CEILING' > "$VISTA_OUT/ceiling.txt"
    echo "$(date +%T) single-GPU reference: $(wc -l < "$VISTA_OUT/ceiling.txt") shape(s)" | tee -a "$LOG"
  fi
  # RingX's own parser, for comparison with upstream numbers (it reads log.algo* in the current dir).
  (cd "$VISTA_OUT/bench" && for f in *.log; do ln -sf "$f" "log.algo-$f"; done
   "${AX[@]}" python "$R/benchmark/utils.py" 2>/dev/null | grep -v xalt > "$VISTA_OUT/utils_summary.txt"; rm -f log.algo-*)
fi

"${AX[@]}" python "$RUN_DIR_Lux/summarize.py" "$VISTA_OUT" 2>&1 | grep -v xalt | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ -s "$VISTA_OUT/summary.txt" ] || fail 4 "summarize.py wrote no summary.txt; see job.log"
echo "=== exit $rc" | tee -a "$LOG"
exit "$rc"
