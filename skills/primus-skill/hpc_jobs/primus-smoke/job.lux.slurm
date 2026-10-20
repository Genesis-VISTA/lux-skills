#!/bin/bash
# Primus smoke test on OLCF Lux: mock data, no checkpoint, ungated tokenizer.
# Carries no #SBATCH lines: resources come from cluster_defaults.json under VISTA (which writes the
# header), and from scripts/run_smoke.sh by hand. setup_lux.sh has already run on the login node.
#
# Env:
#   RUN_DIR_Lux   this folder's files (configs, models, build and summarize scripts); uploaded by VISTA
#   VISTA_OUT     per-job output dir; VISTA creates it, by hand it defaults to $RUNS_DIR/<jobid>
#   VISTA_JOB_DIR shared state (the Primus clone); PRIMUS_ROOT/PRIMUS_DIR/SIF as in setup_lux.sh
# Args ("$@", VISTA's script_args): KEY=VALUE pairs, see README.md.
set -u
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
PRIMUS_ROOT=${PRIMUS_ROOT:-${VISTA_JOB_DIR:?set PRIMUS_ROOT, or source env.lux.sh}}
PRIMUS_DIR=${PRIMUS_DIR:-$PRIMUS_ROOT/Primus}
SIF=${SIF:-}
VISTA_OUT=${VISTA_OUT:-${RUNS_DIR:?set VISTA_OUT or RUNS_DIR}/$SLURM_JOB_ID}
mkdir -p "$VISTA_OUT"

# Only these knobs are accepted: each one is forwarded into the container below, and a
# ${VAR:default} in a YAML that is not forwarded is silently ignored.
CONFIG_NAME=smoke-mistral_7B-bf16-mock.yaml GLOBAL_BATCH_SIZE= TRAIN_ITERS=
for kv in "$@"; do
  case "$kv" in
    CONFIG_NAME=*|GLOBAL_BATCH_SIZE=*|TRAIN_ITERS=*) declare "$kv" ;;
    *) echo "ERROR: unknown argument '$kv' (accepted: CONFIG_NAME= GLOBAL_BATCH_SIZE= TRAIN_ITERS=)" >&2; exit 2 ;;
  esac
done
[ -f "$RUN_DIR_Lux/$CONFIG_NAME" ] || { echo "ERROR: no config $CONFIG_NAME in $RUN_DIR_Lux" >&2; exit 2; }

export http_proxy=${http_proxy:-http://proxy.ccs.ornl.gov:3128} https_proxy=${https_proxy:-http://proxy.ccs.ornl.gov:3128}
export no_proxy=${no_proxy:-localhost,127.0.0.1,.ornl.gov,.ccs.ornl.gov}
FIRST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

# Image: the shared $SIF if readable, else this user's own copy (USER_SIF, built by the job if missing).
# USER_SIF defaults to the job's own working dir, beside the uploaded sources; the shared one is never written.
USER_SIF=${USER_SIF:-$(dirname "$RUN_DIR_Lux")/images/primus-v26.5.sif}
[ -s "$SIF" ] && [ -r "$SIF" ] || SIF=$USER_SIF
# No image yet: build the user's own here, on a compute node (see build_image.sh), then helpers_cpp with it.
if [ ! -s "$SIF" ]; then
  # Fail now rather than 30 minutes in: the build alone takes ~26 min.
  left=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null)   # [d-][h:]mm:ss, or UNLIMITED
  case "$left" in
    ""|*[!0-9:-]*) secs=999999 ;;   # unknown or unlimited: don't block the build
    *) d=0; [[ $left == *-* ]] && { d=${left%%-*}; left=${left#*-}; }
       IFS=: read -r -a t <<< "$left"; secs=$(( 10#$d * 86400 ))
       m=0; for v in "${t[@]}"; do m=$(( m * 60 + 10#$v )); done; secs=$(( secs + m )) ;;
  esac
  need=${IMAGE_BUILD_MIN_SECONDS:-2700}
  [ "$secs" -ge "$need" ] || { echo "ERROR: no readable shared image and none at $SIF; building it needs ~45 min; this job has $left left. Resubmit with a duration of at least 1:00:00." >&2; exit 3; }
  srun -N1 -n1 -w "$FIRST" --cpus-per-task=128 env SIF="$SIF" bash "$RUN_DIR_Lux/build_image.sh" && [ -s "$SIF" ] || { echo "ERROR: image build failed" >&2; exit 3; }
  srun -N1 -n1 -w "$FIRST" env PRIMUS_DIR="$PRIMUS_DIR" SIF="$SIF" bash "$RUN_DIR_Lux/build_helpers.sh" || { echo "ERROR: helpers_cpp build failed" >&2; exit 3; }
fi

# Primus loads configs only from its own tree. Experiment configs have a top-level `modules:`;
# everything else is a model config.
for f in "$RUN_DIR_Lux"/*.yaml; do
  if grep -q '^modules:' "$f"; then cp -f "$f" "$PRIMUS_DIR/examples/megatron/configs/MI355X/"
  else cp -f "$f" "$PRIMUS_DIR/primus/configs/models/megatron/"; fi
done

IFACE=${PRIVATE_IFACE:-bond0}
NODE_TMP=/tmp/$USER/primus
MASTER=$(srun -N1 -n1 -w "$FIRST" bash -c "ip -4 addr show $IFACE | grep -oP '(?<=inet )[0-9.]+'" | tail -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
# Optional knobs are exported only when given. Apptainer passes the host environment into the container, so an
# empty GLOBAL_BATCH_SIZE there replaces the YAML's ${GLOBAL_BATCH_SIZE:256} default with nothing.
for v in GLOBAL_BATCH_SIZE TRAIN_ITERS; do if [ -n "${!v}" ]; then export "$v"; else unset "$v"; fi; done
export CONFIG_NAME IFACE NODE_TMP SIF PRIMUS_DIR VISTA_OUT NNODES=$SLURM_JOB_NUM_NODES
echo "job $SLURM_JOB_ID nodes=$NNODES master=$FIRST/$MASTER:$MASTER_PORT config=$CONFIG_NAME out=$VISTA_OUT"

# --cpus-per-task=128 is mandatory: without it Slurm confines all 8 ranks to one CPU (~5x slower).
srun -N"$NNODES" -n"$NNODES" --ntasks-per-node=1 --cpus-per-task=128 bash -c '
  mkdir -p $NODE_TMP/{triton,inductor,hf,torch_ext}
  # Pass env via APPTAINERENV_* (NOT --env: it comma-splits values like NCCL_IB_HCA).
  export APPTAINERENV_NCCL_SOCKET_IFNAME=$IFACE APPTAINERENV_GLOO_SOCKET_IFNAME=$IFACE
  export APPTAINERENV_NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
  export APPTAINERENV_NCCL_IB_GID_INDEX=1 APPTAINERENV_USING_AINIC=1 APPTAINERENV_NCCL_DEBUG=WARN
  export APPTAINERENV_RCCL_USE_AMD_SMI_LIB=1 APPTAINERENV_CUDA_DEVICE_MAX_CONNECTIONS=1
  export APPTAINERENV_NVTE_USE_RMSNORM_TRITON=1
  export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export APPTAINERENV_PYTHONPATH=/work/Primus:/work/Primus/third_party/Megatron-LM
  export APPTAINERENV_TRITON_CACHE_DIR=$NODE_TMP/triton APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=$NODE_TMP/inductor
  export APPTAINERENV_PRIMUS_WORKSPACE=$VISTA_OUT/primus
  [ -n "${GLOBAL_BATCH_SIZE+x}" ] && export APPTAINERENV_GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE
  [ -n "${TRAIN_ITERS+x}" ] && export APPTAINERENV_TRAIN_ITERS=$TRAIN_ITERS
  export APPTAINERENV_TORCH_EXTENSIONS_DIR=$NODE_TMP/torch_ext APPTAINERENV_HF_HOME=$NODE_TMP/hf APPTAINERENV_HF_HUB_DISABLE_XET=1
  export APPTAINERENV_http_proxy=$http_proxy APPTAINERENV_https_proxy=$https_proxy APPTAINERENV_no_proxy=$no_proxy
  exec apptainer exec --rocm --pwd /work/Primus \
    --bind $PRIMUS_DIR:/work/Primus --bind /lustre/orion:/lustre/orion --bind $NODE_TMP:$NODE_TMP \
    --bind $VISTA_OUT:$VISTA_OUT \
    $SIF bash -c "unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
      exec torchrun --nnodes $NNODES --nproc_per_node 8 --node_rank $SLURM_NODEID \
        --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
        -m primus.cli.main train pretrain --config examples/megatron/configs/MI355X/$CONFIG_NAME"
' 2>&1 | tee "$VISTA_OUT/train.log"
rc=${PIPESTATUS[0]}
echo "=== exit $rc ===" | tee -a "$VISTA_OUT/train.log"
bash "$RUN_DIR_Lux/summarize.sh" "$VISTA_OUT/train.log" | tee "$VISTA_OUT/summary.txt"
exit "$rc"
