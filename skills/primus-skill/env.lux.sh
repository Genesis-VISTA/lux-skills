#!/usr/bin/env bash
# Site layer for running Primus on OLCF Lux.   source env.lux.sh
# Every value is overridable from the environment before sourcing.
export SKILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# Slurm account: first one the user has (Lux rejects jobs without -A).
export ACCOUNT=${ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep -v '^$' | head -1 | tr -d '|')}
export PARTITION=${PARTITION:-batch}

# Bulk data on Lustre only ($HOME is NFS and small). Per-user scratch under the project.
export LUSTRE_ROOT=${LUSTRE_ROOT:-/lustre/orion}
export PRIMUS_ROOT=${PRIMUS_ROOT:-$LUSTRE_ROOT/${ACCOUNT:-unknown}/scratch/$USER/primus-skill}
export PRIMUS_DIR=${PRIMUS_DIR:-$PRIMUS_ROOT/Primus}      # git clone (bind-mounted over /workspace/Primus)
export SIF=${SIF:-/lustre/orion/world-shared/stf218/junqi/primus/primus-v26.5.sif}   # shared 8.1 GB image, only read
export USER_SIF=${USER_SIF:-$PRIMUS_ROOT/images/primus-v26.5.sif}   # your own, built by the job if the shared one is unreadable

# Mistral-Large-123B continued pretraining (scripts/*_cpt.sh): the shared, read-only package with the model
# (HF + Megatron TP1/PP8), Sci-Base + chat-mix data, the tutorial's Primus image and scripts. Read in place.
export PKG=${PKG:-/lustre/orion/world-shared/stf218/junqi/Genesis-Hackathon}
export CPT_SIF=${CPT_SIF:-$PKG/containers/primus_v26.5.sif}            # ROCm 7.15 build the 123B recipe was measured on
export CPT_WORKDIR=${CPT_WORKDIR:-$LUSTRE_ROOT/${ACCOUNT:-unknown}/scratch/$USER/mistral-cpt}   # Primus tree, logs, checkpoints (224 GB each)

# Fabric (verified on Lux: bond0 = TCP bootstrap, 8x ionic RoCE for data).
export PRIVATE_IFACE=${PRIVATE_IFACE:-bond0}

# Lux has NO direct egress. sbatch does not source ~/.bashrc, so export here.
export http_proxy=${http_proxy:-http://proxy.ccs.ornl.gov:3128} https_proxy=${https_proxy:-http://proxy.ccs.ornl.gov:3128}
export no_proxy=${no_proxy:-localhost,127.0.0.1,.ornl.gov,.ccs.ornl.gov}
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy NO_PROXY=$no_proxy

[ -n "$ACCOUNT" ] || echo "env.lux.sh: could not detect a Slurm account; export ACCOUNT=<proj>" >&2
