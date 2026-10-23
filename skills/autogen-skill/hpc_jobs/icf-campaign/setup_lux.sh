#!/bin/bash
# Optional pre-submit check (seconds), also run by scripts/setup.sh. It builds nothing: the job builds the app's
# venv itself on its compute node (see job.lux.slurm), since a login node may not be reachable.
# Checks that the uploaded app is complete and that the BACKEND=self files, if configured, are readable.
#
# Env: RUN_DIR_Lux (this folder's uploaded files); ICF_GPTOSS_MODEL, ICF_VLLM_IMAGE_TAR (cluster_defaults.json)
set -u
: "${RUN_DIR_Lux:?RUN_DIR_Lux not set}"
APP=$RUN_DIR_Lux/icf_4agent
ok=1; say() { printf '%-8s %s\n' "$1" "$2"; }

missing=
for f in app.py icf_core.py icf_mcp_server.py icf_viz.py requirements.txt constraints.txt tests/test_offline.py \
         tiktoken/o200k_base.tiktoken start_vllm_lux.sh; do
  [ -r "$APP/$f" ] || missing="$missing $f"
done
[ -z "$missing" ] && say app "$APP" || { say app "MISSING in $APP:$missing"; ok=0; }
[ -r "$RUN_DIR_Lux/run_campaign.py" ] || { say app "MISSING: $RUN_DIR_Lux/run_campaign.py"; ok=0; }

# BACKEND=self (a script_arg, which setup does not see): check its files if they are configured.
if [ -n "${ICF_GPTOSS_MODEL:-}" ]; then
  [ -r "$ICF_GPTOSS_MODEL/config.json" ] && say model "$ICF_GPTOSS_MODEL (BACKEND=self)" \
    || say model "MISSING: $ICF_GPTOSS_MODEL/config.json (BACKEND=self will fail; BACKEND=endpoint is fine)"
fi
if [ -n "${ICF_VLLM_IMAGE_TAR:-}" ]; then
  [ -r "$ICF_VLLM_IMAGE_TAR" ] && say image "$ICF_VLLM_IMAGE_TAR (BACKEND=self)" \
    || say image "MISSING: $ICF_VLLM_IMAGE_TAR (BACKEND=self will fail; BACKEND=endpoint is fine)"
fi
[ "$ok" = 1 ] && echo "setup OK" || { echo "setup FAILED: fix the lines above"; exit 1; }
