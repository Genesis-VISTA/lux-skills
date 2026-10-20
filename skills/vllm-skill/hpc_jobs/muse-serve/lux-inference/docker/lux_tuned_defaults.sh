# Per-configuration serving defaults tuned on Lux. Sourced (not run) by
# submit_lux.slurm.sh and run_node.sh once ZENITH_MODEL is known and before
# their generic defaults:
#
#   source ./lux_tuned_defaults.sh <number of nodes>
#
# Each value is only a default: anything already set in the environment wins,
# including an empty string (ZENITH_MAX_NUM_BATCHED_TOKENS= keeps vLLM's own
# value). ZENITH_TUNED_DEFAULTS=0 turns the whole file off and gives the
# generic defaults back.
#
# Five configurations have defaults here: Inkling-MXFP4 on 2 nodes (the
# submit script's default, sized for the full 1M-token context) and on 1 node
# (tuned for throughput), Mistral-Large-Instruct-2411 on 1 node at 1M tokens,
# and Muse Spark Open (MXFP4 and bf16) on 1 node at 1M tokens (Lux addition,
# from zenith v0.1.9's bench/muse/muse_ladder_spec.sbatch). Every other model
# and node count gets the generic defaults.
# ZENITH_TUNED_AS=<model> applies a model's defaults to a fine-tune of it
# served under another directory name.
#
# Inkling-MXFP4 on one 8 x MI355X node (Lux hackathon tuning, 2026-10). The
# node serves one TP8 instance on all 8 GPUs, which is the launcher's default
# split and gives the highest total token rate a single job can reach:
#
#   ZENITH_MTP=1, ZENITH_MTP_TOKENS=3   MTP with 3 draft tokens. The head has
#                                       8, but acceptance is ~2.8 tokens, and
#                                       9 tokens per sequence overflow the
#                                       graph capture limit from c=57 on.
#   ZENITH_TARGET_KV_GB=128             KV pool per GPU (the generic default
#                                       is 64).
#   ZENITH_MAX_NUM_BATCHED_TOKENS=32768 scheduler token budget (vLLM: 16384);
#                                       fewer, larger prefill chunks.
#   ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE=2048  CUDA graphs up to 2048 tokens per
#                                       step (vLLM stops at 512).
#
# Together with the kernels in patches/hl_overlay (applied on gfx950 by
# run_container.sh), these take the throughput-vs-interactivity score over
# c=1..1024 at 8k in / 1k out from 70.1 (TP8, KV 128, MTP-3, everything else
# vLLM's defaults) to 89.2. Output tok/s per GPU / tok/s per user:
#
#   c=1  18 / 153    c=16  152 / 102    c=128  358 / 36    c=1024  437 / 6.8
#
# Under the old 1-node defaults (MTP-8, KV 64, capture 512) every decode step
# past c=56 ran without a graph and throughput collapsed at c=64..128.

_ZENITH_TUNED_MODEL="${ZENITH_TUNED_AS:-${ZENITH_MODEL:-}}"

if [ "${ZENITH_TUNED_DEFAULTS:-1}" != "0" ] \
   && [ "${_ZENITH_TUNED_MODEL}" = "Inkling-MXFP4" ] && [ "${1:-0}" = "1" ]; then
    export ZENITH_MTP="${ZENITH_MTP-1}"
    export ZENITH_MTP_TOKENS="${ZENITH_MTP_TOKENS-3}"
    export ZENITH_TARGET_KV_GB="${ZENITH_TARGET_KV_GB-128}"
    export ZENITH_MAX_NUM_BATCHED_TOKENS="${ZENITH_MAX_NUM_BATCHED_TOKENS-32768}"
    export ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE="${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE-2048}"
    echo "lux: tuned defaults for Inkling-MXFP4 on 1 node:" \
         "MTP=${ZENITH_MTP} MTP_TOKENS=${ZENITH_MTP_TOKENS:-all} KV_GB=${ZENITH_TARGET_KV_GB}" \
         "MAX_NUM_BATCHED_TOKENS=${ZENITH_MAX_NUM_BATCHED_TOKENS:-vllm} CAPTURE=${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE:-vllm}" \
         "(ZENITH_TUNED_DEFAULTS=0 for the generic ones)"
fi

# Inkling-MXFP4 on two nodes, TP8 x PP2: the Lux hackathon tutorial
# configuration, sized for the model's full 1,048,576-token context. With PP2
# each GPU holds half the layers (~44 GiB of weights), which leaves most of its
# 288 GiB for KV, so many long-context agent sessions fit at once:
#
#   ZENITH_MAX_MODEL_LEN=1048576        the model's model_max_length
#   ZENITH_TARGET_KV_GB=200             gpu-memory-utilization ~0.84; vLLM
#                                       reports a 10.3M-token KV pool, room
#                                       for 9.8 concurrent 1M-token requests
#                                       (kv_cache_sizing.py's per-token
#                                       estimate is conservative here)
#   ZENITH_MTP=0                        off: see ZENITH_MTP=1 below
#   ZENITH_MAX_NUM_BATCHED_TOKENS=32768 as on one node; a 1M prompt prefills
#                                       in 32 chunks
#
# These are a working baseline, not tuned for throughput.
#
# ZENITH_MTP=1 is an opt-in for one or a few users at a time. It also defaults
# ZENITH_MTP_PP=1, ZENITH_MTP_TOKENS=3, ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE=2048
# and ZENITH_PP_DEFERRED_RECV=0 (MTP + PP2 hangs in warmup on Lux with the
# deferred recv; see patches/pp_broadcast_deferred_recv.py). Measured at
# 8k in / 1k out: tok/s/user 143 vs 82.5 at c=1, about even at c=2-16, and
# 13-44% less tok/s/GPU from c=8 up (curve score 43.4 vs 51.5). Decode at a
# 1M-token prompt goes from ~16 to 53-63 tok/s. MMLU-Pro 0.889, needle 15/15.
if [ "${ZENITH_TUNED_DEFAULTS:-1}" != "0" ] \
   && [ "${_ZENITH_TUNED_MODEL}" = "Inkling-MXFP4" ] && [ "${1:-0}" = "2" ]; then
    export ZENITH_MAX_MODEL_LEN="${ZENITH_MAX_MODEL_LEN-1048576}"
    export ZENITH_TARGET_KV_GB="${ZENITH_TARGET_KV_GB-200}"
    export ZENITH_MTP="${ZENITH_MTP-0}"
    export ZENITH_MAX_NUM_BATCHED_TOKENS="${ZENITH_MAX_NUM_BATCHED_TOKENS-32768}"
    if [ "${ZENITH_MTP}" = "1" ]; then
        export ZENITH_MTP_PP="${ZENITH_MTP_PP-1}"
        export ZENITH_MTP_TOKENS="${ZENITH_MTP_TOKENS-3}"
        export ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE="${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE-2048}"
        export ZENITH_PP_DEFERRED_RECV="${ZENITH_PP_DEFERRED_RECV-0}"
    fi
    echo "lux: tuned defaults for Inkling-MXFP4 on 2 nodes (TP8 x PP2):" \
         "MAX_MODEL_LEN=${ZENITH_MAX_MODEL_LEN} KV_GB=${ZENITH_TARGET_KV_GB} MTP=${ZENITH_MTP}" \
         "${ZENITH_MTP_TOKENS:+MTP_TOKENS=${ZENITH_MTP_TOKENS} }${ZENITH_MTP_PP:+MTP_PP=${ZENITH_MTP_PP} }${ZENITH_PP_DEFERRED_RECV:+PP_DEFERRED_RECV=${ZENITH_PP_DEFERRED_RECV} }${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE:+CAPTURE=${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE} }MAX_NUM_BATCHED_TOKENS=${ZENITH_MAX_NUM_BATCHED_TOKENS:-vllm}" \
         "(ZENITH_TUNED_DEFAULTS=0 for the generic ones)"
fi

# Mistral-Large-Instruct-2411 (dense 123B) on one node, TP8, at 1,048,576
# tokens (Lux hackathon, 2026-10; docs/LUX_MISTRAL_LARGE.md). The model is
# trained to 131,072 and stretched with YaRN factor 16 (8 would be enough
# positions, but at 8 the model loses a needle at 1M). Its 229 GiB of weights
# spread over 8 GPUs leave room for a 3.3M-token bf16 KV pool, so one node
# holds about three full-length requests and more nodes mean more copies, not
# a wider one.
#
#   ZENITH_MAX_MODEL_LEN=1048576
#   ZENITH_TARGET_KV_GB=200             vLLM reports 3.34M KV tokens (3.2x 1M)
#   ZENITH_MAX_NUM_BATCHED_TOKENS=32768 larger prefill chunks
#   ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE=1024
#   ZENITH_REASONING_PARSER=none, ZENITH_TOOL_CALL_PARSER=mistral
#   ZENITH_VLLM_EXTRA_ARGS:
#     --tokenizer-mode mistral          the model's own tokenizer
#                                       (tokenizer.model.v7) and chat format
#     --attention-backend ROCM_AITER_FA fastest long-context prefill (a 1M
#                                       prompt in ~5 min vs ~9 with the
#                                       others); ROCM_AITER_UNIFIED_ATTN is
#                                       ~10% faster on short prompts
#     --hf-overrides <YaRN>             rope_scaling for 1M. It replaces the
#                                       config's whole rope dict, so it has to
#                                       repeat rope_theta 1e6: without it vLLM
#                                       falls back to base 10000 and all output
#                                       past a few thousand tokens is garbage.
#                                       attention_factor 1.15 instead of
#                                       YaRN's 1.28: the default scale costs
#                                       accuracy at every length (science
#                                       suite 0.61 vs 0.66 native) and 1.15
#                                       recovers it while finding a needle at
#                                       all 10 depths of 512k and 1,040k.
#
# With ZENITH_MAX_MODEL_LEN=131072 or less the model runs at its native
# context instead: no YaRN, ROCM_AITER_UNIFIED_ATTN, a 4.2M-token pool (32
# requests of 131k). Use it for anything that does not need long prompts,
# and to score fine-tunes.
#
# Setting ZENITH_VLLM_EXTRA_ARGS replaces all three flags; copy them from the
# line this prints and add to them.
if [ "${ZENITH_TUNED_DEFAULTS:-1}" != "0" ] \
   && [ "${_ZENITH_TUNED_MODEL}" = "Mistral-Large-Instruct-2411" ] && [ "${1:-0}" = "1" ]; then
    export ZENITH_MAX_MODEL_LEN="${ZENITH_MAX_MODEL_LEN-1048576}"
    export ZENITH_TARGET_KV_GB="${ZENITH_TARGET_KV_GB-200}"
    export ZENITH_MTP="${ZENITH_MTP-0}"
    export ZENITH_MAX_NUM_BATCHED_TOKENS="${ZENITH_MAX_NUM_BATCHED_TOKENS-32768}"
    export ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE="${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE-1024}"
    export ZENITH_REASONING_PARSER="${ZENITH_REASONING_PARSER-none}"
    export ZENITH_TOOL_CALL_PARSER="${ZENITH_TOOL_CALL_PARSER-mistral}"
    # (JSON braces cannot go inside a ${VAR-default})
    if [ "${ZENITH_MAX_MODEL_LEN:-100000}" -gt 131072 ] 2>/dev/null; then
        _ZENITH_MISTRAL_ARGS='--tokenizer-mode mistral --attention-backend ROCM_AITER_FA --hf-overrides {"rope_scaling":{"rope_type":"yarn","factor":16.0,"original_max_position_embeddings":131072,"rope_theta":1000000.0,"attention_factor":1.15}}'
    else
        _ZENITH_MISTRAL_ARGS='--tokenizer-mode mistral --attention-backend ROCM_AITER_UNIFIED_ATTN'
    fi
    [ -n "${ZENITH_VLLM_EXTRA_ARGS+set}" ] || export ZENITH_VLLM_EXTRA_ARGS="${_ZENITH_MISTRAL_ARGS}"
    unset _ZENITH_MISTRAL_ARGS
    echo "lux: tuned defaults for Mistral-Large-Instruct-2411 on 1 node:" \
         "MAX_MODEL_LEN=${ZENITH_MAX_MODEL_LEN} KV_GB=${ZENITH_TARGET_KV_GB}" \
         "MAX_NUM_BATCHED_TOKENS=${ZENITH_MAX_NUM_BATCHED_TOKENS:-vllm} CAPTURE=${ZENITH_MAX_CUDAGRAPH_CAPTURE_SIZE:-vllm}" \
         "EXTRA_ARGS=${ZENITH_VLLM_EXTRA_ARGS}" \
         "(ZENITH_TUNED_DEFAULTS=0 for the generic ones)"
fi
# Muse Spark Open (816B-total, 42B-active MoE, 1,048,576 tokens) on one 8 x
# MI355X node, TP8, with the DSpark draft: the two "best" release recipes of
# zenith v0.1.9 (README.md §1d.3; bench/muse/muse_ladder_spec.sbatch sets the
# same values; the MXFP4 launch environment of job 360800 is in
# results/muse/release_validation_2026-10-20/lux/360800/config.txt). Lux
# addition: upstream sets these in the sbatch, here they are defaults keyed on
# ZENITH_MODEL like Inkling's. Needs the musemain2 image (vLLM 0e939af8 from
# vllm_src/, AITER a3158e48). One node per endpoint: above ~256 concurrent
# 8k/1k requests run more 1-node replicas.
#
# Both checkpoints:
#   ZENITH_MAX_MODEL_LEN=1048576, ZENITH_MTP=0, ZENITH_HL_OVERLAY=0
#   env  VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16 VLLM_USE_BREAKABLE_CUDAGRAPH=1
#        VLLM_USE_V2_MODEL_RUNNER=1 (DSpark runs only on Model Runner V2)
#   args --language-model-only --dtype bfloat16 --attention-backend
#        ROCM_AITER_FA --no-enable-prefix-caching, graphs FULL_AND_PIECEWISE
#   ZENITH_MUSE_DRAFT=Muse-Spark-Open-DSpark (a directory in ZENITH_MODELPATH,
#        method dspark, 7 tokens, a K schedule per batch size); empty = no draft
# MXFP4 (amd/...-Quark-MXFP4, score 71.96, gsm8k 0.976): EP8 MoE, INT8
#   quick-reduce, the tuned FlyDSL MoE (ZENITH_MUSE_FLYDSL_TUNED=1), KV 180 GB,
#   8192-token budget, graphs to 1024, K 7/4/3/0 at c 1-4/5-16/17-256/257+.
# bf16 (meta-models/..., score 56.87, gsm8k 0.984): TP MoE with the tuned bf16
#   tables (VLLM_TUNED_CONFIG_FOLDER), no quick-reduce, KV 60 GB (60 is the
#   ceiling with the draft), 4096-token budget, graphs to 512, 512 sequences,
#   K 7/3/0 at c 1-8/9-256/257+.
#
# The env is prepended to ZENITH_VLLM_EXTRA_ENV / ZENITH_CONTAINER_EXTRA_ENV
# (anything already there is kept and wins); ZENITH_VLLM_EXTRA_ARGS, if set,
# replaces all of the args, as for Mistral.
case "${_ZENITH_TUNED_MODEL}" in
    Muse-Spark-Open-816B-A42B-Quark-MXFP4*) _ZENITH_MUSE=mxfp4 ;;
    Muse-Spark-Open-816B-A42B)               _ZENITH_MUSE=bf16 ;;
    *)                                       _ZENITH_MUSE= ;;
esac
if [ "${ZENITH_TUNED_DEFAULTS:-1}" != "0" ] && [ -n "${_ZENITH_MUSE}" ] && [ "${1:-0}" = "1" ]; then
    export ZENITH_MAX_MODEL_LEN="${ZENITH_MAX_MODEL_LEN-1048576}"
    export ZENITH_MTP="${ZENITH_MTP-0}"
    export ZENITH_HL_OVERLAY="${ZENITH_HL_OVERLAY-0}"
    export ZENITH_MUSE_DRAFT="${ZENITH_MUSE_DRAFT-Muse-Spark-Open-DSpark}"
    _ZENITH_MUSE_ENV="VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16 VLLM_USE_BREAKABLE_CUDAGRAPH=1"
    [ -n "${ZENITH_MUSE_DRAFT}" ] && _ZENITH_MUSE_ENV+=" VLLM_USE_V2_MODEL_RUNNER=1"
    if [ "${_ZENITH_MUSE}" = mxfp4 ]; then
        export ZENITH_TARGET_KV_GB="${ZENITH_TARGET_KV_GB-180}"
        export ZENITH_MAX_NUM_BATCHED_TOKENS="${ZENITH_MAX_NUM_BATCHED_TOKENS-8192}"
        export ZENITH_MAX_NUM_SEQS="${ZENITH_MAX_NUM_SEQS-2048}"
        _ZENITH_MUSE_CG=1024
        _ZENITH_MUSE_ENV+=" VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8 ZENITH_MUSE_FLYDSL_TUNED=1"
        _ZENITH_MUSE_PAR=" --enable-expert-parallel"
        _ZENITH_MUSE_K='[[1,4,7],[5,16,4],[17,256,3],[257,2048,0]]'
    else
        export ZENITH_TP="${ZENITH_TP-8}"
        export ZENITH_PP="${ZENITH_PP-1}"
        export ZENITH_TARGET_KV_GB="${ZENITH_TARGET_KV_GB-60}"
        export ZENITH_MAX_NUM_BATCHED_TOKENS="${ZENITH_MAX_NUM_BATCHED_TOKENS-4096}"
        export ZENITH_MAX_NUM_SEQS="${ZENITH_MAX_NUM_SEQS-512}"
        _ZENITH_MUSE_CG=512
        _ZENITH_MUSE_ENV+=" VLLM_TUNED_CONFIG_FOLDER=/opt/zenith_patches/muse_bf16_moe_configs"
        _ZENITH_MUSE_PAR=""
        _ZENITH_MUSE_K='[[1,8,7],[9,256,3],[257,2048,0]]'
    fi
    # (JSON braces cannot go inside a ${VAR-default})
    _ZENITH_MUSE_ARGS="--language-model-only --dtype bfloat16 --attention-backend ROCM_AITER_FA --no-enable-prefix-caching"
    _ZENITH_MUSE_ARGS+=' --compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":'"${_ZENITH_MUSE_CG}"'}'
    _ZENITH_MUSE_ARGS+="${_ZENITH_MUSE_PAR}"
    [ -n "${ZENITH_MUSE_DRAFT}" ] && _ZENITH_MUSE_ARGS+=' --speculative-config {"method":"dspark","model":"/app/models/'"${ZENITH_MUSE_DRAFT}"'","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":'"${_ZENITH_MUSE_K}"'}'
    [ -n "${ZENITH_VLLM_EXTRA_ARGS+set}" ] || export ZENITH_VLLM_EXTRA_ARGS="${_ZENITH_MUSE_ARGS}"
    export ZENITH_VLLM_EXTRA_ENV="${_ZENITH_MUSE_ENV}${ZENITH_VLLM_EXTRA_ENV:+ ${ZENITH_VLLM_EXTRA_ENV}}"
    export ZENITH_CONTAINER_EXTRA_ENV="${_ZENITH_MUSE_ENV}${ZENITH_CONTAINER_EXTRA_ENV:+ ${ZENITH_CONTAINER_EXTRA_ENV}}"
    echo "lux: tuned defaults for ${_ZENITH_TUNED_MODEL} (${_ZENITH_MUSE}) on 1 node:" \
         "MAX_MODEL_LEN=${ZENITH_MAX_MODEL_LEN} KV_GB=${ZENITH_TARGET_KV_GB} MAX_NUM_SEQS=${ZENITH_MAX_NUM_SEQS}" \
         "MAX_NUM_BATCHED_TOKENS=${ZENITH_MAX_NUM_BATCHED_TOKENS:-vllm} DRAFT=${ZENITH_MUSE_DRAFT:-none}" \
         "EXTRA_ENV=${_ZENITH_MUSE_ENV} EXTRA_ARGS=${ZENITH_VLLM_EXTRA_ARGS}" \
         "(ZENITH_TUNED_DEFAULTS=0 for the generic ones)"
    unset _ZENITH_MUSE_ENV _ZENITH_MUSE_ARGS _ZENITH_MUSE_CG _ZENITH_MUSE_PAR _ZENITH_MUSE_K
fi
unset _ZENITH_MUSE
unset _ZENITH_TUNED_MODEL
