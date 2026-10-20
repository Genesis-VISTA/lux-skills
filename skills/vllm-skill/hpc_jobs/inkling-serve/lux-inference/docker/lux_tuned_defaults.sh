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
# Three configurations have defaults here: Inkling-MXFP4 on 2 nodes (the
# submit script's default, sized for the full 1M-token context) and on 1 node
# (tuned for throughput), and Mistral-Large-Instruct-2411 on 1 node at 1M
# tokens. Every other model and node count gets the generic defaults.
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
unset _ZENITH_TUNED_MODEL
