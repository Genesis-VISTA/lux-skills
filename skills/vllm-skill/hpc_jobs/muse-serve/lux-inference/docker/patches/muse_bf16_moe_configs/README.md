Triton fused_moe configs for Muse Spark bf16 on MI355X, from bench/muse/bf16_moe_tune.py (lux job 350079).
Served with VLLM_TUNED_CONFIG_FOLDER=/opt/zenith_patches/muse_bf16_moe_configs (this dir as the container sees it).
E=256,N=512: TP8 per rank. E=32,N=4096: EP8 per rank.
