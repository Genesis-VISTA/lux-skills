---
name: parallel-attention-skill
description: Choose, benchmark and integrate RingX context-parallel attention (ringX1 to ringX4, jqyin/ringX-attention) for long-sequence training on GPU clusters such as OLCF Lux or Frontier. Use when asked which context-parallel attention algorithm fits a model and machine, how to benchmark RingX kernels against ring or zigzag ring attention, or how to replace scaled_dot_product_attention or flash_attn_func with RingX in a ViT or GPT training loop.
---

# Parallel attention with RingX

RingX ([jqyin/ringX-attention](https://github.com/jqyin/ringX-attention), package `ringX_attn`) is a set of
context-parallel (CP) attention kernels. Each GPU holds a shard of the sequence, and the kernels exchange Q or KV
with collectives (broadcast, reduce, all-gather, reduce-scatter) instead of ring attention's point-to-point
send/recv. They are built on FlashAttention 2 and replace `scaled_dot_product_attention` / `flash_attn_func`
directly: inputs are `(batch, seqlen_local, nheads, head_dim)` plus a CP process group.

This skill has three parts, each in its own reference file:

1. **Choose an algorithm** from the table below → [`references/algorithm-selection.md`](references/algorithm-selection.md)
2. **Benchmark the kernels** with your dimensions on your machine → [`references/benchmarking.md`](references/benchmarking.md)
3. **Integrate the winner** into a training script → [`references/integration.md`](references/integration.md)

Nothing here is vendored: install RingX from upstream. The job scripts upstream are written for Frontier
(MI250X, Slingshot, `cxi`). On Lux (8x MI355X per node, ROCm/RCCL, Slurm partition `batch`), use the adaptations in
`references/benchmarking.md`. No RingX run has been made on Lux yet.

## Choosing an algorithm

```
bi-directional attention (ViT, BERT, encoders)?
 └─ yes → ringX1 (pipelined variant ringX1o)
causal attention (GPT, decoders)?
 ├─ one node, or a high-bandwidth, homogeneous network        → ringX4
 ├─ medium context (< ~100K tokens) that fits on a few nodes  → ringX2
 └─ many nodes / millions of tokens, and full KV fits per GPU → ringX3
then: benchmark the candidates (and ring/zigzag ring) at YOUR batch, sequence, heads and GPU count
```

| Algorithm | Attention | Communication | Load balancing | Context length | Best for |
|---|---|---|---|---|---|
| `ringX1` | bi-directional | broadcast(Q), all-reduce(lse), reduce(num, den) | n/a (pipelined) | unlimited | ViT/BERT, large Q/K/V tensors |
| `ringX2` | causal | same as ringX1 | unbalanced | unlimited | medium contexts (< ~100K) on a few nodes |
| `ringX3` | causal | all-gather(KV), reduce-scatter(dK, dV) | zigzag | bounded by per-GPU memory | many nodes, millions of tokens |
| `ringX4` | causal | broadcast(KV), reduce(dK, dV) | zigzag | unlimited | a single node or a high-bandwidth homogeneous network |

This is a heuristic, not a rule. Which kernel is fastest depends on collective bandwidth and message size, and those
change with the network, node count and tensor shape. The final choice comes from benchmarking with your own
dimensions (part 2).

Reported results (SC '25 paper, on Frontier):

- up to **3.4x** kernel speedup over ring attention;
- **~1.5x** end-to-end speedup for a ViT (climate) and for GPT (Llama3 8B);
- **38% MFU** for Llama3 8B at 1M tokens on 4,096 GPUs.

Evidence, the communication/memory table and the bandwidth crossover are in `references/algorithm-selection.md`.

## Quickstart

```bash
git clone https://github.com/jqyin/ringX-attention && cd ringX-attention
pip install -e .                                 # package ringX_attn; needs torch + flash_attn already installed
pip install ring-flash-attn                      # baselines the benchmark imports (ring_flash_attn); the paper used v0.1.1
# kernel benchmark (Frontier script; see references/benchmarking.md for Lux and other clusters)
cd script && sbatch submit_frontier.sh ringX1_attn
python ../benchmark/utils.py                     # parses log.algo* here into time and TFLOPS/GPU
```

In a model (bi-directional; for causal pass `causal=True` and zigzag-shard the sequence, see part 3):

```python
from ringX_attn import ringX1_attn_func
out = ringX1_attn_func(q, k, v, causal=False, group=cp_group)   # q,k,v: (b, s/N, h, d), same layout as flash_attn_func
```

## Gotchas

- **The rank order is the sequence order.** The kernels decide which blocks to mask from `dist.get_rank(group)`. For
  causal kernels, CP rank r must hold the right chunks: chunk r for ringX2, chunks r and 2N−1−r (zigzag) for ringX3
  and ringX4. Data, labels, position ids and RoPE must all be sharded the same way.
- **Causal-only and non-causal-only kernels.** ringX2's backward, ringX3, ringX4 and ringX4o assert `causal=True`.
  ringX4's assert message wrongly reads "ringX1 is intended for causal=False". ringX3b asserts `causal=False`.
  `ringX1o_varlen` raises `NotImplementedError` for causal. All of them default to `causal=False`.
- **ringX3 memory.** ringX3 all-gathers the full KV onto every GPU (O(hbs) memory), so the context it supports is
  bounded by one GPU's memory. ringX4 has the same communication volume but O(hbs/N) memory.
- **No masks or ALiBi.** There is no attention-mask argument, and `alibi_slopes` must be `None`.
- **Private FlashAttention API.** The kernels call `flash_attn.flash_attn_interface._flash_attn_forward/_backward`.
  A FlashAttention build that changes those signatures will break them.
- **The upstream benchmark as written.** `--module ringX1_attn` with `PYTHONPATH=../ringX_attn` likely fails,
  because the modules use relative imports. Without `--causal`, it stops at the zigzag baseline, which asserts
  causal, so plain ring is never timed. `benchmark/utils.py` assumes 5 iterations. Fixes are in
  `references/benchmarking.md`.

## Verify after any change

Compare the CP output and gradients against single-GPU `flash_attn_func` on the same (broadcast) inputs, as
RingX's `test/test_ringX*_attn_func.py` do. See "Numerical check" in `references/integration.md`.

## License and acknowledgments

The skill's own text is Apache-2.0 (`LICENSE`). RingX is MIT (Copyright 2025). The benchmark baselines come from
[ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) (zhuzilin). The ViT and GPT examples link to
[climate-vit](https://github.com/jqyin/climate-vit) and [forge](https://github.com/at-aaims/forge) (`xforge` branch).
None of them is copied here.

If you use RingX, cite:

```bibtex
@inproceedings{yin2025ringx,
  title     = {RingX: Scalable Parallel Attention for Long-Context Learning on HPC},
  author    = {Yin, Junqi and Palash, Mijanur and Shankar, Mallikarjun and Wang, Feiyi},
  booktitle = {Proceedings of the International Conference for High Performance
               Computing, Networking, Storage and Analysis (SC '25)},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3712285.3759859}
}
```
