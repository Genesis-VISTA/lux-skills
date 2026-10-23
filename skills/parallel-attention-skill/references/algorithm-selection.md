# Choosing a RingX algorithm

All figures and claims here come from the RingX SC '25 paper (DOI 10.1145/3712285.3759859) and the
[ringX-attention README](https://github.com/jqyin/ringX-attention). Every measurement was taken on **Frontier**.
A Frontier node has 4x MI250X (8 GCDs): 100 GB/s Infinity Fabric between GCDs (200 GB/s within one MI250X), and
Slingshot-11 at 100 GB/s per node. Software: PyTorch 2.4.0, FlashAttention 2.6.3, ROCm 6.1.3. Treat the numbers as
evidence for the heuristics, not as a prediction for another machine.

## How the kernels work

N is the CP group size. Each rank holds `s/N` tokens of Q, K and V, as `(b, s/N, h, d)`.

- **ringX1 / ringX2**: each step, one rank broadcasts its **Q** (step i = N−1 … 0). Every rank computes a partial
  attention of that Q against its local K/V. Then an all-reduce(MAX) of the log-sum-exp and a reduce of the partial
  numerators and denominators (to the owner of Q) combine the partials. Backward broadcasts KV and reduces dK/dV.
  Only Q moves forward, so the payload is (s/N)·h·b + 4(s/N)·b. That is about 75% less than ring's KV
  send/recv, or "~70% less communication than ring" in the paper's conclusion.
  - ringX1: bi-directional. `ringX1o` double-buffers Q to pipeline communication with compute; pipelining adds 5–10%.
  - ringX2: causal. Ranks skip blocks that are fully masked (rank > i), so the work is **unbalanced**: later ranks
    do more.
- **ringX3**: causal. Asynchronous all-gather of KV into N buffers, then reduce-scatter of dK/dV in backward.
  Zigzag sharding balances the causal work. Every GPU holds the full KV, so memory is O(hbs) and context length is
  bounded by per-GPU memory.
- **ringX4**: causal. Broadcasts KV one step at a time and reduces dK/dV, with zigzag sharding. It has ringX3's
  communication volume but O(hbs/N) memory.
- Others in `ringX_attn/__init__.py`, not in the paper's main comparison:
  - `ringX2o` and `ringX4o`: pipelined variants.
  - `ringX1o_varlen`: bi-directional, variable-length shards.
  - `ringX3b`: non-causal, 2-D collectives through the optional `pccl` package; it asserts world_size % 8 == 0.

## Communication and memory

Paper Table 2, for a sequence of s tokens, hidden size h, batch b, over N GPUs:

| Method | Collective | Volume | Memory | Context length |
|---|---|---|---|---|
| Megatron-style | all-gather(QKV) | 6(s/N)hb | O(shb) | limited |
| Ring attention | send-recv(KV) | 4(s/N)hb | O((s/N)hb) | unlimited |
| ringX1 / ringX2 | broadcast(Q), all-reduce(lse), reduce(num, den) | (s/N)hb + 4(s/N)b | O((s/N)hb) | unlimited |
| ringX3 | all-gather(KV), reduce-scatter(dKV) | 4(s/N)hb | O(shb) | limited |
| ringX4 | broadcast(KV), reduce(dKV) | 4(s/N)hb | O((s/N)hb) | unlimited |

Rule of thumb from the paper: with T the attention FLOPs, G the GPU rate, M the message size and B the collective
bandwidth, the kernel is communication-bound when T/G ≪ M/B. RingX wins by making M/B smaller, either through less
volume (ringX1/2) or by using collectives that run faster at that size and scale (ringX3/4).

## When each one wins (paper evidence)

**Bi-directional (paper Fig. 4).** Embedding 4096 (32 heads × 128); global sequence up to 2M.

- ringX1 beats ring attention in every configuration measured: up to **3.4x** at 1024 tokens per GPU on 32 GPUs,
  and up to 2x on a single node.
- At 8192 tokens per GPU, scaling efficiency to 256 GPUs is 85% for ringX1 against 72% for ring.
- ringX1 does best with large Q/K/V tensors (large batch and/or hidden size, as device memory permits).

**Causal (paper Fig. 5).**

- On **one node**, ringX4 is the fastest in every case, up to 2x over ring. The paper: "Broadcast and Reduce are
  preferred in a high-bandwidth and somewhat homogeneous network". ringX4 reached 84.5 TFLOPS/GPU at a 64K
  sequence, 44% of the MI250X peak.
- At **scale**, ringX3 is the fastest: up to **3.2x** over zigzag ring at 1024 tokens per GPU on 128 GPUs, and
  near-perfect scaling to 128 GPUs where ring falls to ~50%.
- **On one Lux node (8x MI355X), one measurement** (`benchmarking.md`): at b=1, 16K tokens/GPU, 32×128, ringX3
  (366 TFLOPS/GPU) and ringX2 (362) beat ringX4 (334) and ring (340). ringX4 was not the fastest there. Treat the
  "one node → ringX4" rule as Frontier-derived, and benchmark on Lux.
- **ringX2** beats ring at medium scale.
- Zigzag load balancing is worth about 10% over stripe.

**Where the time goes (Fig. 6).** Communication is ~80% of runtime for ring/stripe, against ~50% for ringX1/ringX3.

**The collective crossover (Fig. 7).** This is why the choice depends on the machine.

- On **8 GPUs** (one node), broadcast/reduce run slightly faster than all-gather/reduce-scatter for messages of a
  few hundred MB to 1 GB. Llama3 8B at 8192 tokens per GPU (32 heads × 128) is ~537 MB in FP32 at batch 1.
- On **32–64 GPUs**, all-gather/reduce-scatter are significantly faster than broadcast/reduce.
- So ringX4 (broadcast/reduce) suits a node or a fast homogeneous fabric, and ringX3 (all-gather/reduce-scatter)
  suits multi-node scale. Where the crossover falls on your machine depends on its intra-node and inter-node
  bandwidth and on your message size. Measure it (see `benchmarking.md`, "Collective bandwidth").

## End-to-end results

**ViT (climate-vit).** ERA5 360×720, 20M parameters, embedding 384, 8 heads.

- ringX1 is ~1.5x faster than Megatron-style CP at patch sizes 1 and 2, and up to 1.5x faster than ring at patch
  sizes 4 and 6.
- It uses ~50% less memory than Megatron-style CP.
- Scaling the embedding from 1024 to 4096 at a 64,800-token sequence: up to 1.4x over ring.

**GPT (Llama3 8B, forge `xforge`).** RedPajama books, bf16.

- At 128K tokens, ringX3 and ringX4 beat ring/stripe by ~1.3x and ~1.2x. ringX3 holds ~84 TFLOPS/GPU to 256 GPUs.
- At 1M tokens (CP 128), ringX3 is ~1.5x faster than stripe/ring. It scales at 94% efficiency to 4,096 GPUs,
  ~73 TFLOPS/GPU, **38% MFU**.

Paper Table 3 (ringX3):

| Sequence | CP | DP | TFLOPS/GPU | MFU |
|---|---|---|---|---|
| 128K | 16 | 8 | 85.1 | 44% |
| 256K | 32 | 4 | 77.6 | 40% |
| 512K | 64 | 2 | 78.1 | 41% |
| 1M | 128 | 1 | 77.5 | 40% |

## The paper's heuristics

From §5:

- Bi-directional attention: ringX1 with pipelining does well on large Q, K, V tensors (large batch sizes and/or
  hidden dimensions, as device memory permits).
- Causal attention: ringX2 is typically better for medium context lengths (e.g. < 100K) that fit on a few nodes.
  ringX3 is suited to pushing the context length to millions of tokens.
- From Fig. 5 and 7: ringX4 on a single node or a high-bandwidth, somewhat homogeneous network.

## Turning this into a choice

1. Fix the attention type. Causal needs ringX2/3/4; bi-directional needs ringX1 or ringX1o.
2. Work out the per-GPU shard size s/N and the KV size. If the full KV of one layer (2·s·h·b elements) fits in
   memory alongside the model, ringX3 is possible. If not, use ringX4 or ringX2.
3. Measure broadcast/reduce against all-gather/reduce-scatter at your message size and GPU count (rccl-tests or
   nccl-tests).
4. Benchmark the two or three candidates, plus ring/zigzag ring as a baseline, at your batch, s/N, heads and
   head_dim, at the CP size you will train with. Pick the fastest fwd+bwd time.
5. Confirm with a few end-to-end training steps. Kernel wins shrink end to end; the paper saw up to 3.4x for the
   kernel and ~1.5x end to end.
