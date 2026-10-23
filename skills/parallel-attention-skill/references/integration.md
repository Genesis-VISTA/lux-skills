# Integrating RingX into a training script

There are two worked examples upstream, and this file links to them rather than copying them:

- **ViT, bi-directional, ringX1**: [jqyin/climate-vit](https://github.com/jqyin/climate-vit) (read at `3e4dab6`).
- **GPT, causal, ringX3**: [at-aaims/forge](https://github.com/at-aaims/forge), branch `xforge` (read at `e8406cf`),
  GPT-NeoX based.

Both vendor their own copy of the kernel: `distributed/ringX_attn.py` in climate-vit (ringX1 with a double-buffered
Q, like ringX1o) and `train/megatron/ringX3_attn.py` in forge. A new integration should import from the
`ringX_attn` package instead.

## Prerequisites

- PyTorch with `torch.distributed` on NCCL (CUDA) or RCCL (ROCm).
- FlashAttention 2. RingX calls the private `flash_attn.flash_attn_interface._flash_attn_forward` and
  `_flash_attn_backward`. `ringX_attn/utils.py:get_default_args` adapts to the 2.6/2.7 argument names; another
  version may change them.
- RingX: `pip install -e .` in a clone of ringX-attention. Its `setup.py` declares no dependencies.
- DeepSpeed for the forge GPT example.

The paper's stack on Frontier (PyTorch 2.4.0, FlashAttention 2.6.3, DeepSpeed 0.15.1, ROCm 6.1.3, Python 3.12.0)
is a known-working reference, not a requirement. The README says FlashAttention was "tested with v2.7".

## The API

```python
from ringX_attn import ringX1_attn_func   # or ringX1o_, ringX2_, ringX2o_, ringX3_, ringX4_, ringX4o_attn_func
out = ringX1_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                       window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                       return_attn_probs=False, group=cp_group)
```

- q, k, v: `(batch, seqlen_local, nheads, head_dim)`, the layout `flash_attn_func` takes. They are not in the
  `(b, h, s, d)` layout that `scaled_dot_product_attention` takes.
- fp16/bf16, as FlashAttention requires.
- K/V may have fewer heads than Q (GQA/MQA). The kernels allocate buffers with `empty_like(k)`, and forge passes
  un-repeated KV. TODO: no upstream test covers GQA.
- `softmax_scale=None` means `head_dim ** -0.5`.
- It returns `out` with the same shape as q, or `(out, softmax_lse, None)` with `return_attn_probs=True`.
- `group` is the CP process group. The local rank in that group decides the mask, so it must match the sequence
  order (see "Sharding").

## 1. Build the CP process group

CP composes with data parallelism and tensor parallelism. Each CP group holds one copy of the model replica's
activations, split along the sequence.

- **Composing CP with TP.** TP shards heads, so CP sees `num_heads / tp` local heads. Data parallelism runs over
  the remaining dimension: `dp = world / (tp × cp × pp)`.
- **climate-vit, `utils/comm.py`.**
  - `init(params)` reads tp/cp/pp and asserts pp == 1.
  - `init_model_parallel_info` builds the groups `dp`, `tp`, `cp`, `pp`, `tp-cp` and `dp-cp` with a
    `RankGenerator(tp, dp, pp, cp, order)` and `dist.new_group`.
  - The layout comes from the CLI: `--tensor_parallel`, `--context_parallel` and `--parallel_order`
    (e.g. `tp-cp-dp`).
  - Layers fetch their group with `comm.get_group("cp")`.
- **forge, `megatron/mpu/initialize.py`.**
  - `initialize_model_parallel(..., context_parallel_size)` cuts each data-parallel group into consecutive slices of
    `context_parallel_size` ranks.
  - `world_size` must be divisible by `model_parallel_size × context_parallel_size`.
  - CP with pipeline parallelism raises "Context parallel not tested with pipeline parallelism".
  - `mpu.get_context_parallel_group()` returns the group.
- **From scratch.** Put CP ranks inside a node where possible, because CP traffic is the heaviest. With
  `cp | ranks_per_node` and rank-major placement:

  ```python
  world, rank = dist.get_world_size(), dist.get_rank()
  cp_group = None
  for start in range(0, world, cp):                 # every rank must call new_group for every group
      g = dist.new_group(list(range(start, start + cp)))
      if start <= rank < start + cp:
          cp_group = g
  dp_group = ...   # ranks with the same CP rank: range(rank % cp, world, cp)
  ```

## 2. Shard the sequence and un-shard the output

- **Bi-directional (ringX1, ringX1o).** Contiguous chunks work: rank r holds `x.chunk(N, dim=seq)[r]`.
  - climate-vit `networks/vit.py` computes
    `self.cp_shapes = compute_split_shapes(num_patches, comm.get_size("cp"))`.
  - It shards the patch tokens with `x = scatter_to_parallel_region(x, dim=1, comm_name="cp")` before the blocks
    (line 224).
  - It gathers them back with `x = gather_from_parallel_region(x, dim=1, shapes=self.cp_shapes, comm_name="cp")`
    after the final norm (line 232).
  - `ringX1o_varlen` supports uneven shards.
- **Causal, ringX2.** Contiguous chunks, in rank order. The work is unbalanced by design.
- **Causal, ringX3 / ringX4: zigzag.** Split the sequence into 2N chunks; rank i holds chunks i and 2N−1−i
  (0-indexed). This is the code of forge's `megatron/mpu/data.py:zigzag_data`:

  ```python
  def zigzag_data(data, seq_dim=1):
      worldsize = get_context_parallel_world_size()
      if worldsize == 1:
          return data
      seq_chunks = torch.chunk(data, 2 * worldsize, dim=seq_dim)
      data = [torch.cat((seq_chunks[i], seq_chunks[-(i + 1)]), dim=seq_dim) for i in range(worldsize)]
      return data[get_context_parallel_rank()].contiguous()
  ```

  - The sequence length must be divisible by 2N.
  - Apply the same permutation to **everything indexed by position**: tokens, labels, loss mask, position ids, and
    the attention mask if there is one. forge's `megatron/training.py:_get_batch` calls `mpu.zigzag_data` on
    tokens, labels, loss mask and position ids, and `mpu.zigzag_data(attention_mask, -2)` on the mask.
  - It also applies to **rotary embeddings**. forge's `RotaryEmbedding(..., zigzag=True)` in
    `megatron/model/positional_embeddings.py:_prepare_cache` slices the position range `t` the same way.
- **Un-sharding a zigzag output** (for evaluation or the numerical check): all-gather the local chunks, then put
  each rank's two halves back in their places.

  ```python
  def unzigzag(local, cp_group, dim=1):
      n = dist.get_world_size(cp_group)
      parts = [torch.empty_like(local) for _ in range(n)]
      dist.all_gather(parts, local.contiguous(), group=cp_group)
      chunks = [None] * (2 * n)
      for r, p in enumerate(parts):
          chunks[r], chunks[2 * n - 1 - r] = p.chunk(2, dim=dim)
      return torch.cat(chunks, dim=dim)
  ```

  A language model trained this way does not need the output un-sharded: each rank computes the loss on its own
  tokens and labels, and the losses are combined over the CP group (step 4).

## 3. Replace the attention call

### ViT (climate-vit), `scaled_dot_product_attention` → ringX1

Before, `networks/vit.py` `Attention.forward`:

```python
B, N, C = x.shape
q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
x = x.transpose(1, 2).reshape(B, N, C)
x = self.proj(x)
```

After, `distributed/layers.py` `DistributedAttention.forward` (lines 209–257; group set at line 206):

```diff
-B, N, C = x.shape
+# note: N is local sequence shard if CP is on
+B, N, C = x.shape
-q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
-k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
-v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
-x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
-x = x.transpose(1, 2).reshape(B, N, C)
+q = self.q(x).reshape(B, N, self.num_heads_local, self.head_dim).to(torch.bfloat16)   # no permute
+k = self.k(x).reshape(B, N, self.num_heads_local, self.head_dim).to(torch.bfloat16)
+v = self.v(x).reshape(B, N, self.num_heads_local, self.head_dim).to(torch.bfloat16)
+dtype = x.dtype
+x = ringX_attn_func(q, k, v, causal=False, dropout_p=self.attn_drop.p, group=self.cp_group)
+x = x.reshape(B, N, self.num_heads_local * self.head_dim).to(dtype)
 x = self.proj(x)
```

- `self.cp_group = comm.get_group(self.comm_cp_name)`.
- `num_heads_local = num_heads / tp`.
- The import is `from .ringX_attn import ringX_attn_func`; in a new integration, use
  `from ringX_attn import ringX1_attn_func` (or `ringX1o_attn_func`).
- The algorithm is picked by `attn_type: ringX | ring | megatron` in `config/ViT.yaml`; see the `mp_p{1,2,4,6}_*`
  and `mp_emb{1024,2048,4096}_*` configs.
- Frontier launch: `submit_frontier_cp.sh` (N=8, cp=64, tp=1) runs
  `python -u train_mp.py --config=mp_p${patch}_${algo} --tensor_parallel=$tp --context_parallel=$cp --parallel_order=tp-cp-dp`.

### GPT (forge `xforge`), `flash_attn_func` → ringX3

Before, `train/megatron/model/transformer.py` `flash_attention` (line 643):

```python
output = self.flash_qkv_fn(query_layer, key_layer, value_layer,
                           self.dropout_p if self.training else 0.0,
                           softmax_scale=None, causal=True, **extra_kwargs)
```

After, `ring_attention` (line 762). The inputs are transposed `[s, b, np, hn] → [b, s, np, hn]` exactly as in the
flash path, with KV reshaped to `num_kv_heads_per_partition` heads:

```diff
-output = self.flash_qkv_fn(query_layer, key_layer, value_layer,
-                           self.dropout_p if self.training else 0.0,
-                           softmax_scale=None, causal=True, **extra_kwargs)
+output = self.ring_attn_fn(query_layer, key_layer, value_layer,
+                           self.dropout_p if self.training else 0.0,
+                           softmax_scale=None, causal=True,
+                           group=mpu.get_context_parallel_group(), **extra_kwargs)
```

The surrounding wiring in the same file:

- line 31: `from megatron.ringX3_attn import ringX_attn_func`;
- line 471: `self.use_ring_attention = self.attention_type == "ring"`;
- line 513: `self.ring_attn_fn = ringX_attn_func`, with the zigzag ring alternative commented out on line 514;
- line 927: GQA `repeat_interleave` is skipped when `use_flash_attention or use_ring_attention`;
- line 1038: dispatch to `self.ring_attention(...)`.

Configuration, in `train/configs/xforge/llama3-8b-1m.yml`:

- `"context_parallel_size": 128`;
- `"attention_config": [[["ring"], 32]]`;
- 32 heads, 8 KV heads, `seq_length` 1048576;
- `pipe_parallel_size` 0, `model_parallel_size` 1, ZeRO stage 2, bf16.

`neox_arguments/arguments.py` enforces "Context parallel requires ring attention!". Sibling configs cover 8k to
512k. The Frontier launch is `cd train; sbatch job.sb xforge/llama3-8b-1m` (`job.sb` runs `deepy.py train.py -d
configs <model>.yml frontier.yml`).

To use ringX4 or ringX2 instead, change the import. ringX4 also takes zigzag-sharded data. ringX2 takes contiguous
data, so turn zigzag off, both `zigzag_data` and the rotary `zigzag=True`.

## 4. The rest of the training loop

- **Loss over CP.** Each rank sees only its tokens. forge `model/gpt2_model.py` (around line 78) all-reduces
  `loss_mask_sum` and then the masked loss over the CP group, so every CP rank holds the same global token-mean
  loss.
- **Gradients.**
  - Parameters are replicated across a CP group, but each rank's gradient covers only its own tokens, so gradients
    must be summed over CP as well as DP.
  - climate-vit: `distributed/mappings.py:init_ddp_model_and_reduction_hooks` wraps the model in DDP over `dp`. A
    comm hook SUM-all-reduces the parameters marked `mark_for_reduction` with the CP group: weights flagged
    `is_shared_mp` in `distributed/layers.py`.
  - forge builds each CP group as a slice of a data-parallel group (`mpu/initialize.py`), so DeepSpeed's
    data-parallel gradient all-reduce already spans the CP ranks.
  - forge also builds an IO group from the first rank of each CP group.
  - Whatever the framework, check that every parameter's gradient is reduced over CP exactly once.
- **Data loading.** The sampler shards over DP only: climate-vit uses `comm.get_size("dp")` / `get_rank("dp")`.
  Every rank in a CP group must load the **same** sample and cut its own shard.

## Numerical check

Before training, compare one CP forward/backward against single-GPU FlashAttention on the same full tensors, as
`test/test_ringX1_attn_func.py` (contiguous) and `test/test_ringX3_attn_func.py` (zigzag) do upstream:

1. Seed, then create full `q, k, v, dout` of shape `(b, s, h, d)` in bf16, and `dist.broadcast` them from rank 0.
2. Reference: `out, lse, _ = flash_attn_func(q, k, v, causal=..., return_attn_probs=True)`; `out.backward(dout)`.
3. Shard with the layout of the kernel under test:
   - contiguous: `x.chunk(N, dim=1)[rank]`;
   - zigzag: `extract_local`, which takes chunks `rank` and `2N−rank−1` of `x.chunk(2N, dim=1)`.

   Detach, set `requires_grad`, and run the RingX kernel with `return_attn_probs=True`, then `.backward(local_dout)`.
4. Compare `out`, `lse` and `dq`/`dk`/`dv` with the same shard of the reference: max and mean absolute difference.
   Shard lse on its sequence dimension: `dim=-1` for contiguous in the ringX1 test, `dim=2` for zigzag in the
   ringX3 test.

Running the upstream tests:

- They import `from ringX3_attn import ...` and `from utils import log, set_seed`. ringX-attention has no
  `test/utils.py`. The functions match ring-flash-attention's `test/utils.py` (`set_seed(rank, seed=42)`,
  `log(msg, a, rank0_only=False)`). Copy that file next to the test, and import the kernel from `ringX_attn`, or use
  the shim from `benchmarking.md`.
- They use `device = cuda:{rank}`, so they run as written only on **one node** with all GPUs visible to every
  rank. The README runs them as `srun -n8 bash -c "source setup_dist_vars.sh; python ../test/test_ringX1_attn_func.py"`.
- Tests exist for ringX1, ringX1o_varlen, ringX2 and ringX3. For ringX4, adapt the ringX3 test (both use zigzag),
  and pass `group=dist.group.WORLD`: ringX4 fails on `group=None` (the ringX3 test passes none).

Expect bf16-level differences. Upstream gives no tolerance. Measured on one Lux node (8 ranks, b=1, s≈3.8K,
5 heads × 128, bf16; worst rank):

| Kernel | out | lse | dq | dk | dv |
|---|---|---|---|---|---|
| ringX1 (bi-dir) | 9.8e-4 | 3.8e-6 | 9.8e-4 | 9.8e-4 | 9.8e-4 |
| ringX2 | 2.0e-3 | 2.9e-6 | 2.0e-3 | 1.6e-2 | 1.6e-2 |
| ringX3 | 4.9e-4 | 3.8e-6 | 2.0e-3 | 3.1e-2 | 3.1e-2 |
| ringX4 | 4.9e-4 | 3.8e-6 | 2.0e-3 | 1.6e-2 | 1.6e-2 |

These are max absolute differences, against outputs of scale ~4 and lse ~9. The causal tests compare only one
sequence position per gradient (`dq[:, 0]`, `dk[:, 1]`, `dv[:, 2]`). The largest dk/dv errors are on rank 0, whose
first positions are attended by every later token, and they fall to ≤2e-3 on the other ranks. Mean differences
are 2e-3 or less. Use these as a reference scale, and
compare with ring-flash-attention on the same inputs if in doubt. Re-run the check whenever the CP group, sharding
or FlashAttention version changes. Then run a few training steps with CP = 1 and CP = N on the same data, and
compare the losses.

## Pitfalls

- **Rank order is sequence order.** `dist.get_rank(group)` decides which blocks are masked. If the CP group's
  ranks are not in sequence order, or the data was sharded for a different layout, causal attention is silently
  wrong. Contiguous for ringX1/ringX2, zigzag for ringX3/ringX4. Check this with the numerical test above.
- **Every position-indexed tensor uses the same sharding.** That means labels, loss mask, position ids, RoPE and
  any mask. Missing RoPE is the classic bug: forge needs `zigzag=True` on the rotary embedding.
- **Causal flags.** ringX2's backward, ringX3, ringX4 and ringX4o assert `causal=True`. ringX4's message wrongly
  says "ringX1 is intended for causal=False". ringX3b asserts `causal=False`. ringX1o_varlen raises
  `NotImplementedError` for causal. The default everywhere is `causal=False`, so pass `causal=True` explicitly for
  decoders.
- **No masks.** There is no `attn_mask` argument; only causal, `window_size` and dropout are supported.
  - Padding masks and document masks are not supported, apart from ringX1o_varlen's variable shard lengths.
  - forge sets `requires_attention_mask` to False when every layer is ring or flash attention.
  - `alibi_slopes` must be `None` (asserted), although forge's `ring_attention` would pass slopes when
    `pos_emb == "alibi"`.
- **ringX3 memory.** ringX3 all-gathers KV into `world_size` buffers of `2 × b × s/N × h_kv × d`, so it holds the
  full KV, O(hbs), during forward and again during backward. With long contexts and a large CP group, this is what
  runs out of memory first. Use ringX4 (same volume, O(hbs/N)) when it does.
- **Inference path in forge.** In eval mode, `ring_attention` sets `causal = (max_seqlen_q == max_seqlen_k)`, so
  incremental decoding with a KV cache passes `causal=False` to a causal-only kernel and fails its assert.
  TODO: check before using it for generation.
- **Divisibility.** For zigzag, the global sequence must be divisible by 2N; the benchmark also asserts
  `seq_length % (2 × N) == 0` per GPU. head_dim must be a multiple of 8.
- **Upstream docs.** The README's ViT inference line names `inference-cp8.py`, which is not in climate-vit; the repo
  has `inference.py` and `inference_mp.py`.
