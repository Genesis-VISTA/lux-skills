# ring-flash-attn 0.1.8 calls _flash_attn_backward without rng_state (default None). The ROCm FlashAttention 2.8.3
# backward in the Primus image then fails with "Cannot access data pointer of Tensor that doesn't have storage".
# RingX passes torch.empty(2) and works. Supply it the same way. Explicit signature so get_default_args sees the args.
import torch
import flash_attn.flash_attn_interface as fai
import ring_flash_attn.ring_flash_attn as _r
import ring_flash_attn.zigzag_ring_flash_attn as _z
import ring_flash_attn.stripe_flash_attn as _s

_orig = fai._flash_attn_backward


def _flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal,
                         window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state=None):
    if rng_state is None:
        rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return _orig(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal,
                 window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state)


for m in (_r, _z, _s):
    m._flash_attn_backward = _flash_attn_backward
