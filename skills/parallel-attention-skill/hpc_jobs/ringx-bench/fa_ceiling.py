# 1 GPU: flash_attn_func fwd+bwd at the given shapes, the single-GPU compute reference for the CP kernels.
#   python fa_ceiling.py b,s,H,d,causal [...]        (causal is 0 or 1)
# Counts the real attention FLOPs (4*b*s^2*H*d forward, x3.5 with the 2.5x backward; /2 if causal). A failed shape
# (e.g. out of memory) prints CEILING ... error=<reason> and the next shape still runs.
import sys

import torch
from flash_attn import flash_attn_func

for spec in sys.argv[1:]:
    b, s, H, d, causal = spec.split(",")
    b, s, H, d, causal = int(b), int(s), int(H), int(d), causal == "1"
    tag = f"CEILING b={b} s={s} H={H} d={d} causal={int(causal)}"
    try:
        q, k, v = (torch.randn(b, s, H, d, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        dout = torch.randn_like(q)
        for _ in range(2):
            flash_attn_func(q, k, v, causal=causal).backward(dout)
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        n = 5
        t0.record()
        for _ in range(n):
            flash_attn_func(q, k, v, causal=causal).backward(dout)
        t1.record()
        torch.cuda.synchronize()
        t = t0.elapsed_time(t1) / 1e3 / n
        real = 4 * b * s * s * H * d * 3.5 / (2 if causal else 1)
        print(f"{tag} ms={t * 1e3:.2f} tflops={real / t / 1e12:.1f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{tag} error={type(e).__name__}", flush=True)
    finally:
        q = k = v = dout = None
        torch.cuda.empty_cache()
