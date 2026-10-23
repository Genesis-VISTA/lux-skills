# 1 GPU: known-FLOP workloads to check the MFMA counter scale. bf16 GEMM 8192^3 x2, then flash_attn fwd+bwd once.
import sys, torch
from flash_attn import flash_attn_func
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
for _ in range(2): c = a @ b
torch.cuda.synchronize()
B, s, H, d, causal = 1, 16384, 32, 128, sys.argv[1] == "1"
q, k, v = (torch.randn(B, s, H, d, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
out = flash_attn_func(q, k, v, causal=causal); out.backward(torch.randn_like(out)); torch.cuda.synchronize()
print("CALIB gemm_flops", 2 * 2 * 8192**3, "fa_flops_formula", 4 * B * s * s * H * d * 3.5 / (2 if causal else 1))
