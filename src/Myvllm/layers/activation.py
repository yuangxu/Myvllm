# SiLU 激活函数
# Qwen3 MLP 使用 SwiGLU 变体：output = silu(gate) * x
# silu(x) = x * sigmoid(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        x, y = x.chunk(2, -1)
        return F.silu(x) * y



if __name__ == "__main__":
    # warm
    model = SiluAndMul().to("cuda")
    x = torch.randn(8, 4000, 8000, device="cuda")

    for _ in range(10):
        _ = model(x)
        
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        out = model(x)
        torch.cuda.synchronize()

        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Avg time over 100 runs: {avg_time * 1000:.4f} ms")
