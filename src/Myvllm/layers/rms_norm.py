# RMSNorm 层
#
# 公式: output = (x / RMS(x)) * weight
#   其中 RMS(x) = sqrt(mean(x²) + eps)
#
# 作用：对每个 token 的 hidden_size 维度做归一化，再乘可学习的 weight
#
# 使用场景（Qwen3 中有两种）：
#   1. hidden_size=1024  → Decoder Layer 的 input_layernorm / post_attention_layernorm
#   2. head_dim=64       → Attention 中的 per-head QK Norm（每个头单独归一化）
#
# 实现步骤：
#   Step 1: __init__ 接收 hidden_size 和 eps，初始化 weight = nn.Parameter(ones)
#   Step 2: forward 中先把 x 转成 float32（防止 bf16 下 x² 溢出）
#   Step 3: 计算 variance = mean(x²)，注意 keepdim=True 保持 shape
#   Step 4: x_norm = x * rsqrt(variance + eps)  （rsqrt 即 1/sqrt，torch 有现成函数）
#   Step 5: 乘以 weight，转回输入原始 dtype 输出
#
# 注意：
#   - 用 x.float() 而不是 x.to(torch.float32)，写法更简洁
#   - rsqrt 比手写 1/sqrt 快，torch.rsqrt() 直接用
#   - weight 和 x_norm 相乘时，dtype 要一致

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class RMS_Norm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        variance = x.pow(2).mean(-1)








if __name__ == "__main__":
    model = RMS_Norm().cuda()
    input = torch.randn(8, 400, 4000).cuda()

    for _ in range(10):
        _ = model(input)

    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        output = model()
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print("time is %d", avg_time)









