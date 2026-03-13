# RMSNorm
# 公式：x / sqrt(mean(x²) + eps) * weight
# 注意：计算在 float32 下进行，防止 bf16/fp16 溢出
# weight shape: [hidden_size] 或 [head_dim]（用于 Qwen3 的 per-head QK Norm）



import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class LayerNorm(nn.Module):
    





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




