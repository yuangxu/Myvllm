# RMSNorm
# 公式：x / sqrt(mean(x²) + eps) * weight
# 注意：计算在 float32 下进行，防止 bf16/fp16 溢出
# weight shape: [hidden_size] 或 [head_dim]（用于 Qwen3 的 per-head QK Norm）
