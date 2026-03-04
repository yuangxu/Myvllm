# 线性层（包装 nn.Linear）
# Qwen3 所有线性层均无 bias（q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj）
# 接口统一，方便后续替换为量化版本或 tensor parallel 版本
