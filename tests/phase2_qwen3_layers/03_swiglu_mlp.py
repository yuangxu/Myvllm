"""
Exercise 6: SwiGLU MLP
Goal: 实现 Qwen3 的 MLP block

公式：
  gate = gate_proj(x)      # (B, T, hidden) → (B, T, intermediate)
  up   = up_proj(x)        # (B, T, hidden) → (B, T, intermediate)
  act  = silu(gate) * up   # element-wise
  out  = down_proj(act)    # (B, T, intermediate) → (B, T, hidden)

SiLU: silu(x) = x * sigmoid(x)，PyTorch 里有 F.silu

写之前先在脑子里回答：
- 为什么要两个独立的投影（gate_proj 和 up_proj）而不是一个？
- SwiGLU 和普通 FFN+ReLU 的本质区别是什么？
- Qwen3 的线性层有没有 bias？
"""
