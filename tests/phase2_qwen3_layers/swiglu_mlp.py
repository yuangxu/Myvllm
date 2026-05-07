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

import torch
import torch.nn as nn
import torch.nn.functional as F


class MySwiGLU(nn.Module):
	def __init__(self, hidden, intermediate):
		super().__init__()
		self.gate = nn.Linear(hidden, intermediate, bias= False)
		self.up = nn.Linear(hidden, intermediate, bias=False)
		self.down = nn.Linear(intermediate, hidden, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.down(F.silu(self.gate(x)) * self.up(x))


if __name__ == "__main__":
    model = MySwiGLU(256, 512)
    x = torch.randn(2, 4, 256)
    print(model(x).shape)  # 应该是 (2, 4, 256)




