from __future__ import annotations

import torch
import torch.nn as nn

from Myvllm.layers.activation import SiluAndMul
from Myvllm.layers.linear import Linear


class Qwen3MLP(nn.Module):
    """Qwen3 SwiGLU MLP.

    output = down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate_up = torch.cat([gate, up], dim=-1)
        return self.down_proj(self.act(gate_up))
