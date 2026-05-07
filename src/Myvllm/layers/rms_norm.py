# RMSNorm: output = (x / RMS(x)) * gamma，在最后维上归一化。
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        x_f = x_f * torch.rsqrt(var + self.eps)
        out = x_f * self.weight.float()
        return out.to(orig)
