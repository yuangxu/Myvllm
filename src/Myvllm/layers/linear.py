# 线性层（包装 nn.Linear）
# Qwen3 线性层无 bias（q/k/v/o、gate/up/down）
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """与 nn.Linear 一致，便于后续换量化或 TP split。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
