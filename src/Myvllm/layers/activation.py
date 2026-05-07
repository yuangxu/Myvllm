# SiLU / SwiGLU 中的 silu(gate) * up
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)


class SiluAndMul(nn.Module):
    """输入最后一维为 2 * intermediate，前半为 gate，后半为 up。"""

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b
