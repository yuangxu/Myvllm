# 从 logits 采样下一 token。
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: [batch, vocab]
        temperatures: [batch]，0 表示 greedy（argmax）。
        """
        if logits.dim() != 2:
            raise ValueError("logits must be [batch, vocab]")
        b = logits.size(0)
        if temperatures.numel() == 1:
            temperatures = temperatures.expand(b)
        if temperatures.shape[0] != b:
            raise ValueError("temperatures batch must match logits")

        out = torch.empty(b, dtype=torch.long, device=logits.device)
        for i in range(b):
            t = temperatures[i].item()
            row = logits[i : i + 1]
            if t == 0.0:
                out[i] = row.argmax(dim=-1).squeeze(0)
            else:
                probs = F.softmax(row / t, dim=-1)
                out[i] = torch.multinomial(probs.squeeze(0), num_samples=1)
        return out
