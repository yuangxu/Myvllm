# 词嵌入 + LM Head；支持 tie_word_embeddings。
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class LMHead(nn.Module):
    """logits = hidden @ W^T，W 与词表维一致，形状 [vocab_size, hidden_size]。"""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tie_weight: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        if tie_weight is not None:
            self.weight = tie_weight
        else:
            self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
            nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight)


def last_token_hidden(
    hidden_states: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """Prefill：按累积长度取每条序列最后一个 token 的 hidden。"""
    ends = cu_seqlens[1:].long() - 1
    return hidden_states[ends]
