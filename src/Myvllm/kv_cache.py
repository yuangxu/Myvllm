"""Simple dense KV cache for autoregressive generation (no paging)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class LayerCache:
    """Cached K/V for a single attention layer. Shape: [seq_len, num_kv_heads, head_dim]."""
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[0]

    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return full K/V history."""
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=0)
            self.v = torch.cat([self.v, v], dim=0)
        return self.k, self.v


@dataclass
class KVCache:
    """Per-layer KV cache for the full model."""
    num_layers: int
    layers: list[LayerCache] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layers:
            self.layers = [LayerCache() for _ in range(self.num_layers)]

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    def reset(self) -> None:
        self.layers = [LayerCache() for _ in range(self.num_layers)]
