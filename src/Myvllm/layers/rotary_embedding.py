# RoPE：与 HF LLaMA/Qwen 系列一致的 inv_freq 形式。
from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """
        q, k: [total_tokens, num_heads, head_dim] 或 [total_tokens, num_kv_heads, head_dim]
        positions: [total_tokens]，每个 token 的绝对位置。
        """
        cos = self.cos_cached[positions].unsqueeze(1)
        sin = self.sin_cached[positions].unsqueeze(1)
        q_out = (q.float() * cos + _rotate_half(q.float()) * sin).to(q.dtype)
        k_out = (k.float() * cos + _rotate_half(k.float()) * sin).to(k.dtype)
        return q_out, k_out
