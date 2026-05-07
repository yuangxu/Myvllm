# GQA Attention + RoPE + per-head RMSNorm（Q/K）；支持写 paged KV 与 decode gather。
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Myvllm.layers.linear import Linear
from Myvllm.layers.rotary_embedding import RotaryEmbedding
from Myvllm.layers.rms_norm import RMSNorm
from Myvllm.utils.context import ForwardContext, get_context


class Attention(nn.Module):
    """Qwen3 风格：无前缀 cache 时等价于单次因果注意力（便于单机测试）。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope: RotaryEmbedding,
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads
        self.rope = rope

        q_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim

        self.q_proj = Linear(hidden_size, q_out, bias=False)
        self.k_proj = Linear(hidden_size, kv_out, bias=False)
        self.v_proj = Linear(hidden_size, kv_out, bias=False)
        # HF: [hidden, num_heads * head_dim]（如 1024×2048），聚合资拼接维为 num_heads * head_dim
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # 由外部（ModelRunner.allocate_kv_cache）注入 [num_blocks, block_size, n_kv, head_dim]
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def set_kv_cache(
        self,
        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
    ) -> None:
        self.k_cache = k_cache
        self.v_cache = v_cache

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=1)

    @staticmethod
    def _apply_norm_heads(x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
        t, nh, hd = x.shape
        x2 = x.reshape(t * nh, hd)
        y = norm(x2)
        return y.view(t, nh, hd)

    def _write_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        assert self.k_cache is not None and self.v_cache is not None
        nk, hd = k.shape[1], k.shape[2]
        k_flat = self.k_cache.view(-1, nk, hd)
        v_flat = self.v_cache.view(-1, nk, hd)
        idx = slot_mapping.long()
        k_flat[idx] = k.to(k_flat.dtype)
        v_flat[idx] = v.to(v_flat.dtype)

    def _gather_seq_kv_decode(
        self,
        seq_idx: int,
        seq_len: int,
        ctx: ForwardContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert ctx.block_tables is not None and self.k_cache is not None
        blk_size = self.k_cache.shape[1]
        nkv = self.k_cache.shape[2]
        hd = self.k_cache.shape[3]
        tbl = ctx.block_tables[seq_idx]

        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for pos in range(seq_len):
            bidx = pos // blk_size
            off = pos % blk_size
            blk_id = int(tbl[bidx].item())
            if blk_id < 0:
                raise ValueError("invalid block_tables entry")
            slot = blk_id * blk_size + off
            ks.append(self.k_cache.view(-1, nkv, hd)[slot])
            vs.append(self.v_cache.view(-1, nkv, hd)[slot])  # type: ignore[union-attr]
        return torch.stack(ks, dim=0), torch.stack(vs, dim=0)

    def _forward_prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)

        q = self._apply_norm_heads(q, self.q_norm)
        k = self._apply_norm_heads(k, self.k_norm)
        q, k = self.rope.forward(q, k, positions)

        if self.k_cache is not None and ctx.slot_mapping is not None:
            self._write_kv(k, v, ctx.slot_mapping)

        assert ctx.cu_seqlens is not None
        k_e = self._repeat_kv(k)
        v_e = self._repeat_kv(v)

        out = torch.empty_like(q)
        cu = ctx.cu_seqlens
        for i in range(cu.numel() - 1):
            s, e = int(cu[i].item()), int(cu[i + 1].item())
            qi = q[s:e].transpose(0, 1).unsqueeze(0)
            ki = k_e[s:e].transpose(0, 1).unsqueeze(0)
            vi = v_e[s:e].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)
            out[s:e] = oi.squeeze(0).transpose(0, 1).contiguous()

        return self.o_proj(out.reshape(t, self.num_heads * self.head_dim))

    def _forward_decode(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        b = x.shape[0]
        q = self.q_proj(x).view(b, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, self.num_kv_heads, self.head_dim)

        q = self._apply_norm_heads(q, self.q_norm)
        k = self._apply_norm_heads(k, self.k_norm)
        q, k = self.rope.forward(q, k, positions)

        assert ctx.slot_mapping is not None and ctx.context_lens is not None
        self._write_kv(k, v, ctx.slot_mapping)

        assert self.k_cache is not None
        out = torch.empty(b, self.num_heads, self.head_dim, device=x.device, dtype=q.dtype)
        for bi in range(b):
            L = int(ctx.context_lens[bi].item())
            kh, vh = self._gather_seq_kv_decode(bi, L, ctx)
            kh = self._repeat_kv(kh)
            vh = self._repeat_kv(vh)
            qh = q[bi : bi + 1].transpose(0, 1).unsqueeze(0)
            ki = kh.transpose(0, 1).unsqueeze(0)
            vi = vh.transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qh, ki, vi, is_causal=False)
            out[bi] = oi.squeeze(0).squeeze(1)

        return self.o_proj(out.reshape(b, self.num_heads * self.head_dim))

    def _forward_simple_causal(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """无 ForwardContext 时：单段因果注意力（总 token 为一段）。"""
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)
        q = self._apply_norm_heads(q, self.q_norm)
        k = self._apply_norm_heads(k, self.k_norm)
        q, k = self.rope.forward(q, k, positions)
        k_e = self._repeat_kv(k)
        v_e = self._repeat_kv(v)
        qi = q.transpose(0, 1).unsqueeze(0)
        ki = k_e.transpose(0, 1).unsqueeze(0)
        vi = v_e.transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)
        o = o.squeeze(0).transpose(0, 1).contiguous().reshape(
            t, self.num_heads * self.head_dim
        )
        return self.o_proj(o)

    def forward_cached(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        layer_cache: "LayerCache",
    ) -> torch.Tensor:
        """带 KV cache 的前向：prefill 时 is_causal=True，decode 时 is_causal=False。"""
        from Myvllm.kv_cache import LayerCache  # noqa: F811

        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)

        q = self._apply_norm_heads(q, self.q_norm)
        k = self._apply_norm_heads(k, self.k_norm)
        q, k = self.rope.forward(q, k, positions)

        k_full, v_full = layer_cache.append(k, v)

        k_e = self._repeat_kv(k_full)
        v_e = self._repeat_kv(v_full)

        qi = q.transpose(0, 1).unsqueeze(0)
        ki = k_e.transpose(0, 1).unsqueeze(0)
        vi = v_e.transpose(0, 1).unsqueeze(0)

        is_causal = (t == k_full.shape[0])
        o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=is_causal)
        o = o.squeeze(0).transpose(0, 1).contiguous().reshape(
            t, self.num_heads * self.head_dim
        )
        return self.o_proj(o)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        if ctx is None:
            return self._forward_simple_causal(x, positions)
        if ctx.is_prefill:
            return self._forward_prefill(x, positions, ctx)
        return self._forward_decode(x, positions, ctx)
