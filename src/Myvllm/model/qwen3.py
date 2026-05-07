from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from Myvllm.config import ModelConfig
from Myvllm.layers.embedding_head import VocabEmbedding, LMHead, last_token_hidden
from Myvllm.layers.rms_norm import RMSNorm
from Myvllm.layers.rotary_embedding import RotaryEmbedding
from Myvllm.model.decoder_layer import Qwen3DecoderLayer


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 完整因果语言模型.

    结构:
        embed_tokens → N × DecoderLayer → norm → lm_head
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = VocabEmbedding(config.vocab_size, config.hidden_size)

        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )

        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
                rope=self.rope,
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        tie_weight = self.embed_tokens.weight if config.tie_word_embeddings else None
        self.lm_head = LMHead(
            config.hidden_size, config.vocab_size, tie_weight=tie_weight,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，返回最终 hidden_states（未做 logits 投影）.

        input_ids:  [total_tokens]  —— 拼平的 token id
        positions:  [total_tokens]  —— 每个 token 的绝对位置

        Returns: hidden_states [total_tokens, hidden_size]
        """
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """将 hidden_states 投影到 vocab logits.

        Prefill 模式下传入 cu_seqlens，只取每条序列最后一个 token 的 hidden；
        Decode 模式下 cu_seqlens=None，每行本身就是一个 token。

        Returns: logits [batch, vocab_size]
        """
        if cu_seqlens is not None:
            hidden_states = last_token_hidden(hidden_states, cu_seqlens)
        return self.lm_head(hidden_states)

    def load_weights(self, model_path: str) -> None:
        """从 HuggingFace safetensors 加载 Qwen3 预训练权重."""
        from safetensors.torch import load_file
        from pathlib import Path
        import glob as glob_mod

        path = Path(model_path)
        if path.is_dir():
            shard_files = sorted(glob_mod.glob(str(path / "*.safetensors")))
        else:
            shard_files = [str(path)]

        hf_state: dict[str, torch.Tensor] = {}
        for f in shard_files:
            hf_state.update(load_file(f))

        param_map = self._build_weight_map()
        loaded, skipped = 0, 0
        for hf_key, tensor in hf_state.items():
            local_key = self._translate_key(hf_key)
            if local_key is None:
                skipped += 1
                continue
            if local_key not in param_map:
                skipped += 1
                continue
            param = param_map[local_key]
            if param.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch: {hf_key} HF={tensor.shape} vs local={param.shape}"
                )
            param.data.copy_(tensor)
            loaded += 1

        print(f"[load_weights] loaded {loaded} tensors, skipped {skipped}")

    def _build_weight_map(self) -> dict[str, nn.Parameter]:
        """构建 local_key -> Parameter 的映射."""
        return {name: param for name, param in self.named_parameters()}

    @staticmethod
    def _translate_key(hf_key: str) -> Optional[str]:
        """将 HuggingFace 权重 key 转换为本模型的 key.

        HF key 样例:
            model.embed_tokens.weight           → embed_tokens.weight
            model.layers.0.self_attn.q_proj.weight → layers.0.self_attn.q_proj.weight
            model.layers.0.self_attn.q_norm.weight → layers.0.self_attn.q_norm.weight
            model.layers.0.mlp.gate_proj.weight → layers.0.mlp.gate_proj.weight
            model.layers.0.input_layernorm.weight → layers.0.input_layernorm.weight
            model.norm.weight                   → norm.weight
            lm_head.weight                      → lm_head.weight (tied → skip)
        """
        if hf_key == "lm_head.weight":
            return None

        if hf_key.startswith("model."):
            return hf_key[len("model."):]

        return hf_key
