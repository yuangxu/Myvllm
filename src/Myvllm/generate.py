"""Autoregressive text generation with KV cache."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from Myvllm.config import ModelConfig
from Myvllm.kv_cache import KVCache
from Myvllm.model.qwen3 import Qwen3ForCausalLM


@dataclass
class SamplingParams:
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 256
    eos_token_id: Optional[int] = None


@torch.inference_mode()
def generate(
    model: Qwen3ForCausalLM,
    input_ids: list[int],
    sampling_params: SamplingParams,
    *,
    stream: bool = False,
) -> list[int]:
    """Generate tokens autoregressively with KV cache.

    Args:
        model: loaded Qwen3ForCausalLM (already on device)
        input_ids: prompt token ids
        sampling_params: temperature / max_new_tokens / eos
        stream: if True, yield token ids one by one (generator)

    Returns:
        List of generated token ids (excluding prompt).
    """
    device = next(model.parameters()).device
    cfg = model.config
    cache = KVCache(num_layers=cfg.num_hidden_layers)

    prompt_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    positions = torch.arange(len(input_ids), dtype=torch.long, device=device)

    # --- Prefill: process all prompt tokens at once ---
    hidden = _forward_with_cache(model, prompt_tensor, positions, cache)
    logits = model.lm_head(hidden[-1:])  # only last token
    next_id = _sample(logits, sampling_params)

    generated: list[int] = [next_id]

    # --- Decode: one token at a time ---
    for _ in range(sampling_params.max_new_tokens - 1):
        if sampling_params.eos_token_id is not None and next_id == sampling_params.eos_token_id:
            break

        token_tensor = torch.tensor([next_id], dtype=torch.long, device=device)
        pos_tensor = torch.tensor([cache.seq_len], dtype=torch.long, device=device)

        hidden = _forward_with_cache(model, token_tensor, pos_tensor, cache)
        logits = model.lm_head(hidden)
        next_id = _sample(logits, sampling_params)
        generated.append(next_id)

    return generated


@torch.inference_mode()
def generate_stream(
    model: Qwen3ForCausalLM,
    input_ids: list[int],
    sampling_params: SamplingParams,
):
    """Streaming variant — yields one token id at a time."""
    device = next(model.parameters()).device
    cfg = model.config
    cache = KVCache(num_layers=cfg.num_hidden_layers)

    prompt_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    positions = torch.arange(len(input_ids), dtype=torch.long, device=device)

    hidden = _forward_with_cache(model, prompt_tensor, positions, cache)
    logits = model.lm_head(hidden[-1:])
    next_id = _sample(logits, sampling_params)
    yield next_id

    for _ in range(sampling_params.max_new_tokens - 1):
        if sampling_params.eos_token_id is not None and next_id == sampling_params.eos_token_id:
            return

        token_tensor = torch.tensor([next_id], dtype=torch.long, device=device)
        pos_tensor = torch.tensor([cache.seq_len], dtype=torch.long, device=device)

        hidden = _forward_with_cache(model, token_tensor, pos_tensor, cache)
        logits = model.lm_head(hidden)
        next_id = _sample(logits, sampling_params)
        yield next_id


def _forward_with_cache(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    cache: KVCache,
) -> torch.Tensor:
    """Run model forward using per-layer KV cache."""
    hidden = model.embed_tokens(input_ids)
    for i, layer in enumerate(model.layers):
        hidden = layer.forward_cached(hidden, positions, cache.layers[i])
    hidden = model.norm(hidden)
    return hidden


def _sample(logits: torch.Tensor, params: SamplingParams) -> int:
    """Sample a single token from logits [1, vocab]."""
    logits = logits.squeeze(0).float()
    if params.temperature == 0.0:
        return logits.argmax(dim=-1).item()

    logits = logits / params.temperature

    if params.top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = probs.cumsum(dim=-1)
        mask = cumsum - probs > params.top_p
        sorted_logits[mask] = float("-inf")
        probs = torch.softmax(sorted_logits, dim=-1)
        chosen = torch.multinomial(probs, num_samples=1)
        return sorted_idx[chosen].item()

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
