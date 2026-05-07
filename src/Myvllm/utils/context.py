# ForwardContext：model_runner ↔ attention，避免每层传 KV 相关张量。
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class ForwardContext:
    is_prefill: bool
    slot_mapping: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None


_ctx: Optional[ForwardContext] = None


def set_context(**kwargs: Any) -> None:
    global _ctx
    _ctx = ForwardContext(**kwargs)


def get_context() -> Optional[ForwardContext]:
    return _ctx


def reset_context() -> None:
    global _ctx
    _ctx = None
