"""
Practice exercises — run on H100:
    python test.py

Each exercise has a # YOUR CODE HERE block.
Fill them in top to bottom. The verification at the bottom
runs all exercises and prints PASS/FAIL for each.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1: Linear
# Formula: y = x @ W.T + b
# ══════════════════════════════════════════════════════════════════════════════

class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 2: Softmax
# Formula: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
# ══════════════════════════════════════════════════════════════════════════════

class MySoftmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 3: Embedding
# Core idea: lookup table — weight[token_ids]
# ══════════════════════════════════════════════════════════════════════════════

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 4: RMSNorm
# Formula: x / sqrt(mean(x^2) + eps) * weight
# ══════════════════════════════════════════════════════════════════════════════

class MyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 5: RoPE
# rotate_half([x1, x2]) = [-x2, x1]
# RoPE(x) = x * cos + rotate_half(x) * sin
# ══════════════════════════════════════════════════════════════════════════════

def precompute_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Returns cos, sin — each shape (max_seq_len, head_dim)"""
    # YOUR CODE HERE
    pass


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half, return [-x2, x1]"""
    # YOUR CODE HERE
    pass


def apply_rope(q, k, cos, sin):
    """
    q: (B, T, n_heads, head_dim)
    k: (B, T, n_kv_heads, head_dim)
    cos/sin: (T, head_dim)
    """
    # YOUR CODE HERE
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 6: SwiGLU MLP
# gate = gate_proj(x); up = up_proj(x)
# out = down_proj(silu(gate) * up)
# ══════════════════════════════════════════════════════════════════════════════

class MySwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # YOUR CODE HERE  (3 linear layers, no bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 7: GQA Attention (no KV cache)
# ══════════════════════════════════════════════════════════════════════════════

class MyGQAAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = hidden_size // num_heads
        self.groups       = num_heads // num_kv_heads
        # YOUR CODE HERE: q_proj, k_proj, v_proj, o_proj

    def forward(self, x, cos, sin):
        # YOUR CODE HERE
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 8: DecoderLayer
# pre-norm: rmsnorm → attn → residual → rmsnorm → mlp → residual
# ══════════════════════════════════════════════════════════════════════════════

class MyDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size, eps=1e-6):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, x, cos, sin):
        # YOUR CODE HERE
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 9: Full QwenModel
# token_ids → Embedding → [DecoderLayer x N] → RMSNorm → lm_head → logits
# ══════════════════════════════════════════════════════════════════════════════

class MyQwenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) → logits: (B, T, vocab_size)"""
        # YOUR CODE HERE
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Verification — do not modify below this line
# ══════════════════════════════════════════════════════════════════════════════

def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}  →  {e}")
        return False


def test_linear():
    torch.manual_seed(0)
    in_f, out_f = 4, 8
    x   = torch.randn(2, in_f)
    ref = nn.Linear(in_f, out_f)
    my  = MyLinear(in_f, out_f)
    my.weight.data = ref.weight.data.clone()
    my.bias.data   = ref.bias.data.clone()
    assert torch.allclose(ref(x), my(x), atol=1e-6)


def test_softmax():
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    assert torch.allclose(nn.Softmax(dim=-1)(x), MySoftmax(dim=-1)(x), atol=1e-6)


def test_embedding():
    torch.manual_seed(0)
    vocab, dim = 100, 16
    ids = torch.randint(0, vocab, (2, 5))
    ref = nn.Embedding(vocab, dim)
    my  = MyEmbedding(vocab, dim)
    my.weight.data = ref.weight.data.clone()
    assert torch.allclose(ref(ids), my(ids), atol=1e-6)


def test_rmsnorm():
    torch.manual_seed(0)
    hidden = 64
    x, eps = torch.randn(2, 10, hidden), 1e-6
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    ref = (x / rms) * torch.ones(hidden)
    my  = MyRMSNorm(hidden, eps=eps)
    nn.init.ones_(my.weight)
    assert torch.allclose(ref, my(x), atol=1e-6)


def test_rope():
    torch.manual_seed(0)
    B, T, n_heads, head_dim = 2, 6, 4, 32
    q = torch.randn(B, T, n_heads, head_dim)
    k = torch.randn(B, T, n_heads, head_dim)
    cos, sin = precompute_freqs(head_dim, T)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert q_rot.shape == (B, T, n_heads, head_dim)
    assert abs(q.norm(dim=-1).mean().item() - q_rot.norm(dim=-1).mean().item()) < 0.01


def test_swiglu():
    torch.manual_seed(0)
    B, T, hidden, inter = 2, 5, 64, 128
    mlp = MySwiGLUMLP(hidden, inter)
    out = mlp(torch.randn(B, T, hidden))
    assert out.shape == (B, T, hidden)


def test_gqa():
    torch.manual_seed(0)
    B, T, hidden, n_heads, n_kv = 2, 6, 64, 4, 2
    attn = MyGQAAttention(hidden, n_heads, n_kv)
    cos, sin = precompute_freqs(hidden // n_heads, T)
    out = attn(torch.randn(B, T, hidden), cos, sin)
    assert out.shape == (B, T, hidden)


def test_decoder():
    torch.manual_seed(0)
    B, T, hidden, n_heads, n_kv, inter = 2, 6, 64, 4, 2, 128
    layer = MyDecoderLayer(hidden, n_heads, n_kv, inter)
    cos, sin = precompute_freqs(hidden // n_heads, T)
    out = layer(torch.randn(B, T, hidden), cos, sin)
    assert out.shape == (B, T, hidden)


def test_model():
    torch.manual_seed(0)
    B, T = 2, 10
    cfg = dict(vocab_size=1000, hidden_size=64, num_layers=2,
               num_heads=4, num_kv_heads=2, intermediate_size=128, max_seq_len=64)
    model  = MyQwenModel(**cfg)
    logits = model(torch.randint(0, cfg["vocab_size"], (B, T)))
    assert logits.shape == (B, T, cfg["vocab_size"])
    print(f"         param count: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    print("\nRunning exercises...\n")
    tests = [
        ("1. Linear",       test_linear),
        ("2. Softmax",      test_softmax),
        ("3. Embedding",    test_embedding),
        ("4. RMSNorm",      test_rmsnorm),
        ("5. RoPE",         test_rope),
        ("6. SwiGLU MLP",   test_swiglu),
        ("7. GQA Attention",test_gqa),
        ("8. DecoderLayer", test_decoder),
        ("9. QwenModel",    test_model),
    ]
    results = [check(name, fn) for name, fn in tests]
    print(f"\n{sum(results)}/{len(results)} passed")
