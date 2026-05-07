import torch
import torch.nn as nn
import math
from rope import MyRoPE


class MyGQA(nn.Module):

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, rope: MyRoPE, position_ids: torch.Tensor):
        B, T, _ = x.shape

        # 1. 投影
        q = self.q_proj(x)  # (B, T, num_heads * head_dim)
        k = self.k_proj(x)  # (B, T, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, num_kv_heads * head_dim)

        # 2. reshape 成多头
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (B, num_kv_heads, T, head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (B, num_kv_heads, T, head_dim)

        # 3. RoPE
        q, k = rope(q, k, position_ids)

        # 4. 扩展 KV 头数
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (B, num_heads, T, head_dim)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 5. scaled dot-product attention
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v  # (B, num_heads, T, head_dim)

        # 6. 合并头 + 输出投影
        output = output.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        output = self.o_proj(output)

        return output
    

if __name__ == "__main__":
    hidden_size = 1024
    num_heads = 16
    num_kv_heads = 8
    head_dim = hidden_size // num_heads  # 64

    rope = MyRoPE(head_dim)
    gqa = MyGQA(hidden_size, num_heads, num_kv_heads)

    B, T = 2, 10
    x = torch.randn(B, T, hidden_size)
    position_ids = torch.arange(T)

    output = gqa(x, rope, position_ids)

    print("输入 shape:", x.shape)           # (2, 10, 1024)
    print("输出 shape:", output.shape)       # (2, 10, 1024)
    print("输入输出 shape 一致:", x.shape == output.shape)  # True
