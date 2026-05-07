"""
Exercise 5: RoPE（旋转位置编码）
Goal: 实现 Qwen3 用的旋转位置编码

数学原理：
  对于位置 m，维度对 i，旋转角度为：θ_i = m / 10000^(2i/d)
  对每对 (x0, x1) 做二维旋转：
    x0' = x0*cos(θ) - x1*sin(θ)
    x1' = x0*sin(θ) + x1*cos(θ)

简化写法（rotate_half trick）：
  RoPE(x) = x * cos + rotate_half(x) * sin
  rotate_half([x1, x2]) = [-x2, x1]  （后半取负放前面）

需要实现的函数：
  1. precompute_freqs(head_dim, max_seq_len, base) → cos, sin  shape: (T, head_dim)
  2. rotate_half(x) → (..., head_dim)
  3. apply_rope(q, k, cos, sin) → q_rot, k_rot

写之前先在脑子里回答：
- 位置编码为什么加在 Q 和 K 上，不加在 V 上？
- 为什么是 head_dim 而不是 hidden_dim？
- cos/sin 怎么 broadcast 到 (B, n_heads, T, head_dim) 的 shape？
"""

import torch
import torch.nn as nn

class MyRoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        # 第一步：构造频率
        # 每一对元素一个频率，所以是 head_dim // 2 个
        i = torch.arange(0, head_dim, 2).float()  # [0, 2, 4, ...]
        freqs = 1.0 / (base ** (i / head_dim))     # (head_dim // 2,)

        # 第二步：构造角度矩阵
        positions = torch.arange(max_seq_len).float()          # [0, 1, 2, ..., max_seq_len-1]
        angles = positions[:, None] * freqs[None, :]           # (max_seq_len, head_dim // 2)

        # 算 cos 和 sin，注册为 buffer（不是参数，不参与训练）
        self.register_buffer('cos', torch.cos(angles))  # (max_seq_len, head_dim // 2)
        self.register_buffer('sin', torch.sin(angles))  # (max_seq_len, head_dim // 2)

    def forward(self, q, k, position_ids):
        # q, k shape: (batch, num_heads, seq_len, head_dim)
        seq_len = q.shape[2]

        # 取出当前位置的 cos/sin
        cos = self.cos[position_ids]  # (seq_len, head_dim // 2)
        sin = self.sin[position_ids]  # (seq_len, head_dim // 2)

        # 广播到 (1, 1, seq_len, head_dim // 2) 方便和 q、k 相乘
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # 第三步：拆两半，应用旋转
        q1, q2 = q.chunk(2, dim=-1)   # 各 (batch, heads, seq_len, head_dim // 2)
        k1, k2 = k.chunk(2, dim=-1)

        q_rotated = torch.cat([q1 * cos - q2 * sin,
                                q1 * sin + q2 * cos], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin,
                                k1 * sin + k2 * cos], dim=-1)

        return q_rotated, k_rotated


if __name__ == "__main__":
    head_dim = 64
    rope = MyRoPE(head_dim)

    batch, num_heads, seq_len = 2, 8, 10
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    position_ids = torch.arange(seq_len)

    q_rot, k_rot = rope(q, k, position_ids)
    print("输入 shape:", q.shape)
    print("输出 shape:", q_rot.shape)        # 应该一样
    print("旋转改变了值:", not torch.equal(q, q_rot))  # 应该 True