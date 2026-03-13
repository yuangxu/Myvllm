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
