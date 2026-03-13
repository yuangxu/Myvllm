"""
Exercise 7: GQA Attention（无 KV cache）
Goal: 实现 Grouped Query Attention

GQA = Q 有完整的 num_heads 个头，但 KV 只有 num_kv_heads 个头。
每个 KV 头服务多个 Q 头。
Qwen3 示例: num_heads=16, num_kv_heads=8 → 每个 KV 头服务 2 个 Q 头

实现步骤：
  1. x → q_proj, k_proj, v_proj
  2. reshape → (B, T, n_heads, head_dim)，transpose → (B, n_heads, T, head_dim)
  3. 对 q, k 施加 RoPE
  4. 把 k, v repeat 到 num_heads 个头
  5. scaled dot-product attention:
       scores = (q @ k.T) / sqrt(head_dim)
       weights = softmax(scores, dim=-1)
       out = weights @ v
  6. reshape → (B, T, hidden_size)，经过 o_proj

写之前先在脑子里回答：
- repeat_interleave 和 repeat 有什么区别？
- 为什么除以 sqrt(head_dim)？
- causal mask 应该加在哪一步？（本练习先跳过）
"""
