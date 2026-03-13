"""
Exercise 9: 完整 QwenModel
Goal: 把所有组件串成完整的模型前向传播

结构：
  token_ids
    → Embedding
    → [DecoderLayer] × N
    → RMSNorm
    → lm_head (Linear, vocab_size 输出)
    → logits

写之前先在脑子里回答：
- lm_head 的输入输出 shape 是什么？
- cos/sin 预计算在哪里做？在 __init__ 里还是 forward 里？
- 为什么有些模型 embedding 和 lm_head 的 weight 是共享的？
"""
