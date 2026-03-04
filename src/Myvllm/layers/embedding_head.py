# 词嵌入层 + LM Head
#
# VocabEmbedding：nn.Embedding(vocab_size, hidden_size)
#
# LMHead：nn.Linear(hidden_size, vocab_size, bias=False)
#   - Qwen3 tie_word_embeddings=True，即 lm_head.weight = embed_tokens.weight
#   - 共享权重可节省约 150M 参数（151936 × 1024）
#
# compute_logits()：
#   - Prefill：只取每条序列最后一个 token 的 hidden state 做投影（不需要全部）
#   - Decode：每条序列本来就只有 1 个 token，直接投影
