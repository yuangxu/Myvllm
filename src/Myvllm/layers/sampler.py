# 采样层：从 logits 采样出下一个 token id
#
# 输入：
#   logits:      [batch_size, vocab_size]
#   temperatures: [batch_size]
#
# 流程：
#   1. logits /= temperature（temperature 缩放）
#   2. probs = softmax(logits, dim=-1)
#   3. token_ids = torch.multinomial(probs, num_samples=1)
#
# 注意：temperature=0 时退化为 greedy（取 argmax）
