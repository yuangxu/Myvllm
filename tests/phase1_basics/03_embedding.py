"""
Exercise 3: MyEmbedding
Goal: 实现查表式 embedding

核心思路: embedding 本质就是用 token id 去索引一个权重矩阵
- weight shape: (num_embeddings, embedding_dim)
- forward: 输入 (B, T) 的 token ids，输出 (B, T, embedding_dim)

写之前先在脑子里回答：
- 这为什么不是矩阵乘法？indexing 和 matmul 的区别是什么？
- weight[token_ids] 在 token_ids 是二维时怎么工作的？
"""
