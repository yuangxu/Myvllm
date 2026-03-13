"""
Exercise 4: RMSNorm
Goal: 不看任何参考，从记忆里重写 RMSNorm

公式:
  RMSNorm(x) = x / RMS(x) * w
  RMS(x) = sqrt(mean(x²) + ε)

写之前先在脑子里回答：
- RMSNorm 和 LayerNorm 的区别是什么？（少了哪一步？）
- 为什么 mean 要加 keepdim=True？
- weight 的 shape 是什么？为什么是向量而不是矩阵？
"""
