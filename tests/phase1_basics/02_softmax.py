"""
Exercise 2: MySoftmax
Goal: 实现数值稳定的 softmax
公式: softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))

写之前先在脑子里回答：
- 为什么要减 max(x)？（提示：overflow）
- keepdim=True 是干什么的，什么时候需要它？
- 输出 shape 和输入 shape 相同吗？
"""
