"""
Exercise 1: MyLinear
Goal: 从零实现线性层
公式: y = x @ W.T + b

写之前先在脑子里回答：
- weight 矩阵的 shape 是什么？
- 为什么用 nn.Parameter 而不是普通 torch.tensor？
- model(x) 是怎么触发 forward(x) 的？
"""


import torch
print(torch.__version__)
x = torch.randn(3, 4)
print(x @ x.T)



import torch
import torch.nn as nn
import time


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.torch:
        return x @ self.weight.T + self.bias



if __name__ == "__main__":
    my = MyLinear(4, 3)
    official = nn.Linear(4, 3)

    with torch.no_grad():
        official.weight.copy_(my.weight)
        official.bias.copy_(my.bias)

    x = torch.randn(2, 4)
    print(torch.allclose(my(x), official(x)))  # 应该输出 True

