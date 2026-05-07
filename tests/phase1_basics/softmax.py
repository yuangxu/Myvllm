"""
Exercise 2: MySoftmax
Goal: 实现数值稳定的 softmax
公式: softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))

写之前先在脑子里回答：
- 为什么要减 max(x)？（提示：overflow）
- keepdim=True 是干什么的，什么时候需要它？
- 输出 shape 和输入 shape 相同吗？
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MySoftmax(nn.Module):

    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim


    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x_max = torch.max(x, dim=self.dim, keepdim=True).values
    #     exp_x = torch.exp(x - x_max)
    #     return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = torch.max(x, dim= self.dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)



if __name__ == "__main__":
    model = MySoftmax()
    x = torch.randn(2, 8)
    print(torch.allclose(model(x), F.softmax(x, dim=-1)))  # True

    # 测试数值稳定性
    big = torch.tensor([1000., 2000., 3000.])
    print(model(big))  # 不应该有 nan


