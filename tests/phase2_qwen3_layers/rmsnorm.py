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


import torch
import torch.nn as nn
import torch.nn.functional as F

class MyRMSNorm(nn.Module): 

    def __init__(self, hidden_size, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()

        rms = torch.rsqrt(x.pow(2).mean(dim= -1, keepdim=True) + self.eps)
        return (x * rms * self.weight).to(dtype)




if __name__ == "__main__":
    hidden_size = 8
    eps = 1e-6

    my = MyRMSNorm(hidden_size, eps)
    x = torch.randn(2, 4, hidden_size)  # (batch, seq_len, hidden)

    # 手动算期望结果
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    expected = x / rms * my.weight

    print(torch.allclose(my(x), expected))  # 应该 True