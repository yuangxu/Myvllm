# SiLU 激活函数
# Qwen3 MLP 使用 SwiGLU 变体：output = silu(gate) * x
# silu(x) = x * sigmoid(x)

import torch
import torch.nn as nn
import torch.nn.Function as F
import time






if __name__ == "__main__":
    pass