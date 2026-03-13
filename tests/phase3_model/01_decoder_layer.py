"""
Exercise 8: DecoderLayer
Goal: 把前面写好的组件串成一个完整的 Transformer decoder 层

结构（pre-norm 风格）：
  residual = x
  x = rmsnorm(x)
  x = attention(x, cos, sin)
  x = residual + x          ← 第一个残差连接

  residual = x
  x = rmsnorm(x)
  x = mlp(x)
  x = residual + x          ← 第二个残差连接

写之前先在脑子里回答：
- pre-norm 和 post-norm 的区别是什么？Qwen3 用哪个？
- 为什么需要残差连接？去掉会怎样？
- 这层有几个 RMSNorm？
"""
