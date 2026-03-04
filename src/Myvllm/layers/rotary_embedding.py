# RoPE 旋转位置编码
# Qwen3 参数：rope_theta=1000000.0, head_dim=64
#
# θ_i = rope_theta^(-2i/head_dim),  i = 0..head_dim//2-1
#
# 预计算 cos/sin 表：shape [max_position, head_dim]
# 旋转操作（_rotate_half）：
#   x = [x1, x2]（前半/后半）
#   rotate(x) = [-x2, x1]
#   out = x * cos + rotate(x) * sin
#
# 输入 q/k shape：[total_tokens, num_heads, head_dim]
# positions shape：[total_tokens]
