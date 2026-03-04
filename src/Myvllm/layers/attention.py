# Attention 层（带 Paged KV Cache）
# Qwen3 特性：
#   - GQA：num_attention_heads=16，num_key_value_heads=8，ratio=2
#   - per-head QK Norm（q_norm/k_norm，对每个 head 做 RMSNorm）
#   - 无 QKV bias
#
# KV Cache 结构（由 model_runner 分配后注入）：
#   k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
#   v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
#
# Prefill 阶段：
#   1. 投影 Q/K/V
#   2. 对 K/V 做 RMSNorm，然后做 RoPE
#   3. 按 slot_mapping 写入 KV cache：k_cache.view(-1, ...)[slot_mapping] = k
#   4. 逐序列用 F.scaled_dot_product_attention（causal=True）
#   5. GQA：k/v 用 repeat_interleave 扩展到 num_heads
#
# Decode 阶段：
#   1. 投影单个新 token 的 Q/K/V
#   2. 按 slot_mapping 写新 K/V 到 cache
#   3. 按 block_tables gather 历史 K/V（截取 context_lens 个 token）
#   4. GQA 扩展后做 attention（causal=False，Q 只有 1 个 token）
#
# 所需 context（从 utils/context.py 读取）：
#   is_prefill, slot_mapping, cu_seqlens（prefill），block_tables/context_lens（decode）
