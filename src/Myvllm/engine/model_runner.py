# ModelRunner：负责 GPU 上的实际计算
#
# __init__(config):
#   1. 加载模型到 GPU（model.cuda()）
#   2. 加载 HuggingFace 权重（model.load_weights()）
#   3. warmup_model()：用最大 batch 跑一次空 forward，测 peak 显存
#   4. allocate_kv_cache()：根据剩余显存分配 KV cache block pool
#
# allocate_kv_cache()：
#   free_mem, total = torch.cuda.mem_get_info()
#   available = free_mem * gpu_memory_utilization - (peak_mem - current_mem)
#   block_bytes = block_size * 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
#   num_blocks = available // block_bytes
#   分配 Tensor：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
#   注入到每个 attention 层的 k_cache / v_cache
#
# prepare_prefill(seqs) -> (input_ids, positions):
#   input_ids：拼接所有序列 token_ids → [total_tokens]
#   positions：每个 token 的位置编号（每条序列从 0 开始）→ [total_tokens]
#   slot_mapping：每个 token 写入 KV cache 的物理槽（block_id * block_size + offset）
#   cu_seqlens：序列边界累积和 [0, len0, len0+len1, ...]
#   → set_context(is_prefill=True, slot_mapping=..., cu_seqlens=...)
#
# prepare_decode(seqs) -> (input_ids, positions):
#   input_ids：每条序列最后一个 token → [batch_size]
#   positions：context_lens - 1（新 token 的位置）
#   slot_mapping：新 token 写入的物理槽
#   block_tables：[batch_size, max_num_blocks]，padding 用 -1 或 0
#   context_lens：每条序列的历史长度（含新 token）
#   → set_context(is_prefill=False, slot_mapping=..., block_tables=..., context_lens=...)
#
# run(prefill_seqs, decode_seqs) -> List[int]:
#   if prefill_seqs:
#       input_ids, positions = prepare_prefill(prefill_seqs)
#       hidden = model(input_ids, positions)
#       logits = model.compute_logits(hidden, cu_seqlens)  # 只取每条序列最后一个 token
#       token_ids = sampler(logits, temperatures)
#       reset_context()
#   if decode_seqs:
#       input_ids, positions = prepare_decode(decode_seqs)
#       hidden = model(input_ids, positions)
#       logits = model.compute_logits(hidden)
#       token_ids = sampler(logits, temperatures)
#       reset_context()
#   return token_ids
