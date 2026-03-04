# Scheduler：每个 iteration 决定处理哪些序列
#
# 队列：
#   waiting: deque[Sequence]   新请求等待 prefill
#   running: deque[Sequence]   正在 decode 的序列
#
# add_sequence(seq)：把新序列加入 waiting 队列
#
# schedule() -> (prefill_seqs, decode_seqs)：
#
#   Step 1：从 waiting 搬序列到 running（prefill）
#     条件：block_manager.can_allocate(seq)
#           且当前 prefill batch 总 token 数不超过 max_num_batched_tokens
#     满足条件：block_manager.allocate(seq)，seq.status = RUNNING，移入 running
#
#   Step 2：对 running 中的序列做 decode
#     对每条序列：block_manager.can_append(seq) ?
#       是：block_manager.append(seq)，加入 decode_seqs
#       否（显存不足）：preempt
#         block_manager.deallocate(seq)
#         seq.status = WAITING，seq.block_table = []
#         放回 waiting 队列头部
#
#   返回 (prefill_seqs, decode_seqs)
#
# postprocess(decode_seqs, new_token_ids)：
#   把新 token append 到各序列 token_ids
#   检查停止条件：EOS token / max_tokens / max_model_length
#   finished 的序列 status = FINISHED，从 running 移除，deallocate blocks
#
# is_finished() -> bool：waiting 和 running 都为空
