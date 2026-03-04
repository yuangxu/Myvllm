# 全局 ForwardContext：在 model_runner 和 attention 层之间传递元数据
# 避免把 slot_mapping/block_tables 等张量层层传参
#
# ForwardContext 字段：
#   is_prefill:    bool
#   slot_mapping:  Tensor[total_tokens]         写入 KV cache 的物理槽位
#   cu_seqlens:    Tensor[num_seqs+1] or None   prefill 专用，序列边界累积和
#   block_tables:  Tensor[batch, max_blocks] or None  decode 专用
#   context_lens:  Tensor[batch] or None              decode 专用，每条序列的历史长度
#
# API：
#   set_context(**kwargs)   # model_runner 在 forward 前调用
#   get_context()           # attention 层调用
#   reset_context()         # model_runner 在 forward 后调用
