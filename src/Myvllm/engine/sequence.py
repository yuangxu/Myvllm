# Sequence：一条推理请求的完整状态
#
# SequenceStatus：WAITING / RUNNING / FINISHED
#
# Sequence 字段：
#   seq_id:              int（自增唯一 ID）
#   token_ids:           List[int]（prompt tokens + 已生成的 tokens 全放这里）
#   num_prompt_tokens:   int（prompt 长度，区分 prompt vs 生成部分）
#   block_table:         List[int]（逻辑 block i → 物理 block id）
#   num_cached_tokens:   int（prefix cache 命中数，暂时填 0）
#   status:              SequenceStatus
#   sampling_params:     SamplingParams
#
# 关键 property（用 @property 实现）：
#   num_tokens          = len(token_ids)
#   num_blocks          = ceil(num_tokens / block_size)
#   last_block_num_tokens  = num_tokens % block_size（0 则表示最后一个 block 满了）
#   last_token_id       = token_ids[-1]
#   completion_token_ids = token_ids[num_prompt_tokens:]
#   is_finished         = status == FINISHED
#
# block(i) 方法：返回第 i 个逻辑 block 对应的 token_ids 列表
