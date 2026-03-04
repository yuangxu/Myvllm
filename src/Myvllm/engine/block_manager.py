# Block 和 BlockManager：物理 KV Cache Block 的生命周期管理
#
# Block：
#   block_id:   int
#   ref_count:  int（有多少序列在使用这个 block）
#
# BlockManager：
#   blocks:         List[Block]（所有物理 block）
#   free_block_ids: deque（空闲 block id）
#   used_block_ids: set
#
# 关键方法：
#
#   can_allocate(seq) -> bool
#     len(free_block_ids) >= seq.num_blocks
#
#   allocate(seq) -> None
#     从 free 队列取 block，填入 seq.block_table，ref_count = 1
#
#   can_append(seq) -> bool
#     如果下一个 token 需要新 block（last_block_num_tokens == 0）：检查 free 数量
#     否则当前 block 还有空间，直接返回 True
#
#   append(seq) -> None
#     decode 时，如果需要新 block 就分配一个追加到 seq.block_table
#
#   deallocate(seq) -> None
#     遍历 seq.block_table，每个 block ref_count -= 1
#     降到 0 的 block 移回 free_block_ids，清空 seq.block_table
