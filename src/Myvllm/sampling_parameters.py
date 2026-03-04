# SamplingParams：控制生成行为的参数
#
# 字段：
#   temperature:      float = 0.6    温度，控制随机性（0 = greedy）
#   max_tokens:       int = 256      最多生成多少个新 token
#   ignore_eos:       bool = False   是否忽略 EOS token（强制生成到 max_tokens）
#   max_model_length: int = None     序列总长度上限（prompt + 生成），None 表示不限
