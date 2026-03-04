# 端到端推理演示
#
# 流程：
#   1. 初始化 LLMEngine（加载模型、分配 KV cache）
#   2. 准备几条测试 prompt，用 chat template 包装
#   3. 调用 engine.generate(prompts, sampling_params)
#   4. 打印 prompt → completion
#
# 示例配置（Qwen3-0.6B）：
#   model_config = ModelConfig(model_path="Qwen/Qwen3-0.6B")
#   engine_config = EngineConfig(block_size=16, max_num_sequences=8)
#   sampling_params = SamplingParams(temperature=0.6, max_tokens=200)
#
# 测试 prompt：
#   "你好，介绍一下你自己"
#   "100以内的素数有哪些？"
#   "用Python写一个快速排序"
