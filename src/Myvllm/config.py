from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Qwen3-0.6B 模型结构参数"""
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072        # MLP 中间层维度
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8         # GQA：KV head 数，Q head 数的一半
    # Qwen3 配置里显式给出 head_dim（可与 hidden/num_heads 不一致，如 0.6B 为 128）
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0        # RoPE 基频
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True     # lm_head 与 embed_tokens 共享权重

    # 模型权重路径
    model_path: str = "Qwen/Qwen3-0.6B"


@dataclass
class EngineConfig:
    """推理引擎运行参数"""
    block_size: int = 16                 # KV cache 每个 block 存多少个 token
    max_num_sequences: int = 256         # 同时处理的最大序列数
    max_num_batched_tokens: int = 4096   # 单次 iteration 最多处理的 token 数
    max_model_length: int = 8192         # 序列最大长度（超出则停止生成）
    gpu_memory_utilization: float = 0.9  # GPU 显存使用比例
