from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Llama-3.2-1B 的标准参数
    hidden_size: int = 2048
    intermediate_size: int = 8192  # MLP 的中间层大小
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8   # GQA! 只有 8 个 KV 头
    head_dim: int = 64             # hidden_size / num_attention_heads
    vocab_size: int = 128256       # Llama 3 的超大词表
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0   # RoPE 的基频
    max_position_embeddings: int = 131072 # 上下文长度

    # 你的模型路径 (记得改成你下载的实际路径)
    model_path: str = "/root/autodl-tmp/models/Llama-3.2-1B-Instruct"