# Qwen3ForCausalLM 完整模型
#
# 结构：
#   embed_tokens: VocabEmbedding(vocab_size, hidden_size)
#   layers: 28 × Qwen3DecoderLayer
#     ├─ input_layernorm: RMSNorm(hidden_size)
#     ├─ self_attn: Attention(...)
#     ├─ post_attention_layernorm: RMSNorm(hidden_size)
#     └─ mlp: Qwen3MLP
#          ├─ gate_proj: Linear(hidden_size, intermediate_size)
#          ├─ up_proj:   Linear(hidden_size, intermediate_size)
#          └─ down_proj: Linear(intermediate_size, hidden_size)
#              output = down_proj(silu(gate_proj(x)) * up_proj(x))
#   norm: RMSNorm(hidden_size)
#   lm_head: LMHead(hidden_size, vocab_size)  # 与 embed_tokens 共享权重
#
# forward(input_ids, positions) -> hidden_states
# compute_logits(hidden_states, cu_seqlens_for_prefill) -> logits [batch, vocab]
#
# load_weights(model_path):
#   HuggingFace 权重 key 映射规则：
#     "model.embed_tokens.weight"              -> embed_tokens.weight
#     "model.layers.{i}.self_attn.q_proj.weight" -> layers.{i}.self_attn.q_proj.weight
#     "model.layers.{i}.mlp.gate_proj.weight"    -> layers.{i}.mlp.gate_proj.weight
#     "model.norm.weight"                      -> norm.weight
#     "lm_head.weight"                         -> lm_head.weight（tied，可跳过）
