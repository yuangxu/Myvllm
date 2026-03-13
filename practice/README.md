# Practice Exercises

Work through these in order. For each file:
1. Read the docstring
2. Close all references
3. Fill in `# YOUR CODE HERE`
4. Run `python <file>.py` — must print `PASS`

## Phase 1 — PyTorch basics (muscle memory)
| File | Concept |
|------|---------|
| `phase1_basics/01_linear.py` | `nn.Parameter`, matmul, bias |
| `phase1_basics/02_softmax.py` | numerical stability, keepdim |
| `phase1_basics/03_embedding.py` | lookup table, tensor indexing |

## Phase 2 — Qwen3 layers
| File | Concept |
|------|---------|
| `phase2_qwen3_layers/01_rmsnorm.py` | RMS norm, no mean subtraction |
| `phase2_qwen3_layers/02_rope.py` | rotary embeddings, rotate_half |
| `phase2_qwen3_layers/03_swiglu_mlp.py` | gated activation, SiLU |
| `phase2_qwen3_layers/04_gqa_attention.py` | GQA, repeat_interleave, scaled dot-product |

## Phase 3 — Full model
| File | Concept |
|------|---------|
| `phase3_model/01_decoder_layer.py` | residual connections, pre-norm pattern |
| `phase3_model/02_qwen_model.py` | full forward pass, stacking layers |

## Rules
- No looking at `src/Myvllm/` while writing
- If stuck for >15 min on one thing, look at just that piece, then close and rewrite from scratch
- After finishing phase 3, delete this folder and rewrite everything from blank files
