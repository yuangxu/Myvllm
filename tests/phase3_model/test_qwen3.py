"""Qwen3ForCausalLM 端到端测试.

用法:
    # 1) shape 测试（不需要下载模型，秒级完成）
    python tests/phase3_model/test_qwen3.py --test shape

    # 2) 加载权重 + 和 HuggingFace 对比（需要先下载 Qwen3-0.6B）
    python tests/phase3_model/test_qwen3.py --test compare

    # 3) 实际生成文本
    python tests/phase3_model/test_qwen3.py --test generate
"""
import argparse
import sys
import time

import torch


def test_shape():
    """测试 1：forward shape 验证（随机权重，纯 CPU，几秒完成）."""
    from Myvllm.config import ModelConfig
    from Myvllm.model import Qwen3ForCausalLM

    print("=" * 60)
    print("TEST 1: Shape 验证（随机权重）")
    print("=" * 60)

    cfg = ModelConfig()
    print(f"  构建模型: {cfg.num_hidden_layers} layers, hidden={cfg.hidden_size}")
    model = Qwen3ForCausalLM(cfg)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    unique = len(set(id(p) for p in model.parameters()))
    print(f"  参数量: {total:,} ({total / 1e6:.1f}M), 独立 tensor 数: {unique}")

    T = 12
    input_ids = torch.randint(0, cfg.vocab_size, (T,))
    positions = torch.arange(T)

    with torch.no_grad():
        hidden = model(input_ids, positions)
        logits = model.compute_logits(hidden)

    assert hidden.shape == (T, cfg.hidden_size), f"hidden shape 错误: {hidden.shape}"
    assert logits.shape == (T, cfg.vocab_size), f"logits shape 错误: {logits.shape}"

    cu_seqlens = torch.tensor([0, 5, T])
    logits_prefill = model.compute_logits(hidden, cu_seqlens=cu_seqlens)
    assert logits_prefill.shape == (2, cfg.vocab_size), f"prefill logits shape 错误: {logits_prefill.shape}"

    print(f"  hidden  shape: {hidden.shape}  ✓")
    print(f"  logits  shape: {logits.shape}  ✓")
    print(f"  prefill shape: {logits_prefill.shape}  ✓")
    print("  PASSED!\n")


def test_compare(model_path: str, device: str):
    """测试 2：加载真实权重，和 HuggingFace transformers 逐 token 对比 logits."""
    from Myvllm.config import ModelConfig
    from Myvllm.model import Qwen3ForCausalLM

    print("=" * 60)
    print("TEST 2: 与 HuggingFace 对比")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = "The capital of France is"
    input_ids_list = tokenizer.encode(prompt)
    print(f"  Prompt: '{prompt}'")
    print(f"  Token ids: {input_ids_list}  (len={len(input_ids_list)})")

    # ---- 我们的模型 ----
    print(f"\n  加载 MyModel 权重 ({device})...")
    cfg = ModelConfig(model_path=model_path)
    my_model = Qwen3ForCausalLM(cfg)
    my_model.load_weights(model_path)
    my_model.to(device).eval()

    ids_tensor = torch.tensor(input_ids_list, device=device)
    positions = torch.arange(len(input_ids_list), device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        my_hidden = my_model(ids_tensor, positions)
        my_logits = my_model.compute_logits(my_hidden)
    t1 = time.perf_counter()
    print(f"  MyModel forward: {(t1-t0)*1000:.1f} ms")

    # ---- HuggingFace 参考 ----
    print(f"  加载 HuggingFace 模型...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).to(device).eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        hf_out = hf_model(
            torch.tensor([input_ids_list], device=device)
        )
    t1 = time.perf_counter()
    hf_logits = hf_out.logits[0]  # [T, vocab]
    print(f"  HuggingFace forward: {(t1-t0)*1000:.1f} ms")

    # ---- 对比 ----
    max_diff = (my_logits - hf_logits).abs().max().item()
    mean_diff = (my_logits - hf_logits).abs().mean().item()

    my_top5 = my_logits[-1].topk(5)
    hf_top5 = hf_logits[-1].topk(5)

    print(f"\n  Max  absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  MyModel  top5 ids:  {my_top5.indices.tolist()}")
    print(f"  HF       top5 ids:  {hf_top5.indices.tolist()}")
    print(f"  Top1 一致: {my_top5.indices[0].item() == hf_top5.indices[0].item()}")

    if max_diff < 1e-3:
        print("  PASSED! (max diff < 1e-3)\n")
    elif max_diff < 1e-1:
        print("  WARN: diff 较大但 top token 可能一致，检查精度\n")
    else:
        print("  FAILED: diff 过大，存在 bug\n")
        sys.exit(1)


def test_generate(model_path: str, device: str):
    """测试 3：用我们的模型做简单贪心生成."""
    from Myvllm.config import ModelConfig
    from Myvllm.model import Qwen3ForCausalLM
    from Myvllm.layers.sampler import Sampler

    print("=" * 60)
    print("TEST 3: Greedy 生成")
    print("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cfg = ModelConfig(model_path=model_path)
    model = Qwen3ForCausalLM(cfg)
    model.load_weights(model_path)
    model.to(device).eval()

    sampler = Sampler()
    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    max_new_tokens = 50

    print(f"  Prompt: '{prompt}'")
    print(f"  Generating {max_new_tokens} tokens (greedy)...\n")

    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(max_new_tokens):
            ids_tensor = torch.tensor(generated, device=device)
            positions = torch.arange(len(generated), device=device)
            hidden = model(ids_tensor, positions)
            logits = model.compute_logits(hidden[-1:])  # 只要最后一个 token
            next_id = sampler(logits, torch.tensor([0.0], device=device)).item()
            generated.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
    t1 = time.perf_counter()

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    num_new = len(generated) - len(input_ids)
    print(f"  Output: {output_text}")
    print(f"  Generated {num_new} tokens in {(t1-t0):.2f}s "
          f"({num_new/(t1-t0):.1f} tok/s)")
    print("  DONE!\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen3ForCausalLM 测试")
    parser.add_argument(
        "--test",
        choices=["shape", "compare", "generate", "all"],
        default="shape",
        help="选择测试级别",
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace 模型路径或本地目录",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备 (cuda / cpu)",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Model:  {args.model_path}\n")

    if args.test in ("shape", "all"):
        test_shape()

    if args.test in ("compare", "all"):
        test_compare(args.model_path, args.device)

    if args.test in ("generate", "all"):
        test_generate(args.model_path, args.device)


if __name__ == "__main__":
    main()
