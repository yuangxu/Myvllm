#!/usr/bin/env python3
"""
自动从 arxiv 搜索并下载论文，以论文标题命名 PDF 文件。
"""

import os
import re
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote

PAPERS = [
    # --- Attention & Inference Serving (P1-P10) ---
    "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
    "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
    "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision",
    "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU",
    "Fast Inference from Transformers via Speculative Decoding",
    "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification",
    "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving",
    "Efficient Memory Management for Large Language Model Serving with PagedAttention",
    "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving",
    "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale",
    # --- Parallelism & MoE (P11-P16) ---
    "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning",
    "AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving",
    "PipeDream: Generalized Pipeline Parallelism for DNN Training",
    "Sequence Parallelism: Long Sequence Training from System Perspective",
    "A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize Mixture-of-Experts Training",
    "Tutel: Adaptive Mixture-of-Experts at Scale",
    # --- RLHF & RL (P17-P19) ---
    "HybridFlow: A Flexible and Efficient RLHF Framework",
    "StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation",
    "AREAL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning",
    # --- AI Hardware (P20-P25) ---
    "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings",
    "MTIA: First Generation Silicon Targeting Meta's Recommendation Systems",
    "Meta's Second Generation AI Chip: Model-Chip Co-Design and Productionization Experiences",
    "SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts",
    "Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning",
    "Ascend: a Scalable and Unified Architecture for Ubiquitous Deep Neural Network Computing",
    # --- Frameworks & Compilers (P26-P30) ---
    "PyTorch: An Imperative Style, High-Performance Deep Learning Library",
    "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation",
    "ML-Triton, A Multi-Level Compilation and Language Extension to Triton GPU Programming",
    "Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler",
    "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning",
]

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def sanitize_filename(title: str) -> str:
    """将论文标题转为合法文件名。"""
    name = re.sub(r'[<>:"/\\|?*]', "", title)
    name = re.sub(r"\s+", " ", name).strip()
    return f"{name}.pdf"


def search_arxiv(title: str):
    """在 arxiv 上按标题搜索论文，返回 (pdf_url, 实际标题)。"""
    # 第一次：精确标题搜索
    query = f'ti:"{title}"'
    pdf_url, actual_title = _query_arxiv(query)
    if pdf_url:
        return pdf_url, actual_title

    # 第二次：用冒号前的关键词搜索（如 "FlashAttention"）
    keyword = title.split(":")[0].strip() if ":" in title else title.split()[0]
    query = f"ti:{keyword}"
    return _query_arxiv(query)


def _query_arxiv(query: str):
    url = f"{ARXIV_API}?search_query={quote(query)}&max_results=5&sortBy=relevance"
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        print(f"  [ERROR] 请求失败: {e}")
        return None, None

    if resp.status_code != 200:
        print(f"  [ERROR] HTTP {resp.status_code}")
        return None, None

    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", NS)
    if not entries:
        return None, None

    entry = entries[0]
    actual_title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
    actual_title = re.sub(r"\s+", " ", actual_title)

    # 获取 PDF 链接
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf":
            return link.get("href") + ".pdf", actual_title

    # 备用：从 entry id 构造
    entry_id = entry.find("atom:id", NS).text
    pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"
    return pdf_url, actual_title


def download_pdf(url: str, filepath: str) -> bool:
    """下载 PDF 到指定路径。"""
    try:
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            # 检查是否真的是 PDF
            with open(filepath, "rb") as f:
                header = f.read(5)
            if header != b"%PDF-":
                print(f"  [WARN] 下载的文件可能不是有效 PDF")
            return True
    except requests.RequestException as e:
        print(f"  [ERROR] 下载失败: {e}")
    return False


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    fail_list = []

    for i, title in enumerate(PAPERS, 1):
        print(f"\n[{i}/{len(PAPERS)}] 搜索: {title}")

        pdf_url, actual_title = search_arxiv(title)

        if pdf_url is None:
            print(f"  [SKIP] 在 arxiv 上未找到该论文")
            fail_list.append(title)
            time.sleep(3)
            continue

        filename = sanitize_filename(actual_title or title)
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"  [EXISTS] 已存在: {filename}")
            success_count += 1
            continue

        print(f"  标题: {actual_title}")
        print(f"  URL:  {pdf_url}")
        print(f"  下载中...")

        if download_pdf(pdf_url, filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  [OK] 已保存: {filename} ({size_mb:.1f} MB)")
            success_count += 1
        else:
            print(f"  [FAIL] 下载失败")
            fail_list.append(title)

        # arxiv API 要求至少 3 秒间隔
        time.sleep(3)

    print(f"\n{'='*60}")
    print(f"完成! 成功: {success_count}/{len(PAPERS)}")
    print(f"保存目录: {output_dir}")
    if fail_list:
        print(f"\n以下论文下载失败:")
        for t in fail_list:
            print(f"  - {t}")


if __name__ == "__main__":
    main()
