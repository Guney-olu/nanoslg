# NanoSLG
**Minimal Multi-Mode Parallel LLM Inference Server**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Experimental](https://img.shields.io/badge/Status-Research%20Preview-orange)]()

> A lightweight, educational LLM inference server with support for Pipeline Parallelism, Tensor Parallelism, Hybrid (TP+PP) modes, and a dual-backend KV cache that auto-selects FlashInfer (L4/A100+) or contiguous SDPA (T4/fallback).

---

## ✨ What's New (v0.5)

- **Dual KV Cache Backend** — auto-detects GPU and selects the fastest path:
  - **FlashInfer paged attention** on SM80+ (L4, A100, H100) — fused kernels, zero-copy paged KV
  - **Contiguous SDPA** on SM75 (T4) or as fallback — in-place writes, no gather overhead
- **Radix prefix caching** — reuses KV cache across requests with shared prefixes (FlashInfer backend)
- **No more gather_kv** — eliminated the per-layer full-copy bottleneck from v0.4
- **Native GQA** — uses `enable_gqa=True` on PyTorch 2.5+, no more `repeat_interleave`
- **`--backend` flag** — force `contiguous` or `flashinfer` for testing
- **3-5× throughput improvement** over v0.4

---

## Features

| Feature | Description |
|---------|-------------|
| **Pipeline Parallelism** | Split model layers across GPUs sequentially |
| **Tensor Parallelism** | Shard weights across GPUs for parallel computation |
| **Hybrid Mode** | Combine TP + PP for maximum scalability |
| **Dual KV Cache** | FlashInfer paged (L4+) or contiguous SDPA (T4) — auto-selected |
| **Radix Prefix Caching** | Reuse KV cache for shared prompt prefixes |
| **Batch Scheduling** | Dynamic batch formation for higher throughput |
| **Streaming Responses** | Server-Sent Events (SSE) token streaming |
| **Built-in Benchmarking** | TTFT, tokens/sec, memory tracking |
| **OpenAI-compatible API** | Drop-in replacement for `/v1/chat/completions` |
| **Modular Design** | Easy to extend with new models |

---

## Performance Results

Tested on **2× NVIDIA L4 GPUs (24GB each)** with **Llama-3.1-8B-Instruct FP16**:

### v0.5 (Current) vs v0.4

| Metric | v0.4 (Paged+Gather) | v0.5 Contiguous | v0.5 FlashInfer | Improvement |
|--------|---------------------|-----------------|-----------------|-------------|
| **Single tok/s** | 13.8 | 21.8 | 21.5 | **+58%** |
| **Single TTFT** | 106ms | 57ms | 52ms | **-51%** |
| **Batch×4 tok/s** | 12.1 | 68.4 | **76.0** | **+528%** |
| **KV Pressure tok/s** | 49.7 | 82.5 | 75.6 | **+66%** |
| **Sustained tok/s** | 24.6 | 37.3 | 37.4 | **+52%** |
| **Burst×16 tok/s** | 31.8 | 45.6 | 48.6 | **+53%** |
| **Burst TTFT p50** | 7641ms | 5364ms | — | **-30%** |
| **Benchmark duration** | 101s | 63s | — | **-37%** |

### By Mode (v0.5)

| Mode | Batch Size | Throughput | TTFT | Notes |
|------|------------|------------|------|-------|
| **Tensor Parallel** | 1 | ~22 tok/s | ~52ms | Lowest latency |
| **Tensor Parallel** | 4 | **~76 tok/s** | ~64ms | Best throughput |
| **Pipeline Parallel** | 1 | ~25 tok/s | ~85ms | Lower memory per GPU |
| **Pipeline Parallel** | 4 | ~62 tok/s | ~95ms | Good scaling |
| **Hybrid (2×2)** | 4 | ~100+ tok/s | ~70ms | 4 GPU setup |

---

## Architecture

### Dual KV Cache Backend

```mermaid
flowchart TD
flowchart TD
    A[GPU Detection<br/>get_sm_version()] --> B{SM ≥ 80?}

    B -->|Yes| C[FlashInfer Paged KV Cache<br/>(L4 / A100 / H100)]
    B -->|No| D[Contiguous SDPA KV Cache<br/>(T4 / Fallback)]

    C --> C1[Paged KV Pool]
    C --> C2[Radix Prefix]
    C --> C3[Fused Decode]
    C --> C4[Zero-copy Read]
    C --> C5[CoW Sharing]

    D --> D1[Pre-allocated Cache<br/>[B,S,H,D]]
    D --> D2[In-place Write]
    D --> D3[Slice-based Read]
    D --> D4[Native SDPA]

    C --> E[CacheContext ABC]
    D --> E[CacheContext ABC]

    E --> E1[attend()]
    E --> E2[get_position()]
    E --> E3[get_start_pos()]

    E --> F[Same Interface to Model]
```

### System Architecture

```
                          ┌──────────────────┐
                          │   HTTP Client     │
                          │  (curl/Python)    │
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  FastAPI Server   │
                          │   (async I/O)     │
                          └────────┬─────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │       mp.Queue            │
                     │  (request/response)       │
                     └─────────────┬─────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
 ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
 │     GPU 0       │    │     GPU 1       │    │     GPU N       │
 │ ┌───────────┐   │    │ ┌───────────┐   │    │ ┌───────────┐   │
 │ │ TP Layers │   │───▶│ │ TP Layers │   │───▶│ │ TP Layers │   │
 │ │ (sharded) │   │NCCL│ │ (sharded) │   │NCCL│ │ + LM Head │   │
 │ └───────────┘   │    │ └───────────┘   │    │ └───────────┘   │
 │ ┌───────────┐   │    │ ┌───────────┐   │    │ ┌───────────┐   │
 │ │ KV Cache  │   │    │ │ KV Cache  │   │    │ │ KV Cache  │   │
 │ │(auto-sel) │   │    │ │(auto-sel) │   │    │ │(auto-sel) │   │
 │ └───────────┘   │    │ └───────────┘   │    │ └───────────┘   │
 └─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Installation

### Prerequisites
- Python **3.10+**
- PyTorch **2.0+** with CUDA
- **2+ NVIDIA GPUs** (for parallelism)
- NCCL for GPU communication
- **GCC ≤ 12** (required for CUDA/FlashInfer JIT compilation)

### Setup

```bash
# Clone the repository
git clone https://github.com/Guney-olu/nanoslg
cd nanoslg
# Install dependencies
pip install torch>=2.0 transformers safetensors fastapi uvicorn requests
# Verify GPU setup
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Download a model (example: Llama-3.1-8B)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ./models/Llama-3.1-8B-Instruct
```

---

## Usage

### Basic Commands

```bash
# Tensor Parallel - 2 GPUs
python -m nanoslg --model /path/to/model --mode tensor --tp-size 2
# Force a specific KV cache backend
python -m nanoslg --model /path/to/model --mode tensor --backend contiguous
python -m nanoslg --model /path/to/model --mode tensor --backend flashinfer --max-pages 2000

# Full options
python -m nanoslg \
    --model /path/to/Llama-3.1-8B-Instruct \
    --mode tensor \
    --tp-size 2 \
    --batch-size 4 \
    --dtype float16 \
    --backend auto \
    --port 8000
```

### Client Examples

```bash
# Simple request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What? see below!"}],
    "max_tokens": 100,
    "stream": true
  }'
# Python client
python inference.py --max-tokens 200 --benchmark "Is 500 days of summer a horror movie?"
# Concurrent benchmark
python inference.py --concurrent 4 --max-tokens 100 "Why do you say that name!!"
```

## Roadmap

### Completed ✅
- [x] Pipeline Parallelism (PP)
- [x] Tensor Parallelism (TP)
- [x] Hybrid TP + PP mode
- [x] Batch scheduling
- [x] Streaming responses (SSE)
- [x] OpenAI-compatible API
- [x] Built-in benchmarking
- [x] Llama 3.x support
- [x] Dual KV cache backend (FlashInfer + contiguous SDPA)
- [x] Automatic GPU detection and backend selection
- [x] Radix prefix caching (FlashInfer backend)
- [x] Native GQA support (no expand overhead)
- [x] Paged attention with copy-on-write

### In Progress
- [ ] Continuous batching (iteration-level scheduling)
- [ ] Better warmup strategies

### Planned
- [ ] Quantization (INT8 / INT4 / GPTQ / AWQ)
- [ ] Speculative decoding
- [ ] Multi-node support
- [ ] Qwen / Mistral / GLM model support
- [ ] LoRA adapter support
- [ ] Prometheus metrics
- [ ] Docker deployment

---

## References

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [FlashInfer: Efficient and Customizable Kernels for Large Language Model Inference Serving](https://arxiv.org/abs/2401.08765)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [NCCL Documentation: Optimized Primitives for Multi-GPU Communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

---

## Contributing

Contributions welcome! Areas of interest:
- New model architectures (Qwen, Mistral, GLM etc.)
- Performance optimizations
- Quantization support
- Documentation improvements

---
