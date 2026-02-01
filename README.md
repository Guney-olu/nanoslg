# NanoSLG
**Minimal Multi-Mode Parallel LLM Inference Server**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Experimental](https://img.shields.io/badge/Status-Research%20Preview-orange)]()

> A lightweight, educational LLM inference server with support for Pipeline Parallelism, Tensor Parallelism, and Hybrid (TP+PP) modes.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔀 **Pipeline Parallelism** | Split model layers across GPUs sequentially |
| ⚡ **Tensor Parallelism** | Shard weights across GPUs for parallel computation |
| 🔄 **Hybrid Mode** | Combine TP + PP for maximum scalability |
| 📦 **Continuous Batching** | Dynamic batch formation for higher throughput |
| 🌊 **Streaming Responses** | Server-Sent Events (SSE) token streaming |
| 📊 **Built-in Benchmarking** | TTFT, tokens/sec, memory tracking |
| 🔧 **torch.compile** | Optimized with PyTorch 2.0 compilation |
| 🧩 **Modular Design** | Easy to extend with new models |

---

## 📊 Performance Results

Tested on **2x NVIDIA L4 GPUs** with **Llama-3.1-8B-Instruct**:

| Mode | Batch Size | Throughput | TTFT | Notes |
|------|------------|------------|------|-------|
| **Pipeline Parallel** | 1 | ~25 tok/s | ~85ms | Lower memory per GPU |
| **Pipeline Parallel** | 4 | ~62 tok/s | ~95ms | Good scaling |
| **Tensor Parallel** | 1 | ~29 tok/s | ~60ms | Lower latency |
| **Tensor Parallel** | 4 | **~79 tok/s** | ~74ms | Best throughput |
| **Hybrid (2x2)** | 4 | ~100+ tok/s | ~70ms | 4 GPU setup |

### Throughput Comparison

```
Batch Size 4 Throughput (tok/s)
═══════════════════════════════════════════════════════════════
Pipeline (2 GPU) │████████████████████████████████░░░░░░░░│ 62
Tensor (2 GPU) │█████████████████████████████████████████│ 79
Hybrid (4 GPU) │████████████████████████████████████████████████│ 100+
═══════════════════════════════════════════════════════════════
```

---

## 🏗️ Architecture

### Parallelism Modes

```
┌─────────────────────────────────────────────────────────────────────┐
│ PARALLELISM MODES │
├─────────────────────────────────────────────────────────────────────┤
│ │
│ PIPELINE PARALLEL (PP) TENSOR PARALLEL (TP) │
│ ───────────────────── ───────────────────── │
│ │
│ ┌─────────┐ ┌─────────┬─────────┐ │
│ │ GPU 0 │ Layers 0-15 │ GPU 0 │ GPU 1 │ │
│ │ │ │ │ Head │ Head │ │
│ └────┬────┘ │ │ 0-15 │ 16-31 │ │
│ │ send ▼ └────┬────┴────┬────┘ │
│ ┌────▼────┐ │AllReduce│ │
│ │ GPU 1 │ Layers 16-31 └────┬────┘ │
│ │ + Head │ │ │
│ └─────────┘ ▼ │
│ Output │
│ │
│ HYBRID (TP + PP) │
│ ──────────────── │
│ │
│ PP Stage 0 PP Stage 1 │
│ ┌─────────┬─────────┐ ┌─────────┬─────────┐ │
│ │ GPU 0 │ GPU 1 │──▶│ GPU 2 │ GPU 3 │ │
│ │ TP Rank │ TP Rank │ │ TP Rank │ TP Rank │ │
│ │ 0 │ 1 │ │ 0 │ 1 │ │
│ └─────────┴─────────┘ └─────────┴─────────┘ │
│ Layers 0-15 (sharded) Layers 16-31 (sharded) │
│ │
└─────────────────────────────────────────────────────────────────────┘
```

### System Architecture

```
                              ┌──────────────────┐
                              │ HTTP Client │
                              │ (curl/Python) │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ FastAPI Server │
                              │ (async I/O) │
                              └────────┬─────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │ mp.Queue │
                         │ (request/response) │
                         └─────────────┬─────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │ │ │
              ▼ ▼ ▼
     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
     │ GPU 0 │ │ GPU 1 │ │ GPU N │
     │ ┌───────────┐ │ │ ┌───────────┐ │ │ ┌───────────┐ │
     │ │ TP Layers │ │────▶│ │ TP Layers │ │────▶│ │ TP Layers │ │
     │ │ (sharded) │ │NCCL │ │ (sharded) │ │NCCL │ │ + LM Head │ │
     │ └───────────┘ │ │ └───────────┘ │ │ └───────────┘ │
     │ │ │ │ │ │
     │ KV Cache │ │ KV Cache │ │ KV Cache │
     └─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python **3.10+**
- PyTorch **2.0+** with CUDA
- **2+ NVIDIA GPUs** (for parallelism)
- NCCL for GPU communication

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

## 📖 Usage

### Basic Commands

```bash
# Pipeline Parallel (default) - 2 GPUs
python -m nanoslg --model /path/to/model --mode pipeline --pp-size 2
# Tensor Parallel - 2 GPUs
python -m nanoslg --model /path/to/model --mode tensor --tp-size 2
# Hybrid Mode - 4 GPUs (2 TP x 2 PP)
python -m nanoslg --model /path/to/model --mode hybrid --tp-size 2 --pp-size 2
# With all options
python -m nanoslg \
    --model /path/to/Llama-3.1-8B-Instruct \
    --mode tensor \
    --tp-size 2 \
    --batch-size 4 \
    --dtype bfloat16 \
    --port 8000
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Path to HuggingFace model |
| `--mode` | `pipeline` | Parallelism mode: `pipeline`, `tensor`, `hybrid` |
| `--tp-size` | `1` | Tensor parallel degree |
| `--pp-size` | `2` | Pipeline parallel degree |
| `--batch-size` | `4` | Maximum batch size |
| `--dtype` | `bfloat16` | Data type: `float16`, `bfloat16`, `float32` |
| `--port` | `8000` | Server port |
| `--host` | `0.0.0.0` | Server host |

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

## 🔧 Configuration

### Registering Models

```python
# In nanoslg/config.py
from nanoslg.config import ModelConfig, register_model
# Pipeline Parallel setup
register_model(ModelConfig(
    name="llama-3.1-8b-pp",
    path="/path/to/Llama-3.1-8B-Instruct",
    parallel_mode="pipeline",
    pp_size=2,
    device_map={0: list(range(16)), 1: list(range(16, 32))},
))
# Tensor Parallel setup
register_model(ModelConfig(
    name="llama-3.1-8b-tp",
    path="/path/to/Llama-3.1-8B-Instruct",
    parallel_mode="tensor",
    tp_size=2,
))
# Hybrid setup (4 GPUs)
register_model(ModelConfig(
    name="llama-3.1-8b-hybrid",
    path="/path/to/Llama-3.1-8B-Instruct",
    parallel_mode="hybrid",
    tp_size=2,
    pp_size=2,
))
```

---

## 📁 Project Structure

```
nanoslg/
├── __init__.py
├── __main__.py # Entry point
├── config.py # Configuration & model registry
├── parallel.py # Parallelism infrastructure (TP/PP/Hybrid)
├── tp_layers.py # Tensor parallel layer implementations
├── models.py # Model architectures (LlamaTP, LlamaPP, LlamaHybrid)
├── scheduler.py # Batch scheduler for continuous batching
├── worker.py # Distributed inference workers
├── server.py # FastAPI server
└── benchmark.py # Benchmarking utilities
```

---

## 🗺️ Roadmap

### Completed ✅
- [x] Pipeline Parallelism (PP)
- [x] Tensor Parallelism (TP)
- [x] Hybrid TP + PP mode
- [x] Continuous batching
- [x] `torch.compile` integration
- [x] Streaming responses (SSE)
- [x] OpenAI-compatible API
- [x] Built-in benchmarking
- [x] Llama 3.x support

### In Progress
- [ ] Pre-allocated KV cache (reduce recompilation)
- [ ] CUDA Graphs for decode
- [ ] Better warmup strategies

### Planned
- [ ] Quantization (INT8 / INT4 / GPTQ / AWQ)
- [ ] Flash Attention integration
- [ ] Speculative decoding
- [ ] PagedAttention
- [ ] Prefix caching
- [ ] Multi-node support
- [ ] Qwen / Mistral / GLM model support
- [ ] LoRA adapter support
- [ ] Prometheus metrics
- [ ] Docker deployment

---

## ⚡ Performance Tips

### 1. Choose the Right Mode

| Scenario | Recommended Mode |
|----------|-----------------|
| 2 GPUs, memory constrained | Pipeline Parallel |
| 2 GPUs, latency sensitive | Tensor Parallel |
| 4+ GPUs, maximum throughput | Hybrid |
| Large batch inference | Tensor Parallel |

### 2. Optimize Settings

```bash
# For best throughput
python -m nanoslg \
    --mode tensor \
    --batch-size 8 \
    --dtype bfloat16
# For lowest latency
python -m nanoslg \
    --mode tensor \
    --batch-size 1 \
    --dtype float16
```
---

## 🐛 Troubleshooting

### Common Issues

**1. OOM Errors**

```bash
# Reduce batch size or use pipeline mode
python -m nanoslg --mode pipeline --batch-size 2
```

**2. Slow First Request**

- This is expected due to `torch.compile` warmup
- Subsequent requests will be faster

---

## 📚 References

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- New model architectures (Qwen, Mistral, GLM etc.)
- Performance optimizations
- Quantization support
- Documentation improvements

---
