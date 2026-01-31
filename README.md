# NanoSLG

**Minimal Pipeline-Parallel LLM Inference Server**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status: Experimental](https://img.shields.io/badge/Status-Research%20Preview-orange)]()

---

## ✨ Features

| Feature | Description |
|------|-------------|
| 🔀 Pipeline Parallelism | Split model layers across multiple GPUs |
| 🌊 Streaming Responses | Server-Sent Events (SSE) token streaming |
| 📊 Built-in Benchmarking | TTFT, tokens/sec, memory usage |
| 🧩 Modular Design | Easy to add new models |
| 🎛️ Configurable | Simple Python config (no YAML) |

---

## Installation

### Prerequisites

- Python **3.10+**
- PyTorch **2.0+** with CUDA
- **2+ NVIDIA GPUs** (for pipeline parallelism)

### Setup

```bash
# Clone the repository
git clone https://github.com/Guney-olu/nanoslg
cd nanoslg

# Install dependencies
pip install torch transformers safetensors fastapi uvicorn requests

# Verify GPU setup
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir ./models/Llama-3.2-3B-Instruct
```

### Quick Start
```bash
python -m nanoslg --model llama-3.2-3b --port 8000

python inference.py --max-tokens 200 --benchmark "Explain relativity"
```

### High-Level Overview
```text
HTTP Client
   │
   ▼
FastAPI Server
   │
mp.Queue
   │
   ▼
GPU 0  ── NCCL ── GPU 1
Layers 0–13       Layers 14–27 + LM Head
```
![Flow Diagram](/img/flow.jpeg)
## Roadmap

- CUDA Graphs  
- `torch.compile`  
- Continuous batching  
- Quantization (INT8 / INT4)  
- Flash Attention  
- Speculative decoding  
- Tensor parallelism  
