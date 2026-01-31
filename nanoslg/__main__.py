"""
Entry point: python -m nanoslg [--model MODEL_NAME] [--batch-size BATCH_SIZE]
"""
import os
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["TORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from .config import get_model_config, list_models, STORE_PATH, DEFAULT_PORT
from .worker import run_worker
from .server import app, init_server

torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description="NanoSLG - Batched Pipeline Parallel LLM Server")
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    model_config = get_model_config(args.model)
    print(f"\n{'='*60}")
    print(f"  NanoSLG - Starting {model_config.name}")
    print(f"  Path: {model_config.path}")
    print(f"  GPUs: {model_config.world_size}")
    print(f"  Max Batch Size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    mp.set_start_method("spawn", force=True)
    
    request_queue = mp.Queue()
    response_queue = mp.Queue()  # Shared response queue
    bench_queue = mp.Queue()
    
    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)
    
    workers = []
    for rank in range(model_config.world_size):
        req_q = request_queue if rank == 0 else None
        res_q = response_queue if rank == 0 else None
        bench_q = bench_queue if rank == 0 else None
        
        p = mp.Process(
            target=run_worker,
            args=(rank, model_config.world_size, model_config, req_q, res_q, bench_q, args.batch_size),
        )
        p.start()
        workers.append(p)
    
    print("[Main] Waiting for workers to initialize...")
    time.sleep(15)
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    init_server(tokenizer, request_queue, response_queue, bench_queue, model_config)
    
    print(f"[Main] Starting API server on {args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    
    for p in workers:
        p.terminate()


if __name__ == "__main__":
    main()