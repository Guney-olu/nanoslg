"""
Entry point: python -m nanoslg [--model MODEL_NAME] [--port PORT]
"""

import argparse
import os
import time
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from .config import get_model_config, list_models, STORE_PATH, DEFAULT_PORT
from .worker import run_worker
from .server import app, init_server


def main():
    parser = argparse.ArgumentParser(description="NanoSLG - Pipeline Parallel LLM Server")
    parser.add_argument("--model", type=str, default="llama-3.2-3b", 
                        help=f"Model to serve. Available: {list_models()}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    args = parser.parse_args()
    
    # Get model config
    model_config = get_model_config(args.model)
    print(f"\n{'='*60}")
    print(f"  NanoSLG - Starting {model_config.name}")
    print(f"  Path: {model_config.path}")
    print(f"  GPUs: {model_config.world_size}")
    print(f"  Layers per GPU: {model_config.device_map}")
    print(f"{'='*60}\n")
    
    mp.set_start_method("spawn", force=True)
    
    request_queue = mp.Queue()
    response_queue = mp.Queue()
    bench_queue = mp.Queue()
    
    os.system(f"fuser -k {args.port}/tcp > /dev/null 2>&1")
    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)
    
    workers = []
    for rank in range(model_config.world_size):
        # Only rank 0 gets the queues
        req_q = request_queue if rank == 0 else None
        res_q = response_queue if rank == 0 else None
        bench_q = bench_queue if rank == 0 else None
        
        p = mp.Process(
            target=run_worker,
            args=(rank, model_config.world_size, model_config, req_q, res_q, bench_q),
        )
        p.start()
        workers.append(p)
    
    print("[Main] Waiting for workers to initialize...")
    time.sleep(10)
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    init_server(tokenizer, request_queue, response_queue, bench_queue, model_config)
    
    print(f"[Main] Starting API server on {args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    
    for p in workers:
        p.terminate()


if __name__ == "__main__":
    main()