"""
Entry point: python -m nanoslg --model MODEL --mode [pipeline|tensor|hybrid]
"""
import os
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["TORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoConfig

from .config import get_model_config, ModelConfig, STORE_PATH, DEFAULT_PORT, register_model
from .parallel import ParallelMode, get_layers_for_pp_rank
from .worker import run_worker
from .server import app, init_server

torch.set_float32_matmul_precision('high')


def main():
    parser = argparse.ArgumentParser(description="NanoSLG - Multi-Mode Parallel LLM Server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--mode", type=str, default="pipeline", 
                       choices=["pipeline", "tensor", "hybrid"],
                       help="Parallelism mode")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=2, help="Pipeline parallel size")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    
    # Create model config from args
    model_name = os.path.basename(args.model.rstrip('/'))
    
    # Auto-detect parallelism sizes
    num_gpus = torch.cuda.device_count()
    if args.mode == "tensor":
        tp_size = args.tp_size if args.tp_size > 1 else num_gpus
        pp_size = 1
    elif args.mode == "pipeline":
        tp_size = 1
        pp_size = args.pp_size if args.pp_size > 1 else num_gpus
    else:  # hybrid
        tp_size = args.tp_size
        pp_size = args.pp_size
        assert tp_size * pp_size == num_gpus, \
            f"TP({tp_size}) * PP({pp_size}) must equal num_gpus({num_gpus})"
    
    # Get layer count
    hf_config = AutoConfig.from_pretrained(args.model)
    total_layers = hf_config.num_hidden_layers
    
    # Create layer splits for PP
    device_map = None
    if args.mode in ("pipeline", "hybrid"):
        device_map = {}
        for pp_rank in range(pp_size):
            device_map[pp_rank] = get_layers_for_pp_rank(total_layers, pp_size, pp_rank)
    
    model_config = ModelConfig(
        name=model_name,
        path=args.model,
        dtype=args.dtype,
        parallel_mode=args.mode,
        tp_size=tp_size,
        pp_size=pp_size,
        device_map=device_map,
    )
    
    parallel_config = model_config.get_parallel_config(total_layers)
    
    print(f"\n{'='*60}")
    print(f"  NanoSLG v0.3 - {model_config.name}")
    print(f"{'='*60}")
    print(f"  Path:          {model_config.path}")
    print(f"  Mode:          {args.mode.upper()}")
    print(f"  TP Size:       {tp_size}")
    print(f"  PP Size:       {pp_size}")
    print(f"  Total GPUs:    {model_config.world_size}")
    print(f"  Total Layers:  {total_layers}")
    print(f"  Batch Size:    {args.batch_size}")
    if device_map:
        print(f"  Layer Split:   {device_map}")
    print(f"{'='*60}\n")
    
    mp.set_start_method("spawn", force=True)
    
    request_queue = mp.Queue()
    response_queue = mp.Queue()
    bench_queue = mp.Queue()
    
    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)
    
    workers = []
    for rank in range(model_config.world_size):
        # Only rank 0 needs queues
        req_q = request_queue if rank == 0 else None
        res_q = response_queue if rank == 0 else None
        bench_q = bench_queue if rank == 0 else None
        
        p = mp.Process(
            target=run_worker,
            args=(rank, model_config.world_size, model_config, parallel_config,
                  req_q, res_q, bench_q, args.batch_size),
        )
        p.start()
        workers.append(p)
    
    print("[Main] Waiting for workers to initialize...")
    time.sleep(20)
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    init_server(tokenizer, request_queue, response_queue, bench_queue, model_config)
    
    print(f"[Main] Starting API server on {args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    
    for p in workers:
        p.terminate()


if __name__ == "__main__":
    main()