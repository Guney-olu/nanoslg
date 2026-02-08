"""
Entry point: python -m nanoslg --model MODEL --mode [pipeline|tensor|hybrid]
"""
import os
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["TORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse, time
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoConfig

from .config import get_model_config, ModelConfig, STORE_PATH, DEFAULT_PORT
from .parallel import ParallelMode, get_layers_for_pp_rank
from .worker import run_worker
from .server import app, init_server

torch.set_float32_matmul_precision("high")


def main():
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64 
    parser = argparse.ArgumentParser(description="NanoSLG – Paged-Radix LLM Server")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, default="pipeline",
                        choices=["pipeline", "tensor", "hybrid"])
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Hard cap on KV cache pages (0 = auto from --kv-memory)")
    parser.add_argument("--kv-memory", type=float, default=0.30,
                        help="Fraction of GPU memory for KV cache")
    parser.add_argument("--backend", type=str, default="auto",
                    choices=["auto", "flashinfer", "contiguous"],
                    help="KV cache backend (auto detects GPU)")
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if args.mode == "tensor":
        tp = args.tp_size if args.tp_size > 1 else num_gpus; pp = 1
    elif args.mode == "pipeline":
        tp = 1; pp = args.pp_size if args.pp_size > 1 else num_gpus
    else:
        tp = args.tp_size; pp = args.pp_size
        assert tp * pp == num_gpus

    hf_config = AutoConfig.from_pretrained(args.model)
    total_layers = hf_config.num_hidden_layers
    name = os.path.basename(args.model.rstrip("/"))

    device_map = None
    if args.mode in ("pipeline", "hybrid"):
        device_map = {r: get_layers_for_pp_rank(total_layers, pp, r) for r in range(pp)}

    model_config = ModelConfig(
        name=name, path=args.model, dtype=args.dtype,
        parallel_mode=args.mode, tp_size=tp, pp_size=pp,
        device_map=device_map, page_size=args.page_size,
        kv_memory_fraction=args.kv_memory,
        max_kv_pages=args.max_pages,          
        enable_prefix_caching=not args.no_prefix_cache,
        kv_backend=args.backend,        
    )
    parallel_config = model_config.get_parallel_config(total_layers)

    print(f"\n{'='*60}")
    print(f"  NanoSLG v0.5 – {name}")
    print(f"{'='*60}")
    print(f" Mode:{args.mode.upper()}")
    print(f" TP×PP:{tp}×{pp} = {model_config.world_size} GPUs")
    print(f" Layers:{total_layers}")
    print(f" Page size:{args.page_size}")
    print(f" KV memory:{args.kv_memory:.0%}")
    print(f" Prefix caching:{model_config.enable_prefix_caching}")
    print(f" Batch size:{args.batch_size}")
    if device_map:
        print(f" Layer split: {device_map}")
    print(f"{'='*60}\n")

    mp.set_start_method("spawn", force=True)
    request_queue = mp.Queue()
    response_queue = mp.Queue()
    bench_queue = mp.Queue()

    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)

    workers = []
    for r in range(model_config.world_size):
        p = mp.Process(
            target=run_worker,
            args=(r, model_config.world_size, model_config, parallel_config,
                  request_queue if r == 0 else None,
                  response_queue if r == 0 else None,
                  bench_queue if r == 0 else None,
                  args.batch_size))
        p.start(); workers.append(p)

    print("[Main] Waiting for workers…")
    time.sleep(20)

    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    init_server(tokenizer, request_queue, response_queue, bench_queue, model_config)

    print(f"[Main] API server on {args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    for p in workers: p.terminate()


if __name__ == "__main__":
    main()