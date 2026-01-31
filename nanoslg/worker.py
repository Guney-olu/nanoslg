"""
Distributed inference worker.
Handles pipeline-parallel execution across GPUs.
"""

import os
import time
import traceback
from queue import Empty
from typing import Optional

import torch
import torch.distributed as dist
from transformers import AutoConfig

from .config import ModelConfig, STORE_PATH
from .models import LlamaStage, load_weights_into_stage
from .bench import Benchmark

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def run_worker(
    rank: int,
    world_size: int,
    model_config: ModelConfig,
    input_queue,          # mp.Queue for rank 0
    output_queue,         # mp.Queue for rank 0
    bench_queue,          # mp.Queue for benchmark results
):
    try:
        if rank == 0:
            if os.path.exists(STORE_PATH):
                os.remove(STORE_PATH)
            time.sleep(0.1)
        else:
            time.sleep(1.0)
        
        store = dist.FileStore(STORE_PATH, world_size)
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = DTYPE_MAP[model_config.dtype]
        
        # Load model stage
        hf_config = AutoConfig.from_pretrained(model_config.path)
        model = LlamaStage(hf_config, rank, model_config.device_map)
        model = model.to(dtype).to(device)
        load_weights_into_stage(model, model_config.path, hf_config, dtype)
        
        dist.barrier()
        if rank == 0:
            print(f"[Worker] All {world_size} GPU workers ready")
        
        # NO SO OPTIMISED INFERENCE LOOP        
        kv_caches = []
        benchmark = Benchmark(device=rank, enabled=True)
        
        while True:
            if rank == 0:
                # Poll for work
                try:
                    req = input_queue.get(timeout=0.5)
                    has_work = True
                except Empty:
                    has_work = False
                
                # Broadcasting the  work signal
                work_signal = torch.tensor([1 if has_work else 0], dtype=torch.int32, device=device)
                dist.broadcast(work_signal, src=0)
                
                if not has_work:
                    continue
                
                input_ids = req["input_ids"].to(device)
                max_tokens = req["max_tokens"]
                seq_len = input_ids.shape[1]
                
                # Broadcast metadata
                metadata = torch.tensor([seq_len, max_tokens], dtype=torch.long, device=device)
                dist.broadcast(metadata, src=0)
                
                # Start benchmarking
                benchmark.start(prompt_tokens=seq_len)
                
                with torch.no_grad():
                    with benchmark.track_prefill():
                        hidden = model(input_ids, kv_caches, None)
                        dist.send(hidden, dst=1)
                        
                        next_token = torch.zeros(1, dtype=torch.long, device=device)
                        dist.recv(next_token, src=world_size - 1)
                    
                    output_queue.put(next_token.item())
                    curr_token = next_token.view(1, 1)
                    
                    for i in range(max_tokens - 1):
                        with benchmark.track_decode():
                            hidden = model(curr_token, kv_caches, None)
                            dist.send(hidden, dst=1)
                            dist.recv(curr_token, src=world_size - 1)
                        
                        output_queue.put(curr_token.item())
                    
                    output_queue.put(None)
                
                benchmark.stop()
                if bench_queue:
                    bench_queue.put(benchmark.metrics.to_dict())
                print(benchmark.metrics)
                
                kv_caches = []
            
            else:                
                work_signal = torch.tensor([0], dtype=torch.int32, device=device)
                dist.broadcast(work_signal, src=0)
                
                if work_signal.item() == 0:
                    continue
                
                metadata = torch.zeros(2, dtype=torch.long, device=device)
                dist.broadcast(metadata, src=0)
                seq_len = metadata[0].item()
                max_tokens = metadata[1].item()
                
                with torch.no_grad():
                    # PREFILL
                    hidden = torch.zeros(1, seq_len, hf_config.hidden_size, dtype=dtype, device=device)
                    dist.recv(hidden, src=rank - 1)
                    
                    output = model(hidden, kv_caches, None)
                    
                    if model.is_last:
                        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True).to(torch.long)
                        dist.send(next_token, dst=0)
                    else:
                        dist.send(output, dst=rank + 1)
                        next_token = torch.zeros(1, dtype=torch.long, device=device)
                        dist.recv(next_token, src=world_size - 1)
                    
                    # DECODE
                    for i in range(max_tokens - 1):
                        hidden = torch.zeros(1, 1, hf_config.hidden_size, dtype=dtype, device=device)
                        dist.recv(hidden, src=rank - 1)
                        
                        output = model(hidden, kv_caches, None)
                        
                        if model.is_last:
                            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True).to(torch.long)
                            dist.send(next_token, dst=0)
                        else:
                            dist.send(output, dst=rank + 1)
                            dist.recv(next_token, src=world_size - 1)
                
                # Reset
                kv_caches = []
    
    except Exception as e:
        print(f"[Worker {rank}] FATAL ERROR: {e}")
        traceback.print_exc()
        raise