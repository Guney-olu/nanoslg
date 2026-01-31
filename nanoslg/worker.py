"""
V0.2
Batched distributed inference worker.
"""

import os
import time
import traceback
import uuid
from queue import Empty
from typing import Optional, List, Dict

import torch
import torch.distributed as dist
from transformers import AutoConfig

from .config import ModelConfig, STORE_PATH
from .models import LlamaStage, load_weights_into_stage
from .scheduler import Request, Batch, BatchScheduler, prepare_batch_inputs

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

EOS_TOKEN_IDS = {128001, 128008, 128009}


def setup_nccl_env():
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


class BatchedBuffers:
    """Pre-allocated buffers for batched inference."""
    
    def __init__(self, max_batch_size: int, hidden_size: int, max_seq_len: int, dtype, device):
        self.device = device
        self.max_batch_size = max_batch_size
        
        self.decode_hidden = torch.empty(max_batch_size, 1, hidden_size, dtype=dtype, device=device)
        self.tokens = torch.empty(max_batch_size, dtype=torch.long, device=device)
        self.prefill_hidden = torch.empty(max_batch_size, max_seq_len, hidden_size, dtype=dtype, device=device)
        self.work_signal = torch.empty(1, dtype=torch.int32, device=device)
        self.metadata = torch.empty(4, dtype=torch.long, device=device)
    
    def get_prefill_buffer(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return self.prefill_hidden[:batch_size, :seq_len, :].contiguous()
    
    def get_decode_buffer(self, batch_size: int) -> torch.Tensor:
        return self.decode_hidden[:batch_size, :, :].contiguous()
    
    def get_tokens(self, batch_size: int) -> torch.Tensor:
        return self.tokens[:batch_size].contiguous()


def warmup_batched_model(model, device, hf_config, dtype, is_first: bool, max_batch_size: int):
    """Warmup with various batch sizes."""
    print(f"[Worker] Warming up for batch sizes 1-{max_batch_size}...")
    
    warmup_configs = [(1, 1), (1, 16), (2, 1), (2, 16), (4, 1), (4, 16)]
    
    with torch.inference_mode():
        for batch_size, seq_len in warmup_configs:
            if batch_size > max_batch_size:
                continue
            
            if is_first:
                warmup_input = torch.randint(0, 100, (batch_size, seq_len), device=device)
            else:
                warmup_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=dtype, device=device)
            
            caches = [[] for _ in range(batch_size)]
            _ = model(warmup_input, caches, None)
            
            if is_first:
                decode_input = torch.randint(0, 100, (batch_size, 1), device=device)
            else:
                decode_input = torch.randn(batch_size, 1, hf_config.hidden_size, dtype=dtype, device=device)
            
            for _ in range(2):
                _ = model(decode_input, caches, None)
            
            del warmup_input, decode_input, caches
    
    torch.cuda.empty_cache()
    print(f"[Worker] Batched warmup complete")


def run_worker(
    rank: int,
    world_size: int,
    model_config: ModelConfig,
    input_queue,      # mp.Queue for requests
    response_queue,   # mp.Queue for (request_id, token) pairs
    bench_queue,
    max_batch_size: int = 4,
):
    try:
        setup_nccl_env()
        
        if rank == 0:
            if os.path.exists(STORE_PATH):
                os.remove(STORE_PATH)
            time.sleep(0.1)
        else:
            time.sleep(1.0)
        
        store = dist.FileStore(STORE_PATH, world_size)
        dist.init_process_group(backend="nccl", store=store, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = DTYPE_MAP[model_config.dtype]
        
        hf_config = AutoConfig.from_pretrained(model_config.path)
        model = LlamaStage(hf_config, rank, model_config.device_map)
        model = model.to(dtype).to(device)
        load_weights_into_stage(model, model_config.path, hf_config, dtype)
        
        is_first_stage = model.is_first
        is_last_stage = model.is_last
        
        if rank == 0:
            print(f"[Worker {rank}] Compiling model...")
        
        model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
        warmup_batched_model(model, device, hf_config, dtype, is_first_stage, max_batch_size)
        
        buffers = BatchedBuffers(
            max_batch_size=max_batch_size,
            hidden_size=hf_config.hidden_size,
            max_seq_len=model_config.max_seq_len,
            dtype=dtype,
            device=device
        )
        
        scheduler = BatchScheduler(max_batch_size=max_batch_size, max_wait_time=0.05) if rank == 0 else None
        
        dist.barrier()
        if rank == 0:
            print(f"[Worker] All {world_size} GPU workers ready")
            print(f"[Worker] Batching enabled: max_batch_size={max_batch_size}")
        
        request_count = 0
        
        while True:
            if rank == 0:
                # Collect requests
                while True:
                    try:
                        req_data = input_queue.get_nowait()
                        request = Request(
                            request_id=req_data["request_id"],
                            input_ids=req_data["input_ids"],
                            max_tokens=req_data["max_tokens"],
                        )
                        scheduler.add_request(request)
                    except Empty:
                        break
                
                batch = scheduler.try_form_batch()
                
                if batch is None:
                    buffers.work_signal.fill_(0)
                    dist.broadcast(buffers.work_signal, src=0)
                    time.sleep(0.001)
                    continue
                
                buffers.work_signal.fill_(1)
                dist.broadcast(buffers.work_signal, src=0)
                
                batch_size = batch.batch_size
                request_count += batch_size
                
                input_ids, attention_mask, prompt_lens = prepare_batch_inputs(batch, device)
                max_seq_len = input_ids.shape[1]
                max_tokens = max(r.max_tokens for r in batch.requests)
                
                buffers.metadata[0] = batch_size
                buffers.metadata[1] = max_seq_len
                buffers.metadata[2] = max_tokens
                buffers.metadata[3] = 0
                dist.broadcast(buffers.metadata, src=0)
                
                batch.kv_caches = [[] for _ in range(batch_size)]
                
                torch.cuda.synchronize(device)
                t_start = time.perf_counter()
                
                with torch.inference_mode():
                    # PREFILL
                    hidden = model(input_ids, batch.kv_caches, None)
                    dist.send(hidden, dst=1)
                    
                    recv_tokens = buffers.get_tokens(batch_size)
                    dist.recv(recv_tokens, src=world_size - 1)
                    
                    torch.cuda.synchronize(device)
                    t_first_token = time.perf_counter()
                    
                    # Send first tokens
                    for i, req in enumerate(batch.requests):
                        token_id = recv_tokens[i].item()
                        response_queue.put((req.request_id, token_id))
                        req.generated_tokens = 1
                        
                        if token_id in EOS_TOKEN_IDS:
                            req.finished = True
                            req.finish_reason = "stop"
                            response_queue.put((req.request_id, None))
                    
                    curr_tokens = recv_tokens.clone()
                    
                    # DECODE
                    for step in range(max_tokens - 1):
                        if batch.all_finished:
                            break
                        
                        decode_input = curr_tokens.unsqueeze(1)
                        hidden = model(decode_input, batch.kv_caches, None)
                        dist.send(hidden, dst=1)
                        dist.recv(recv_tokens, src=world_size - 1)
                        
                        for i, req in enumerate(batch.requests):
                            if req.finished:
                                continue
                            
                            token_id = recv_tokens[i].item()
                            response_queue.put((req.request_id, token_id))
                            req.generated_tokens += 1
                            
                            if token_id in EOS_TOKEN_IDS or req.generated_tokens >= req.max_tokens:
                                req.finished = True
                                req.finish_reason = "stop" if token_id in EOS_TOKEN_IDS else "length"
                                response_queue.put((req.request_id, None))
                        
                        curr_tokens = recv_tokens.clone()
                    
                    # Finalize remaining
                    for req in batch.requests:
                        if not req.finished:
                            req.finished = True
                            response_queue.put((req.request_id, None))
                
                torch.cuda.synchronize(device)
                t_end = time.perf_counter()
                
                total_generated = sum(r.generated_tokens for r in batch.requests)
                ttft_ms = (t_first_token - t_start) * 1000
                total_ms = (t_end - t_start) * 1000
                decode_ms = total_ms - ttft_ms
                tps = total_generated / (total_ms / 1000) if total_ms > 0 else 0
                decode_tps = (total_generated - batch_size) / (decode_ms / 1000) if decode_ms > 0 else 0
                
                print(f"\n{'='*50}")
                print(f"BATCH RESULTS (Requests #{request_count - batch_size + 1}-{request_count})")
                print(f"{'='*50}")
                print(f"  Batch size:        {batch_size}")
                print(f"  Prompt tokens:     {sum(prompt_lens)}")
                print(f"  Generated tokens:  {total_generated}")
                print(f"  TTFT:              {ttft_ms:.1f} ms")
                print(f"  Total time:        {total_ms:.1f} ms")
                print(f"  Throughput:        {tps:.1f} tok/s")
                print(f"  Decode throughput: {decode_tps:.1f} tok/s")
                print(f"{'='*50}\n")
                
                if bench_queue:
                    bench_queue.put({
                        "batch_size": batch_size,
                        "total_generated": total_generated,
                        "tokens_per_second": tps,
                    })
            
            else:
                dist.broadcast(buffers.work_signal, src=0)
                
                if buffers.work_signal.item() == 0:
                    time.sleep(0.001)
                    continue
                
                dist.broadcast(buffers.metadata, src=0)
                batch_size = buffers.metadata[0].item()
                max_seq_len = buffers.metadata[1].item()
                max_tokens = buffers.metadata[2].item()
                
                kv_caches = [[] for _ in range(batch_size)]
                
                with torch.inference_mode():
                    prefill_buffer = buffers.get_prefill_buffer(batch_size, max_seq_len)
                    dist.recv(prefill_buffer, src=rank - 1)
                    
                    output = model(prefill_buffer, kv_caches, None)
                    
                    if is_last_stage:
                        last_logits = output[:, -1, :]
                        tokens = last_logits.argmax(dim=-1)
                        dist.send(tokens, dst=0)
                    else:
                        dist.send(output, dst=rank + 1)
                        tokens = buffers.get_tokens(batch_size)
                        dist.recv(tokens, src=world_size - 1)
                    
                    for step in range(max_tokens - 1):
                        decode_buffer = buffers.get_decode_buffer(batch_size)
                        dist.recv(decode_buffer, src=rank - 1)
                        
                        output = model(decode_buffer, kv_caches, None)
                        
                        if is_last_stage:
                            last_logits = output[:, -1, :]
                            tokens = last_logits.argmax(dim=-1)
                            dist.send(tokens, dst=0)
                        else:
                            dist.send(output, dst=rank + 1)
                            dist.recv(tokens, src=world_size - 1)
    
    except Exception as e:
        print(f"[Worker {rank}] FATAL ERROR: {e}")
        traceback.print_exc()
        raise
