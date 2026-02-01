"""
V0.3 - Multi-mode distributed inference workers.
Supports: Pipeline Parallel, Tensor Parallel, Hybrid (TP+PP)
"""

import os
import time
import traceback
import torch
import torch.distributed as dist
from queue import Empty
from transformers import AutoConfig

from .config import ModelConfig, STORE_PATH
from .parallel import (
    ParallelMode, ParallelConfig, ParallelContext,
    get_layers_for_pp_rank, pp_send, pp_recv,
)
from .models import create_model, load_weights
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


class WorkerBuffers:
    """Pre-allocated communication buffers."""
    
    def __init__(self, max_batch_size: int, hidden_size: int, max_seq_len: int, dtype, device):
        self.device = device
        self.max_batch_size = max_batch_size
        
        self.decode_hidden = torch.empty(max_batch_size, 1, hidden_size, dtype=dtype, device=device)
        self.tokens = torch.empty(max_batch_size, dtype=torch.long, device=device)
        self.prefill_hidden = torch.empty(max_batch_size, max_seq_len, hidden_size, dtype=dtype, device=device)
        self.work_signal = torch.empty(1, dtype=torch.int32, device=device)
        self.metadata = torch.empty(4, dtype=torch.long, device=device)
    
    def get_prefill_buffer(self, batch_size: int, seq_len: int):
        return self.prefill_hidden[:batch_size, :seq_len, :].contiguous()
    
    def get_decode_buffer(self, batch_size: int):
        return self.decode_hidden[:batch_size, :, :].contiguous()
    
    def get_tokens(self, batch_size: int):
        return self.tokens[:batch_size].contiguous()


def warmup_model(model, device, hf_config, dtype, parallel_config: ParallelConfig, max_batch: int):
    """Warmup model with various batch/sequence sizes."""
    ctx = ParallelContext.get()
    is_first = ctx.is_first_pp_stage if parallel_config.pp_size > 1 else True
    
    configs = [(1, 1), (1, 16), (2, 1), (4, 1)]
    
    with torch.inference_mode():
        for batch, seq in configs:
            if batch > max_batch:
                continue
            
            if is_first:
                x = torch.randint(0, 100, (batch, seq), device=device)
            else:
                x = torch.randn(batch, seq, hf_config.hidden_size, dtype=dtype, device=device)
            
            caches = [[] for _ in range(batch)]
            _ = model(x, caches, None)
            
            # Decode step
            if is_first:
                decode = torch.randint(0, 100, (batch, 1), device=device)
            else:
                decode = torch.randn(batch, 1, hf_config.hidden_size, dtype=dtype, device=device)
            
            for _ in range(2):
                _ = model(decode, caches, None)
            
            del x, decode, caches
    
    torch.cuda.empty_cache()
    if ctx.rank == 0:
        print(f"[Worker] Warmup complete")


def run_tp_worker(
    rank: int,
    world_size: int,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    input_queue,
    response_queue,
    bench_queue,
    max_batch_size: int = 4,
):
    """
    Tensor Parallel worker.
    All ranks process same batches, synchronize via all-reduce.
    Only rank 0 handles scheduling and responses.
    """
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
        
        # Initialize parallel context
        ctx = ParallelContext.initialize(parallel_config, rank)
        ctx.setup_groups()
        
        hf_config = AutoConfig.from_pretrained(model_config.path)
        
        # Create and load TP model
        model = create_model(hf_config, parallel_config)
        model = model.to(dtype).to(device)
        load_weights(model, model_config.path, hf_config, dtype, parallel_config)
        
        if rank == 0:
            print(f"[TP Worker {rank}] Compiling...")
        
        model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
        warmup_model(model, device, hf_config, dtype, parallel_config, max_batch_size)
        
        buffers = WorkerBuffers(max_batch_size, hf_config.hidden_size, model_config.max_seq_len, dtype, device)
        scheduler = BatchScheduler(max_batch_size=max_batch_size) if rank == 0 else None
        
        dist.barrier()
        if rank == 0:
            print(f"[TP Worker] All {world_size} GPUs ready (TP mode)")
        
        while True:
            # Rank 0 schedules and broadcasts
            if rank == 0:
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
                
                input_ids, attention_mask, prompt_lens = prepare_batch_inputs(batch, device)
                batch_size = batch.batch_size
                max_seq_len = input_ids.shape[1]
                max_tokens = max(r.max_tokens for r in batch.requests)
                
                # Broadcast metadata
                buffers.metadata[0] = batch_size
                buffers.metadata[1] = max_seq_len
                buffers.metadata[2] = max_tokens
                dist.broadcast(buffers.metadata, src=0)
                
                # Broadcast input_ids
                dist.broadcast(input_ids, src=0)
            
            else:
                dist.broadcast(buffers.work_signal, src=0)
                if buffers.work_signal.item() == 0:
                    time.sleep(0.001)
                    continue
                
                dist.broadcast(buffers.metadata, src=0)
                batch_size = buffers.metadata[0].item()
                max_seq_len = buffers.metadata[1].item()
                max_tokens = buffers.metadata[2].item()
                
                input_ids = torch.empty(batch_size, max_seq_len, dtype=torch.long, device=device)
                dist.broadcast(input_ids, src=0)
                batch = None
            
            # All ranks execute model (TP all-reduce happens internally)
            kv_caches = [[] for _ in range(batch_size)]
            
            torch.cuda.synchronize(device)
            t_start = time.perf_counter()
            
            with torch.inference_mode():
                # Prefill
                logits = model(input_ids, kv_caches, None)
                last_logits = logits[:, -1, :]
                tokens = last_logits.argmax(dim=-1)
                
                torch.cuda.synchronize(device)
                t_first = time.perf_counter()
                
                # Rank 0 sends responses
                if rank == 0:
                    for i, req in enumerate(batch.requests):
                        token_id = tokens[i].item()
                        response_queue.put((req.request_id, token_id))
                        req.generated_tokens = 1
                        if token_id in EOS_TOKEN_IDS:
                            req.finished = True
                            response_queue.put((req.request_id, None))
                
                curr_tokens = tokens.clone()
                
                # Decode loop
                for step in range(max_tokens - 1):
                    if rank == 0 and batch.all_finished:
                        break
                    
                    decode_input = curr_tokens.unsqueeze(1)
                    logits = model(decode_input, kv_caches, None)
                    last_logits = logits[:, -1, :]
                    tokens = last_logits.argmax(dim=-1)
                    
                    if rank == 0:
                        for i, req in enumerate(batch.requests):
                            if req.finished:
                                continue
                            token_id = tokens[i].item()
                            response_queue.put((req.request_id, token_id))
                            req.generated_tokens += 1
                            if token_id in EOS_TOKEN_IDS or req.generated_tokens >= req.max_tokens:
                                req.finished = True
                                response_queue.put((req.request_id, None))
                    
                    curr_tokens = tokens.clone()
                    
                    # Sync finish status across TP ranks
                    if rank == 0:
                        finished = 1 if batch.all_finished else 0
                        finished_t = torch.tensor([finished], device=device)
                    else:
                        finished_t = torch.tensor([0], device=device)
                    dist.broadcast(finished_t, src=0)
                    if finished_t.item() == 1:
                        break
                
                # Finalize
                if rank == 0:
                    for req in batch.requests:
                        if not req.finished:
                            response_queue.put((req.request_id, None))
            
            torch.cuda.synchronize(device)
            t_end = time.perf_counter()
            
            if rank == 0:
                total_gen = sum(r.generated_tokens for r in batch.requests)
                ttft = (t_first - t_start) * 1000
                total_ms = (t_end - t_start) * 1000
                tps = total_gen / (total_ms / 1000) if total_ms > 0 else 0
                
                print(f"\n[TP Batch] size={batch_size}, tokens={total_gen}, TTFT={ttft:.1f}ms, {tps:.1f} tok/s")
                
                if bench_queue:
                    bench_queue.put({"batch_size": batch_size, "tokens_per_second": tps})
    
    except Exception as e:
        print(f"[TP Worker {rank}] ERROR: {e}")
        traceback.print_exc()
        raise

def run_pp_worker(
    rank: int,
    world_size: int,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    input_queue,
    response_queue,
    bench_queue,
    max_batch_size: int = 4,
):
    """Pipeline Parallel worker (original implementation)."""
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
        
        ctx = ParallelContext.initialize(parallel_config, rank)
        ctx.setup_groups()
        
        hf_config = AutoConfig.from_pretrained(model_config.path)
        
        model = create_model(hf_config, parallel_config)
        model = model.to(dtype).to(device)
        load_weights(model, model_config.path, hf_config, dtype, parallel_config)
        
        is_first = ctx.is_first_pp_stage
        is_last = ctx.is_last_pp_stage
        
        if rank == 0:
            print(f"[PP Worker {rank}] Compiling...")
        
        model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
        warmup_model(model, device, hf_config, dtype, parallel_config, max_batch_size)
        
        buffers = WorkerBuffers(max_batch_size, hf_config.hidden_size, model_config.max_seq_len, dtype, device)
        scheduler = BatchScheduler(max_batch_size=max_batch_size) if rank == 0 else None
        
        dist.barrier()
        if rank == 0:
            print(f"[PP Worker] All {world_size} GPUs ready (PP mode)")
        
        while True:
            if rank == 0:
                # Scheduler logic
                while True:
                    try:
                        req_data = input_queue.get_nowait()
                        scheduler.add_request(Request(
                            request_id=req_data["request_id"],
                            input_ids=req_data["input_ids"],
                            max_tokens=req_data["max_tokens"],
                        ))
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
                
                input_ids, _, prompt_lens = prepare_batch_inputs(batch, device)
                batch_size = batch.batch_size
                max_seq_len = input_ids.shape[1]
                max_tokens = max(r.max_tokens for r in batch.requests)
                
                buffers.metadata[0] = batch_size
                buffers.metadata[1] = max_seq_len
                buffers.metadata[2] = max_tokens
                dist.broadcast(buffers.metadata, src=0)
                
                kv_caches = [[] for _ in range(batch_size)]
                
                torch.cuda.synchronize(device)
                t_start = time.perf_counter()
                
                with torch.inference_mode():
                    # Prefill
                    hidden = model(input_ids, kv_caches, None)
                    dist.send(hidden, dst=1)
                    
                    recv_tokens = buffers.get_tokens(batch_size)
                    dist.recv(recv_tokens, src=world_size - 1)
                    
                    torch.cuda.synchronize(device)
                    t_first = time.perf_counter()
                    
                    for i, req in enumerate(batch.requests):
                        token_id = recv_tokens[i].item()
                        response_queue.put((req.request_id, token_id))
                        req.generated_tokens = 1
                        if token_id in EOS_TOKEN_IDS:
                            req.finished = True
                            response_queue.put((req.request_id, None))
                    
                    curr_tokens = recv_tokens.clone()
                    
                    # Decode
                    for step in range(max_tokens - 1):
                        if batch.all_finished:
                            break
                        
                        decode_input = curr_tokens.unsqueeze(1)
                        hidden = model(decode_input, kv_caches, None)
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
                                response_queue.put((req.request_id, None))
                        
                        curr_tokens = recv_tokens.clone()
                    
                    for req in batch.requests:
                        if not req.finished:
                            response_queue.put((req.request_id, None))
                
                torch.cuda.synchronize(device)
                t_end = time.perf_counter()
                
                total_gen = sum(r.generated_tokens for r in batch.requests)
                ttft = (t_first - t_start) * 1000
                total_ms = (t_end - t_start) * 1000
                tps = total_gen / (total_ms / 1000) if total_ms > 0 else 0
                print(f"\n[PP Batch] size={batch_size}, tokens={total_gen}, TTFT={ttft:.1f}ms, {tps:.1f} tok/s")
            
            else:
                # Non-rank-0 worker
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
                    prefill_buf = buffers.get_prefill_buffer(batch_size, max_seq_len)
                    dist.recv(prefill_buf, src=rank - 1)
                    
                    output = model(prefill_buf, kv_caches, None)
                    
                    if is_last:
                        tokens = output[:, -1, :].argmax(dim=-1)
                        dist.send(tokens, dst=0)
                    else:
                        dist.send(output, dst=rank + 1)
                        tokens = buffers.get_tokens(batch_size)
                        dist.recv(tokens, src=world_size - 1)
                    
                    for step in range(max_tokens - 1):
                        decode_buf = buffers.get_decode_buffer(batch_size)
                        dist.recv(decode_buf, src=rank - 1)
                        
                        output = model(decode_buf, kv_caches, None)
                        
                        if is_last:
                            tokens = output[:, -1, :].argmax(dim=-1)
                            dist.send(tokens, dst=0)
                        else:
                            dist.send(output, dst=rank + 1)
                            dist.recv(tokens, src=world_size - 1)
    
    except Exception as e:
        print(f"[PP Worker {rank}] ERROR: {e}")
        traceback.print_exc()
        raise

def run_hybrid_worker(
    rank: int,
    world_size: int,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    input_queue,
    response_queue,
    bench_queue,
    max_batch_size: int = 4,
):
    """
    Hybrid TP+PP worker.
    
    Communication pattern:
    - TP: all-reduce within TP groups (handled by model)
    - PP: send/recv between PP stages (only TP rank 0 does this)
    """
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
        
        ctx = ParallelContext.initialize(parallel_config, rank)
        ctx.setup_groups()
        
        hf_config = AutoConfig.from_pretrained(model_config.path)
        
        model = create_model(hf_config, parallel_config)
        model = model.to(dtype).to(device)
        load_weights(model, model_config.path, hf_config, dtype, parallel_config)
        
        is_first_pp = ctx.is_first_pp_stage
        is_last_pp = ctx.is_last_pp_stage
        is_tp_master = ctx.is_tp_master
        
        if rank == 0:
            print(f"[Hybrid {rank}] PP={ctx.pp_rank}, TP={ctx.tp_rank}, compiling...")
        
        model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
        warmup_model(model, device, hf_config, dtype, parallel_config, max_batch_size)
        
        buffers = WorkerBuffers(max_batch_size, hf_config.hidden_size, model_config.max_seq_len, dtype, device)
        scheduler = BatchScheduler(max_batch_size=max_batch_size) if rank == 0 else None
        
        dist.barrier()
        if rank == 0:
            print(f"[Hybrid] All {world_size} GPUs ready (TP={parallel_config.tp_size}, PP={parallel_config.pp_size})")
        
        while True:
            # === FIRST PP STAGE ===
            if is_first_pp:
                if is_tp_master:
                    # Rank 0 handles scheduling
                    while True:
                        try:
                            req_data = input_queue.get_nowait()
                            scheduler.add_request(Request(
                                request_id=req_data["request_id"],
                                input_ids=req_data["input_ids"],
                                max_tokens=req_data["max_tokens"],
                            ))
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
                    
                    input_ids, _, _ = prepare_batch_inputs(batch, device)
                    batch_size = batch.batch_size
                    max_seq_len = input_ids.shape[1]
                    max_tokens = max(r.max_tokens for r in batch.requests)
                    
                    buffers.metadata[0] = batch_size
                    buffers.metadata[1] = max_seq_len
                    buffers.metadata[2] = max_tokens
                    dist.broadcast(buffers.metadata, src=0)
                    dist.broadcast(input_ids, src=0)
                
                else:
                    # Other TP ranks in first PP stage
                    dist.broadcast(buffers.work_signal, src=0)
                    if buffers.work_signal.item() == 0:
                        time.sleep(0.001)
                        continue
                    
                    dist.broadcast(buffers.metadata, src=0)
                    batch_size = buffers.metadata[0].item()
                    max_seq_len = buffers.metadata[1].item()
                    max_tokens = buffers.metadata[2].item()
                    
                    input_ids = torch.empty(batch_size, max_seq_len, dtype=torch.long, device=device)
                    dist.broadcast(input_ids, src=0)
                    batch = None
                
                kv_caches = [[] for _ in range(batch_size)]
                
                with torch.inference_mode():
                    # Prefill
                    hidden = model(input_ids, kv_caches, None)
                    
                    # Only TP master sends to next PP stage
                    if is_tp_master:
                        next_pp_rank = ctx.get_pp_next_rank()
                        dist.send(hidden.contiguous(), dst=next_pp_rank)
                    
                    # Wait for tokens from last PP stage
                    recv_tokens = buffers.get_tokens(batch_size)
                    if is_tp_master:
                        last_pp_tp0 = (parallel_config.pp_size - 1) * parallel_config.tp_size
                        dist.recv(recv_tokens, src=last_pp_tp0)
                    
                    # Broadcast tokens within TP group
                    dist.broadcast(recv_tokens, src=ctx.pp_rank * parallel_config.tp_size, group=ctx.tp_group)
                    
                    if is_tp_master:
                        for i, req in enumerate(batch.requests):
                            token_id = recv_tokens[i].item()
                            response_queue.put((req.request_id, token_id))
                            req.generated_tokens = 1
                            if token_id in EOS_TOKEN_IDS:
                                req.finished = True
                                response_queue.put((req.request_id, None))
                    
                    curr_tokens = recv_tokens.clone()
                    
                    # Decode
                    for step in range(max_tokens - 1):
                        all_done = batch.all_finished if is_tp_master else False
                        done_t = torch.tensor([1 if all_done else 0], device=device)
                        dist.broadcast(done_t, src=0)
                        if done_t.item() == 1:
                            break
                        
                        decode_input = curr_tokens.unsqueeze(1)
                        hidden = model(decode_input, kv_caches, None)
                        
                        if is_tp_master:
                            dist.send(hidden.contiguous(), dst=ctx.get_pp_next_rank())
                        
                        dist.recv(recv_tokens, src=last_pp_tp0 if is_tp_master else ctx.pp_rank * parallel_config.tp_size)
                        if not is_tp_master:
                            dist.broadcast(recv_tokens, src=ctx.pp_rank * parallel_config.tp_size, group=ctx.tp_group)
                        
                        if is_tp_master:
                            for i, req in enumerate(batch.requests):
                                if req.finished:
                                    continue
                                token_id = recv_tokens[i].item()
                                response_queue.put((req.request_id, token_id))
                                req.generated_tokens += 1
                                if token_id in EOS_TOKEN_IDS or req.generated_tokens >= req.max_tokens:
                                    req.finished = True
                                    response_queue.put((req.request_id, None))
                        
                        curr_tokens = recv_tokens.clone()
                    
                    if is_tp_master:
                        for req in batch.requests:
                            if not req.finished:
                                response_queue.put((req.request_id, None))
            
            # === MIDDLE/LAST PP STAGES ===
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
                prev_pp_tp0 = (ctx.pp_rank - 1) * parallel_config.tp_size
                first_pp_tp0 = 0
                
                with torch.inference_mode():
                    # Prefill - receive from previous PP stage
                    prefill_buf = buffers.get_prefill_buffer(batch_size, max_seq_len)
                    if is_tp_master:
                        dist.recv(prefill_buf, src=prev_pp_tp0)
                    dist.broadcast(prefill_buf, src=ctx.pp_rank * parallel_config.tp_size, group=ctx.tp_group)
                    
                    output = model(prefill_buf, kv_caches, None)
                    
                    if is_last_pp:
                        tokens = output[:, -1, :].argmax(dim=-1)
                        if is_tp_master:
                            dist.send(tokens, dst=first_pp_tp0)
                    else:
                        if is_tp_master:
                            dist.send(output.contiguous(), dst=ctx.get_pp_next_rank())
                    
                    # Decode loop
                    for step in range(max_tokens - 1):
                        done_t = torch.tensor([0], device=device)
                        dist.broadcast(done_t, src=0)
                        if done_t.item() == 1:
                            break
                        
                        decode_buf = buffers.get_decode_buffer(batch_size)
                        if is_tp_master:
                            dist.recv(decode_buf, src=prev_pp_tp0)
                        dist.broadcast(decode_buf, src=ctx.pp_rank * parallel_config.tp_size, group=ctx.tp_group)
                        
                        output = model(decode_buf, kv_caches, None)
                        
                        if is_last_pp:
                            tokens = output[:, -1, :].argmax(dim=-1)
                            if is_tp_master:
                                dist.send(tokens, dst=first_pp_tp0)
                        else:
                            if is_tp_master:
                                dist.send(output.contiguous(), dst=ctx.get_pp_next_rank())
    
    except Exception as e:
        print(f"[Hybrid {rank}] ERROR: {e}")
        traceback.print_exc()
        raise

def run_worker(
    rank: int,
    world_size: int,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    input_queue,
    response_queue,
    bench_queue,
    max_batch_size: int = 4,
):
    """Dispatch to appropriate worker based on parallelism mode."""
    if parallel_config.mode == ParallelMode.TENSOR:
        run_tp_worker(rank, world_size, model_config, parallel_config,
                     input_queue, response_queue, bench_queue, max_batch_size)
    elif parallel_config.mode == ParallelMode.PIPELINE:
        run_pp_worker(rank, world_size, model_config, parallel_config,
                     input_queue, response_queue, bench_queue, max_batch_size)
    elif parallel_config.mode == ParallelMode.HYBRID:
        run_hybrid_worker(rank, world_size, model_config, parallel_config,
                         input_queue, response_queue, bench_queue, max_batch_size)
    else:
        raise ValueError(f"Unknown mode: {parallel_config.mode}")
