"""
V0.5 — Dual-backend workers (FlashInfer on L4+, Contiguous SDPA on T4).
"""

import os
import time
import traceback
import torch
import torch.distributed as dist
from queue import Empty
from transformers import AutoConfig
import torch.nn.functional as F
from .config import ModelConfig, STORE_PATH
from .parallel import (
    ParallelMode, ParallelConfig, ParallelContext,
    get_layers_for_pp_rank, pp_send, pp_recv,
)
from .models import create_model, load_weights
from .scheduler import Request, Batch, BatchScheduler, prepare_batch_inputs
from .kv_cache import create_cache_manager

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
    def __init__(self, max_batch_size, hidden_size, max_seq_len, dtype, device):
        self.device = device
        self.max_batch_size = max_batch_size
        capped = min(max_seq_len, 2048)

        self.decode_hidden = torch.empty(
            max_batch_size, 1, hidden_size, dtype=dtype, device=device)
        self.prefill_hidden = torch.empty(
            max_batch_size, capped, hidden_size, dtype=dtype, device=device)
        self.tokens = torch.empty(
            max_batch_size, dtype=torch.long, device=device)
        self.work_signal = torch.empty(1, dtype=torch.int32, device=device)
        self.done_signal = torch.empty(1, dtype=torch.int32, device=device)
        self.metadata = torch.empty(4, dtype=torch.long, device=device)
        self.prompt_lens = torch.empty(
            max_batch_size, dtype=torch.long, device=device)

    def get_prefill_buffer(self, bs, sl):
        if sl > self.prefill_hidden.shape[1]:
            self.prefill_hidden = torch.empty(
                self.max_batch_size, sl, self.prefill_hidden.shape[2],
                dtype=self.prefill_hidden.dtype, device=self.device)
        return self.prefill_hidden[:bs, :sl, :].contiguous()

    def get_decode_buffer(self, bs):
        return self.decode_hidden[:bs, :, :].contiguous()

    def get_tokens(self, bs):
        return self.tokens[:bs].contiguous()



# def compile_submodules(model, use_compile=True):
#     """
#     Compile compute-heavy submodules individually.
#     Attention .forward() is never compiled (cache ops aren't compilable).
#     """
#     if not use_compile:
#         print("[Worker] Skipping torch.compile (FlashInfer handles hot path)")
#         return

#     print("[Worker] Compiling submodules with torch.compile…")
#     opts = dict(dynamic=True, mode="default")

#     if hasattr(model, "embed_tokens"):
#         model.embed_tokens = torch.compile(model.embed_tokens, **opts)
#     if hasattr(model, "norm"):
#         model.norm = torch.compile(model.norm, **opts)

#     for layer in model.layers:
#         layer.input_layernorm = torch.compile(layer.input_layernorm, **opts)
#         layer.post_attention_layernorm = torch.compile(
#             layer.post_attention_layernorm, **opts)

#         attn = layer.self_attn
#         attn.q_proj = torch.compile(attn.q_proj, **opts)
#         attn.k_proj = torch.compile(attn.k_proj, **opts)
#         attn.v_proj = torch.compile(attn.v_proj, **opts)
#         attn.o_proj = torch.compile(attn.o_proj, **opts)

#         layer.mlp = torch.compile(layer.mlp, **opts)

def compile_submodules(model, use_compile=True):
    """Skip torch.compile."""
    # BC perf gain from compile is <5% — not worth the instability.
    return


def warmup_model(model, device, hf_config, dtype,
                 parallel_config, max_batch, cache_manager):
    ctx = ParallelContext.get()
    is_first = ctx.is_first_pp_stage if parallel_config.pp_size > 1 else True

    with torch.inference_mode():
        sid = "warmup_0"
        dummy = list(range(16))

        # — prefill —
        cache_manager.allocate_sequence(
            sid, dummy if is_first else None, num_tokens=16)
        cache_ctx = cache_manager.begin_forward([sid], [16], [0])
        
        if is_first:
            x = torch.randint(0, 100, (1, 16), device=device)
        else:
            x = torch.randn(1, 16, hf_config.hidden_size,
                            dtype=dtype, device=device)

        print(f"[Warmup Debug R{ctx.rank}] x: shape={x.shape} dtype={x.dtype} device={x.device}")
        print(f"[Warmup Debug R{ctx.rank}] GPU mem: "
              f"{torch.cuda.memory_allocated(device)/1e9:.2f}GB allocated, "
              f"{torch.cuda.memory_reserved(device)/1e9:.2f}GB reserved, "
              f"{torch.cuda.mem_get_info(device)[0]/1e9:.2f}GB free")
        
        layer0 = model.layers[0]
        attn = layer0.self_attn
        print(f"[Warmup Debug R{ctx.rank}] q_proj weight: "
              f"shape={attn.q_proj.weight.shape} dtype={attn.q_proj.weight.dtype} "
              f"device={attn.q_proj.weight.device}")
        
        try:
            test_x = torch.randn(1, 16, 4096, dtype=dtype, device=device)
            test_w = torch.randn(2048, 4096, dtype=dtype, device=device)
            test_out = F.linear(test_x, test_w)
            print(f"[Warmup Debug R{ctx.rank}] ✓ Test F.linear works: {test_out.shape}")
            del test_x, test_w, test_out
        except Exception as e:
            print(f"[Warmup Debug R{ctx.rank}] ✗ Test F.linear FAILED: {e}")
        # ── END DEBUG ──

        _ = model(x, cache_ctx, None)
        cache_manager.end_forward(cache_ctx)

        # — decode —
        cache_manager.extend_sequence(sid, 1)
        cache_ctx = cache_manager.begin_forward([sid], [1])
        dx = (torch.randint(0, 100, (1, 1), device=device)
              if is_first
              else torch.randn(1, 1, hf_config.hidden_size,
                               dtype=dtype, device=device))
        _ = model(dx, cache_ctx, None)
        cache_manager.end_forward(cache_ctx)

        cache_manager.release_sequence(sid)
        del x, dx

    torch.cuda.empty_cache()
    if ctx.rank == 0:
        print(f"[Worker] Warmup complete  cache={cache_manager.stats}")


def allocate_batch_with_prefix(cache_manager, seq_ids, input_ids,
                               prompt_lens, device):
    """
    Allocate sequences, do radix prefix matching (if available),
    build trimmed model input.

    Args
        input_ids   : [bs, max_sl]  left-padded token IDs
        prompt_lens : list[int]     actual length per request

    Returns
        model_input      : [bs, max_new]  token IDs without cached prefix
        new_token_counts : list[int]
        input_offsets    : list[int]       left-pad offset in model_input
        prefix_lens      : list[int]
    """
    bs = len(seq_ids)
    max_sl = input_ids.shape[1]
    prefix_lens = []

    for i in range(bs):
        pl = prompt_lens[i]
        toks = input_ids[i, max_sl - pl:].tolist()
        plen = cache_manager.allocate_sequence(seq_ids[i], toks)
        # Always process ≥1 token so we can produce logits
        plen = min(plen, pl - 1) if pl > 1 else 0
        prefix_lens.append(plen)

    new_token_counts = [prompt_lens[i] - prefix_lens[i] for i in range(bs)]
    max_new = max(new_token_counts) if new_token_counts else 1

    model_input = torch.zeros(bs, max_new, dtype=torch.long, device=device)
    input_offsets = []

    for i in range(bs):
        ntok = new_token_counts[i]
        offset = max_new - ntok
        input_offsets.append(offset)
        pl = prompt_lens[i]
        pfl = prefix_lens[i]
        src = max_sl - pl + pfl
        if ntok > 0:
            model_input[i, offset:offset + ntok] = input_ids[i, src:src + ntok]

    return model_input, new_token_counts, input_offsets, prefix_lens


def allocate_batch_hidden(cache_manager, seq_ids, input_ids,
                          prompt_lens, max_sl, device):
    """
    Allocation for PP non-first stages.
    They have token_ids (broadcast) for radix matching but receive hidden states.
    Returns (new_token_counts, input_offsets, prefix_lens, max_new).
    """
    bs = len(seq_ids)
    prefix_lens = []

    for i in range(bs):
        pl = prompt_lens[i]
        toks = input_ids[i, max_sl - pl:].tolist()
        plen = cache_manager.allocate_sequence(seq_ids[i], toks)
        plen = min(plen, pl - 1) if pl > 1 else 0
        prefix_lens.append(plen)

    new_token_counts = [prompt_lens[i] - prefix_lens[i] for i in range(bs)]
    max_new = max(new_token_counts) if new_token_counts else 1
    input_offsets = [max_new - new_token_counts[i] for i in range(bs)]

    return new_token_counts, input_offsets, prefix_lens, max_new


def _broadcast_batch_meta(buffers, bs, max_sl, max_tok, prompt_lens, device):
    """Rank 0 packs and broadcasts batch metadata + prompt_lens."""
    buffers.metadata[0] = bs
    buffers.metadata[1] = max_sl
    buffers.metadata[2] = max_tok
    dist.broadcast(buffers.metadata, src=0)
    pl = torch.tensor(prompt_lens, dtype=torch.long, device=device)
    buffers.prompt_lens[:bs] = pl
    dist.broadcast(buffers.prompt_lens[:bs].contiguous(), src=0)


def _recv_batch_meta(buffers, device):
    """Non-rank-0 receives metadata + prompt_lens."""
    dist.broadcast(buffers.metadata, src=0)
    bs = buffers.metadata[0].item()
    max_sl = buffers.metadata[1].item()
    max_tok = buffers.metadata[2].item()
    pl_buf = buffers.prompt_lens[:bs].contiguous()
    dist.broadcast(pl_buf, src=0)
    return bs, max_sl, max_tok, pl_buf.tolist()


def _track_tokens(cache_manager, seq_ids, tokens_tensor, bs):
    """Append generated token IDs to sequence state for prefix tracking."""
    for i in range(bs):
        cache_manager.append_token_ids(seq_ids[i], [tokens_tensor[i].item()])


def run_tp_worker(rank, world_size, model_config, parallel_config,
                  input_queue, response_queue, bench_queue,
                  max_batch_size=4):
    try:
        setup_nccl_env()
        if rank == 0:
            if os.path.exists(STORE_PATH):
                os.remove(STORE_PATH)
            time.sleep(0.1)
        else:
            time.sleep(1.0)

        store = dist.FileStore(STORE_PATH, world_size)
        dist.init_process_group(
            backend="nccl", store=store, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = DTYPE_MAP[model_config.dtype]

        ctx = ParallelContext.initialize(parallel_config, rank)
        ctx.setup_groups()
        hf_config = AutoConfig.from_pretrained(model_config.path)

        model = create_model(hf_config, parallel_config).to(dtype).to(device)
        load_weights(model, model_config.path, hf_config, dtype, parallel_config)

        # Auto-selects FlashInfer (L4+) or Contiguous (T4)
        cache_mgr = create_cache_manager(
            hf_config, parallel_config, device, dtype, model_config,
            max_batch_size=max_batch_size)

        # Only torch.compile when NOT using FlashInfer (avoids CUBLAS clash)
        should_compile = cache_mgr.cfg.backend != "flashinfer"
        if rank == 0:
            print(f"[TP Worker {rank}] backend={cache_mgr.cfg.backend}  "
                  f"compile={'yes' if should_compile else 'skip (FlashInfer)'}")
        compile_submodules(model, use_compile=should_compile)

        warmup_model(model, device, hf_config, dtype,
                     parallel_config, max_batch_size, cache_mgr)

        buffers = WorkerBuffers(
            max_batch_size, hf_config.hidden_size,
            model_config.max_seq_len, dtype, device)
        scheduler = (BatchScheduler(max_batch_size=max_batch_size)
                     if rank == 0 else None)

        dist.barrier()
        if rank == 0:
            print(f"[TP Worker] {world_size} GPUs ready  "
                  f"backend={cache_mgr.cfg.backend}")

        while True:
            # — scheduling (rank 0) —
            if rank == 0:
                while True:
                    try:
                        rd = input_queue.get_nowait()
                        scheduler.add_request(
                            Request(rd["request_id"],
                                    rd["input_ids"], rd["max_tokens"]))
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
                bs = batch.batch_size
                max_sl = input_ids.shape[1]
                max_tok = max(r.max_tokens for r in batch.requests)

                _broadcast_batch_meta(
                    buffers, bs, max_sl, max_tok, prompt_lens, device)
                dist.broadcast(input_ids, src=0)

            else:
                dist.broadcast(buffers.work_signal, src=0)
                if buffers.work_signal.item() == 0:
                    time.sleep(0.001)
                    continue

                bs, max_sl, max_tok, prompt_lens = _recv_batch_meta(
                    buffers, device)
                input_ids = torch.empty(
                    bs, max_sl, dtype=torch.long, device=device)
                dist.broadcast(input_ids, src=0)
                batch = None

            # — all ranks: prefix-aware allocation —
            seq_ids = [f"s{i}" for i in range(bs)]
            model_input, new_counts, offsets, prefix_lens = \
                allocate_batch_with_prefix(
                    cache_mgr, seq_ids, input_ids, prompt_lens, device)

            torch.cuda.synchronize(device)
            t_start = time.perf_counter()

            with torch.inference_mode():
                # — prefill —
                cache_ctx = cache_mgr.begin_forward(
                    seq_ids, new_counts, offsets)
                logits = model(model_input, cache_ctx, None)
                cache_mgr.end_forward(cache_ctx)

                tokens = logits[:, -1, :].argmax(dim=-1)
                torch.cuda.synchronize(device)
                t_first = time.perf_counter()

                _track_tokens(cache_mgr, seq_ids, tokens, bs)

                if rank == 0:
                    for i, req in enumerate(batch.requests):
                        tid = tokens[i].item()
                        response_queue.put((req.request_id, tid))
                        req.generated_tokens = 1
                        if tid in EOS_TOKEN_IDS:
                            req.finished = True
                            response_queue.put((req.request_id, None))

                curr = tokens.clone()

                # — decode —
                for step in range(max_tok - 1):
                    if rank == 0:
                        buffers.done_signal.fill_(
                            1 if batch.all_finished else 0)
                    dist.broadcast(buffers.done_signal, src=0)
                    if buffers.done_signal.item():
                        break

                    cache_ctx = cache_mgr.begin_forward(
                        seq_ids, [1] * bs)
                    logits = model(curr.unsqueeze(1), cache_ctx, None)
                    cache_mgr.end_forward(cache_ctx)

                    tokens = logits[:, -1, :].argmax(dim=-1)
                    _track_tokens(cache_mgr, seq_ids, tokens, bs)

                    if rank == 0:
                        for i, req in enumerate(batch.requests):
                            if req.finished:
                                continue
                            tid = tokens[i].item()
                            response_queue.put((req.request_id, tid))
                            req.generated_tokens += 1
                            if (tid in EOS_TOKEN_IDS
                                    or req.generated_tokens >= req.max_tokens):
                                req.finished = True
                                response_queue.put((req.request_id, None))

                    curr = tokens.clone()

                if rank == 0:
                    for req in batch.requests:
                        if not req.finished:
                            response_queue.put((req.request_id, None))

            # — cleanup —
            for sid in seq_ids:
                cache_mgr.release_sequence(sid)

            torch.cuda.synchronize(device)
            t_end = time.perf_counter()

            if rank == 0:
                gen = sum(r.generated_tokens for r in batch.requests)
                ms = (t_end - t_start) * 1000
                tps = gen / (ms / 1000) if ms > 0 else 0
                pfx = sum(prefix_lens)
                print(f"\n[TP Batch] size={bs} tokens={gen} "
                      f"TTFT={(t_first - t_start) * 1000:.1f}ms "
                      f"{tps:.1f} tok/s  prefix_skip={pfx}  "
                      f"cache={cache_mgr.stats}")
                if bench_queue:
                    bench_queue.put({
                        "batch_size": bs,
                        "tokens_per_second": tps})

    except Exception as e:
        print(f"[TP Worker {rank}] ERROR: {e}")
        traceback.print_exc()
        raise

def run_pp_worker(rank, world_size, model_config, parallel_config,
                  input_queue, response_queue, bench_queue,
                  max_batch_size=4):
    try:
        setup_nccl_env()
        if rank == 0:
            if os.path.exists(STORE_PATH):
                os.remove(STORE_PATH)
            time.sleep(0.1)
        else:
            time.sleep(1.0)

        store = dist.FileStore(STORE_PATH, world_size)
        dist.init_process_group(
            backend="nccl", store=store, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = DTYPE_MAP[model_config.dtype]

        ctx = ParallelContext.initialize(parallel_config, rank)
        ctx.setup_groups()
        hf_config = AutoConfig.from_pretrained(model_config.path)

        model = create_model(hf_config, parallel_config).to(dtype).to(device)
        load_weights(model, model_config.path, hf_config, dtype, parallel_config)

        cache_mgr = create_cache_manager(
            hf_config, parallel_config, device, dtype, model_config,
            max_batch_size=max_batch_size)

        is_first = ctx.is_first_pp_stage
        is_last = ctx.is_last_pp_stage

        should_compile = cache_mgr.cfg.backend != "flashinfer"
        if rank == 0:
            print(f"[PP Worker {rank}] backend={cache_mgr.cfg.backend}  "
                  f"compile={'yes' if should_compile else 'skip (FlashInfer)'}")
        compile_submodules(model, use_compile=should_compile)

        warmup_model(model, device, hf_config, dtype,
                     parallel_config, max_batch_size, cache_mgr)

        buffers = WorkerBuffers(
            max_batch_size, hf_config.hidden_size,
            model_config.max_seq_len, dtype, device)
        scheduler = (BatchScheduler(max_batch_size=max_batch_size)
                     if rank == 0 else None)

        dist.barrier()
        if rank == 0:
            print(f"[PP Worker] {world_size} GPUs ready  "
                  f"backend={cache_mgr.cfg.backend}")

        while True:
            # ═══ RANK 0 (first PP stage) ═══
            if rank == 0:
                while True:
                    try:
                        rd = input_queue.get_nowait()
                        scheduler.add_request(
                            Request(rd["request_id"],
                                    rd["input_ids"], rd["max_tokens"]))
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
                bs = batch.batch_size
                max_sl = input_ids.shape[1]
                max_tok = max(r.max_tokens for r in batch.requests)

                _broadcast_batch_meta(
                    buffers, bs, max_sl, max_tok, prompt_lens, device)
                dist.broadcast(input_ids, src=0)

                seq_ids = [req.request_id for req in batch.requests]
                model_input, new_counts, offsets, prefix_lens = \
                    allocate_batch_with_prefix(
                        cache_mgr, seq_ids, input_ids, prompt_lens, device)
                max_new = model_input.shape[1]

                buffers.metadata[3] = max_new
                dist.broadcast(buffers.metadata[3:4].contiguous(), src=0)

                torch.cuda.synchronize(device)
                t_start = time.perf_counter()

                with torch.inference_mode():
                    # — prefill —
                    cache_ctx = cache_mgr.begin_forward(
                        seq_ids, new_counts, offsets)
                    hidden = model(model_input, cache_ctx, None)
                    cache_mgr.end_forward(cache_ctx)
                    dist.send(hidden.contiguous(), dst=1)

                    recv_tok = buffers.get_tokens(bs)
                    dist.recv(recv_tok, src=world_size - 1)

                    torch.cuda.synchronize(device)
                    t_first = time.perf_counter()

                    _track_tokens(cache_mgr, seq_ids, recv_tok, bs)
                    for i, req in enumerate(batch.requests):
                        tid = recv_tok[i].item()
                        response_queue.put((req.request_id, tid))
                        req.generated_tokens = 1
                        if tid in EOS_TOKEN_IDS:
                            req.finished = True
                            response_queue.put((req.request_id, None))

                    curr = recv_tok.clone()

                    # — decode —
                    for step in range(max_tok - 1):
                        buffers.done_signal.fill_(
                            1 if batch.all_finished else 0)
                        dist.broadcast(buffers.done_signal, src=0)
                        if batch.all_finished:
                            break

                        cache_ctx = cache_mgr.begin_forward(
                            seq_ids, [1] * bs)
                        hidden = model(curr.unsqueeze(1), cache_ctx, None)
                        cache_mgr.end_forward(cache_ctx)
                        dist.send(hidden.contiguous(), dst=1)
                        dist.recv(recv_tok, src=world_size - 1)

                        _track_tokens(cache_mgr, seq_ids, recv_tok, bs)
                        for i, req in enumerate(batch.requests):
                            if req.finished:
                                continue
                            tid = recv_tok[i].item()
                            response_queue.put((req.request_id, tid))
                            req.generated_tokens += 1
                            if (tid in EOS_TOKEN_IDS
                                    or req.generated_tokens >= req.max_tokens):
                                req.finished = True
                                response_queue.put((req.request_id, None))

                        curr = recv_tok.clone()

                    for req in batch.requests:
                        if not req.finished:
                            response_queue.put((req.request_id, None))

                for sid in seq_ids:
                    cache_mgr.release_sequence(sid)

                torch.cuda.synchronize(device)
                t_end = time.perf_counter()
                gen = sum(r.generated_tokens for r in batch.requests)
                ms = (t_end - t_start) * 1000
                tps = gen / (ms / 1000) if ms > 0 else 0
                print(f"\n[PP Batch] size={bs} tokens={gen} "
                      f"TTFT={(t_first - t_start) * 1000:.1f}ms "
                      f"{tps:.1f} tok/s  prefix_skip={sum(prefix_lens)}  "
                      f"cache={cache_mgr.stats}")

            else:
                dist.broadcast(buffers.work_signal, src=0)
                if buffers.work_signal.item() == 0:
                    time.sleep(0.001)
                    continue

                bs, max_sl, max_tok, prompt_lens = _recv_batch_meta(
                    buffers, device)

                input_ids = torch.empty(
                    bs, max_sl, dtype=torch.long, device=device)
                dist.broadcast(input_ids, src=0)

                mn_buf = buffers.metadata[3:4].contiguous()
                dist.broadcast(mn_buf, src=0)
                max_new_r0 = mn_buf.item()

                seq_ids = [f"s{i}" for i in range(bs)]
                new_counts, offsets, prefix_lens, max_new = \
                    allocate_batch_hidden(
                        cache_mgr, seq_ids, input_ids,
                        prompt_lens, max_sl, device)

                max_new = max_new_r0

                with torch.inference_mode():
                    # — prefill —
                    pbuf = buffers.get_prefill_buffer(bs, max_new)
                    dist.recv(pbuf, src=rank - 1)

                    cache_ctx = cache_mgr.begin_forward(
                        seq_ids, new_counts, offsets)
                    output = model(pbuf, cache_ctx, None)
                    cache_mgr.end_forward(cache_ctx)

                    if is_last:
                        tokens = output[:, -1, :].argmax(dim=-1)
                        dist.send(tokens, dst=0)
                    else:
                        dist.send(output.contiguous(), dst=rank + 1)

                    # — decode —
                    for step in range(max_tok - 1):
                        dist.broadcast(buffers.done_signal, src=0)
                        if buffers.done_signal.item():
                            break

                        dbuf = buffers.get_decode_buffer(bs)
                        dist.recv(dbuf, src=rank - 1)

                        cache_ctx = cache_mgr.begin_forward(
                            seq_ids, [1] * bs)
                        output = model(dbuf, cache_ctx, None)
                        cache_mgr.end_forward(cache_ctx)

                        if is_last:
                            tokens = output[:, -1, :].argmax(dim=-1)
                            dist.send(tokens, dst=0)
                        else:
                            dist.send(output.contiguous(), dst=rank + 1)

                for sid in seq_ids:
                    cache_mgr.release_sequence(sid)

    except Exception as e:
        print(f"[PP Worker {rank}] ERROR: {e}")
        traceback.print_exc()
        raise


def run_hybrid_worker(rank, world_size, model_config, parallel_config,
                      input_queue, response_queue, bench_queue,
                      max_batch_size=4):
    try:
        setup_nccl_env()
        if rank == 0:
            if os.path.exists(STORE_PATH):
                os.remove(STORE_PATH)
            time.sleep(0.1)
        else:
            time.sleep(1.0)

        store = dist.FileStore(STORE_PATH, world_size)
        dist.init_process_group(
            backend="nccl", store=store, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = DTYPE_MAP[model_config.dtype]

        ctx = ParallelContext.initialize(parallel_config, rank)
        ctx.setup_groups()
        hf_config = AutoConfig.from_pretrained(model_config.path)

        model = create_model(hf_config, parallel_config).to(dtype).to(device)
        load_weights(model, model_config.path, hf_config, dtype, parallel_config)

        cache_mgr = create_cache_manager(
            hf_config, parallel_config, device, dtype, model_config,
            max_batch_size=max_batch_size)

        is_first_pp = ctx.is_first_pp_stage
        is_last_pp = ctx.is_last_pp_stage
        is_tp_master = ctx.is_tp_master
        tp_size = parallel_config.tp_size

        should_compile = cache_mgr.cfg.backend != "flashinfer"
        if rank == 0:
            print(f"[Hybrid {rank}] PP={ctx.pp_rank} TP={ctx.tp_rank}  "
                  f"backend={cache_mgr.cfg.backend}  "
                  f"compile={'yes' if should_compile else 'skip (FlashInfer)'}")
        compile_submodules(model, use_compile=should_compile)

        warmup_model(model, device, hf_config, dtype,
                     parallel_config, max_batch_size, cache_mgr)

        buffers = WorkerBuffers(
            max_batch_size, hf_config.hidden_size,
            model_config.max_seq_len, dtype, device)
        scheduler = (BatchScheduler(max_batch_size=max_batch_size)
                     if rank == 0 else None)

        dist.barrier()
        if rank == 0:
            print(f"[Hybrid] {world_size} GPUs ready "
                  f"TP={tp_size} PP={parallel_config.pp_size}  "
                  f"backend={cache_mgr.cfg.backend}")

        last_pp_tp0 = (parallel_config.pp_size - 1) * tp_size

        while True:
            if is_first_pp:
                if is_tp_master:
                    while True:
                        try:
                            rd = input_queue.get_nowait()
                            scheduler.add_request(
                                Request(rd["request_id"],
                                        rd["input_ids"], rd["max_tokens"]))
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

                    input_ids, _, prompt_lens = prepare_batch_inputs(
                        batch, device)
                    bs = batch.batch_size
                    max_sl = input_ids.shape[1]
                    max_tok = max(r.max_tokens for r in batch.requests)

                    _broadcast_batch_meta(
                        buffers, bs, max_sl, max_tok, prompt_lens, device)
                    dist.broadcast(input_ids, src=0)

                else:
                    dist.broadcast(buffers.work_signal, src=0)
                    if buffers.work_signal.item() == 0:
                        time.sleep(0.001)
                        continue

                    bs, max_sl, max_tok, prompt_lens = _recv_batch_meta(
                        buffers, device)
                    input_ids = torch.empty(
                        bs, max_sl, dtype=torch.long, device=device)
                    dist.broadcast(input_ids, src=0)
                    batch = None

                seq_ids = [f"s{i}" for i in range(bs)]
                model_input, new_counts, offsets, prefix_lens = \
                    allocate_batch_with_prefix(
                        cache_mgr, seq_ids, input_ids, prompt_lens, device)
                max_new = model_input.shape[1]

                if is_tp_master:
                    buffers.metadata[3] = max_new
                    dist.broadcast(
                        buffers.metadata[3:4].contiguous(), src=0)
                else:
                    mn = buffers.metadata[3:4].contiguous()
                    dist.broadcast(mn, src=0)

                torch.cuda.synchronize(device)
                t_start = time.perf_counter() if is_tp_master else 0

                with torch.inference_mode():
                    cache_ctx = cache_mgr.begin_forward(
                        seq_ids, new_counts, offsets)
                    hidden = model(model_input, cache_ctx, None)
                    cache_mgr.end_forward(cache_ctx)

                    if is_tp_master:
                        dist.send(hidden.contiguous(),
                                  dst=ctx.get_pp_next_rank())

                    recv_tok = buffers.get_tokens(bs)
                    if is_tp_master:
                        dist.recv(recv_tok, src=last_pp_tp0)
                    dist.broadcast(
                        recv_tok,
                        src=ctx.pp_rank * tp_size,
                        group=ctx.tp_group)

                    if is_tp_master:
                        torch.cuda.synchronize(device)
                        t_first = time.perf_counter()

                    _track_tokens(cache_mgr, seq_ids, recv_tok, bs)

                    if is_tp_master and batch:
                        for i, req in enumerate(batch.requests):
                            tid = recv_tok[i].item()
                            response_queue.put((req.request_id, tid))
                            req.generated_tokens = 1
                            if tid in EOS_TOKEN_IDS:
                                req.finished = True
                                response_queue.put((req.request_id, None))

                    curr = recv_tok.clone()

                    for step in range(max_tok - 1):
                        if is_tp_master:
                            done = (1 if (batch and batch.all_finished)
                                    else 0)
                            buffers.done_signal.fill_(done)
                        dist.broadcast(buffers.done_signal, src=0)
                        if buffers.done_signal.item():
                            break

                        cache_ctx = cache_mgr.begin_forward(
                            seq_ids, [1] * bs)
                        hidden = model(
                            curr.unsqueeze(1), cache_ctx, None)
                        cache_mgr.end_forward(cache_ctx)

                        if is_tp_master:
                            dist.send(hidden.contiguous(),
                                      dst=ctx.get_pp_next_rank())
                            dist.recv(recv_tok, src=last_pp_tp0)
                        dist.broadcast(
                            recv_tok,
                            src=ctx.pp_rank * tp_size,
                            group=ctx.tp_group)

                        _track_tokens(cache_mgr, seq_ids, recv_tok, bs)

                        if is_tp_master and batch:
                            for i, req in enumerate(batch.requests):
                                if req.finished:
                                    continue
                                tid = recv_tok[i].item()
                                response_queue.put((req.request_id, tid))
                                req.generated_tokens += 1
                                if (tid in EOS_TOKEN_IDS
                                        or req.generated_tokens
                                        >= req.max_tokens):
                                    req.finished = True
                                    response_queue.put(
                                        (req.request_id, None))

                        curr = recv_tok.clone()

                    if is_tp_master and batch:
                        for req in batch.requests:
                            if not req.finished:
                                response_queue.put((req.request_id, None))

                for sid in seq_ids:
                    cache_mgr.release_sequence(sid)

                if is_tp_master:
                    torch.cuda.synchronize(device)
                    t_end = time.perf_counter()
                    gen = sum(r.generated_tokens for r in batch.requests)
                    ms = (t_end - t_start) * 1000
                    tps = gen / (ms / 1000) if ms > 0 else 0
                    print(f"\n[Hybrid Batch] size={bs} tokens={gen} "
                          f"TTFT={(t_first - t_start) * 1000:.1f}ms "
                          f"{tps:.1f} tok/s  prefix_skip={sum(prefix_lens)}  "
                          f"cache={cache_mgr.stats}")

            else:
                dist.broadcast(buffers.work_signal, src=0)
                if buffers.work_signal.item() == 0:
                    time.sleep(0.001)
                    continue

                bs, max_sl, max_tok, prompt_lens = _recv_batch_meta(
                    buffers, device)
                input_ids = torch.empty(
                    bs, max_sl, dtype=torch.long, device=device)
                dist.broadcast(input_ids, src=0)

                mn_buf = buffers.metadata[3:4].contiguous()
                dist.broadcast(mn_buf, src=0)
                max_new_r0 = mn_buf.item()

                seq_ids = [f"s{i}" for i in range(bs)]
                new_counts, offsets, prefix_lens, _ = \
                    allocate_batch_hidden(
                        cache_mgr, seq_ids, input_ids,
                        prompt_lens, max_sl, device)
                max_new = max_new_r0

                prev_pp_tp0 = (ctx.pp_rank - 1) * tp_size
                first_pp_tp0 = 0

                with torch.inference_mode():
                    pbuf = buffers.get_prefill_buffer(bs, max_new)
                    if is_tp_master:
                        dist.recv(pbuf, src=prev_pp_tp0)
                    dist.broadcast(
                        pbuf,
                        src=ctx.pp_rank * tp_size,
                        group=ctx.tp_group)

                    cache_ctx = cache_mgr.begin_forward(
                        seq_ids, new_counts, offsets)
                    output = model(pbuf, cache_ctx, None)
                    cache_mgr.end_forward(cache_ctx)

                    if is_last_pp:
                        tokens = output[:, -1, :].argmax(dim=-1)
                        if is_tp_master:
                            dist.send(tokens, dst=first_pp_tp0)
                    else:
                        if is_tp_master:
                            dist.send(output.contiguous(),
                                      dst=ctx.get_pp_next_rank())

                    # — decode —
                    for step in range(max_tok - 1):
                        dist.broadcast(buffers.done_signal, src=0)
                        if buffers.done_signal.item():
                            break

                        dbuf = buffers.get_decode_buffer(bs)
                        if is_tp_master:
                            dist.recv(dbuf, src=prev_pp_tp0)
                        dist.broadcast(
                            dbuf,
                            src=ctx.pp_rank * tp_size,
                            group=ctx.tp_group)

                        cache_ctx = cache_mgr.begin_forward(
                            seq_ids, [1] * bs)
                        output = model(dbuf, cache_ctx, None)
                        cache_mgr.end_forward(cache_ctx)

                        if is_last_pp:
                            tokens = output[:, -1, :].argmax(dim=-1)
                            if is_tp_master:
                                dist.send(tokens, dst=first_pp_tp0)
                        else:
                            if is_tp_master:
                                dist.send(output.contiguous(),
                                          dst=ctx.get_pp_next_rank())

                for sid in seq_ids:
                    cache_mgr.release_sequence(sid)

    except Exception as e:
        print(f"[Hybrid {rank}] ERROR: {e}")
        traceback.print_exc()
        raise


def run_worker(rank, world_size, model_config, parallel_config,
               input_queue, response_queue, bench_queue,
               max_batch_size=4):
    if parallel_config.mode == ParallelMode.TENSOR:
        run_tp_worker(rank, world_size, model_config, parallel_config,
                      input_queue, response_queue, bench_queue,
                      max_batch_size)
    elif parallel_config.mode == ParallelMode.PIPELINE:
        run_pp_worker(rank, world_size, model_config, parallel_config,
                      input_queue, response_queue, bench_queue,
                      max_batch_size)
    elif parallel_config.mode == ParallelMode.HYBRID:
        run_hybrid_worker(rank, world_size, model_config, parallel_config,
                          input_queue, response_queue, bench_queue,
                          max_batch_size)
    else:
        raise ValueError(f"Unknown mode: {parallel_config.mode}")