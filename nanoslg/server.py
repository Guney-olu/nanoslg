"""
V0.3
FastAPI server with batching support.
"""
import asyncio
import json
import time
import uuid
from typing import Optional, Dict
from collections import defaultdict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer

from .config import ModelConfig, format_chat

app = FastAPI(title="NanoSLG", version="0.3.0")

_tokenizer = None
_request_queue = None  # mp.Queue - to send requests to worker
_response_queue = None  # mp.Queue - to receive (request_id, token) pairs from worker
_bench_queue = None
_model_config: Optional[ModelConfig] = None

_response_handlers: Dict[str, asyncio.Queue] = {}
_response_router_task = None


def init_server(tokenizer, req_q, res_q, bench_q, config: ModelConfig):
    global _tokenizer, _request_queue, _response_queue, _bench_queue, _model_config
    _tokenizer = tokenizer
    _request_queue = req_q
    _response_queue = res_q
    _bench_queue = bench_q
    _model_config = config


async def response_router():
    """Background task that routes responses from worker to request handlers."""
    global _response_handlers
    
    while True:
        # Check for responses from worker (non-blocking)
        try:
            while not _response_queue.empty():
                request_id, token_id = _response_queue.get_nowait()
                
                if request_id in _response_handlers:
                    await _response_handlers[request_id].put(token_id)
                else:
                    print(f"[Router] Warning: No handler for request {request_id}")
        except Exception as e:
            print(f"[Router] Error: {e}")
        
        await asyncio.sleep(0.001)


@app.on_event("startup")
async def startup_event():
    """Start the response router background task."""
    global _response_router_task
    _response_router_task = asyncio.create_task(response_router())


@app.on_event("shutdown")
async def shutdown_event():
    """Cancel the response router."""
    global _response_router_task
    if _response_router_task:
        _response_router_task.cancel()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": _model_config.name, "object": "model", "owned_by": "nanoslg"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if _tokenizer is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    data = await request.json()
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 100)
    stream = data.get("stream", True)
    
    prompt = format_chat(messages, _model_config.chat_template)
    input_ids = _tokenizer.encode(prompt, return_tensors="pt")
    
    request_id = str(uuid.uuid4())
    response_queue = asyncio.Queue()
    _response_handlers[request_id] = response_queue
    
    try:
        _request_queue.put({
            "request_id": request_id,
            "input_ids": input_ids,
            "max_tokens": max_tokens,
        })
        
        if stream:
            return StreamingResponse(
                _stream_response(request_id, response_queue),
                media_type="text/event-stream",
            )
        else:
            return await _collect_response(request_id, response_queue, max_tokens)
    finally:
        # Cleanup will happen after streaming completes
        pass


async def _stream_response(request_id: str, response_queue: asyncio.Queue):
    """Stream tokens from async queue."""
    try:
        while True:
            # Wait for token with timeout
            try:
                token_id = await asyncio.wait_for(response_queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                print(f"[Stream] Timeout waiting for token, request {request_id}")
                break
            
            if token_id is None:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": _model_config.name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
                break
            
            text = _tokenizer.decode([token_id])
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": _model_config.name,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    finally:
        # Cleanup handler
        if request_id in _response_handlers:
            del _response_handlers[request_id]


async def _collect_response(request_id: str, response_queue: asyncio.Queue, max_tokens: int):
    """Collect all tokens for non-streaming response."""
    tokens = []
    
    try:
        while True:
            try:
                token_id = await asyncio.wait_for(response_queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                break
            
            if token_id is None:
                break
            tokens.append(token_id)
        
        text = _tokenizer.decode(tokens)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _model_config.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(tokens), "total_tokens": len(tokens)},
        }
    finally:
        if request_id in _response_handlers:
            del _response_handlers[request_id]


@app.get("/v1/benchmark/last")
async def get_last_benchmark():
    if _bench_queue and not _bench_queue.empty():
        return _bench_queue.get()
    return {"message": "No benchmark data available"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_config.name if _model_config else None}