"""
FastAPI server with OpenAI-compatible API.
"""
import asyncio
import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer

from .config import ModelConfig, format_chat

app = FastAPI(
    title="NanoSLG",
    description="Minimal Pipeline-Parallel LLM Inference Server",
    version="0.1.0",
)

_tokenizer = None
_request_queue = None
_response_queue = None
_bench_queue = None
_model_config: Optional[ModelConfig] = None


def init_server(tokenizer, req_q, res_q, bench_q, config: ModelConfig):
    """Initialize server state (called from main)."""
    global _tokenizer, _request_queue, _response_queue, _bench_queue, _model_config
    _tokenizer = tokenizer
    _request_queue = req_q
    _response_queue = res_q
    _bench_queue = bench_q
    _model_config = config


# LAME ENDPOINTS
@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [{
            "id": _model_config.name,
            "object": "model",
            "owned_by": "nanoslg",
        }]
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
    
    _request_queue.put({
        "input_ids": input_ids,
        "max_tokens": max_tokens,
    })
    
    if stream:
        return StreamingResponse(
            _stream_response(),
            media_type="text/event-stream",
        )
    else:
        return await _collect_response(max_tokens)


async def _stream_response():
    """Stream tokens as SSE events."""
    request_id = str(uuid.uuid4())
    
    while True:
        while _response_queue.empty():
            await asyncio.sleep(0.005)
        
        token_id = _response_queue.get()
        
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
        
        # Decode token
        text = _tokenizer.decode([token_id])
        
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _model_config.name,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"


async def _collect_response(max_tokens: int):
    """Collect all tokens for non-streaming response."""
    tokens = []
    
    while True:
        while _response_queue.empty():
            await asyncio.sleep(0.005)
        
        token_id = _response_queue.get()
        if token_id is None:
            break
        tokens.append(token_id)
    
    text = _tokenizer.decode(tokens)
    
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_config.name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,  # TODO: tracking
            "completion_tokens": len(tokens),
            "total_tokens": len(tokens),
        },
    }


@app.get("/v1/benchmark/last")
async def get_last_benchmark():
    """Get benchmark results from last generation."""
    if _bench_queue and not _bench_queue.empty():
        return _bench_queue.get()
    return {"message": "No benchmark data available"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": _model_config.name if _model_config else None}
