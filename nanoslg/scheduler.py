"""
Batch scheduler for continuous batching.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from queue import Queue, Empty
import torch


@dataclass
class Request:
    """A single inference request."""
    request_id: str
    input_ids: torch.Tensor  # [1, seq_len]
    max_tokens: int
    
    # Runtime state
    generated_tokens: int = 0
    finished: bool = False
    finish_reason: str = None
    
    @property
    def prompt_len(self) -> int:
        return self.input_ids.shape[1]


@dataclass 
class Batch:
    """A batch of requests for processing."""
    requests: List[Request]
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None
    kv_caches: List[List[Dict]] = None
    positions: List[int] = None
    
    def __post_init__(self):
        if self.kv_caches is None:
            self.kv_caches = [[] for _ in self.requests]
        if self.positions is None:
            self.positions = [0] * len(self.requests)
    
    @property
    def batch_size(self) -> int:
        return len(self.requests)
    
    @property
    def all_finished(self) -> bool:
        return all(r.finished for r in self.requests)


class BatchScheduler:
    """Collects requests and forms batches."""
    
    def __init__(self, max_batch_size: int = 4, max_wait_time: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests: List[Request] = []
        self.lock = threading.Lock()
        self.last_batch_time = time.time()
    
    def add_request(self, request: Request):
        with self.lock:
            self.pending_requests.append(request)
    
    def try_form_batch(self) -> Optional[Batch]:
        with self.lock:
            if not self.pending_requests:
                return None
            
            current_time = time.time()
            time_since_last = current_time - self.last_batch_time
            
            should_batch = (
                len(self.pending_requests) >= self.max_batch_size or
                (len(self.pending_requests) > 0 and time_since_last >= self.max_wait_time)
            )
            
            if not should_batch:
                return None
            
            batch_requests = self.pending_requests[:self.max_batch_size]
            self.pending_requests = self.pending_requests[self.max_batch_size:]
            self.last_batch_time = current_time
            
            return Batch(requests=batch_requests)


def prepare_batch_inputs(batch: Batch, device: torch.device, pad_token_id: int = 0):
    """Prepare padded inputs for batched prefill."""
    prompt_lens = [r.prompt_len for r in batch.requests]
    max_len = max(prompt_lens)
    batch_size = len(batch.requests)
    
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    
    for i, req in enumerate(batch.requests):
        seq_len = req.prompt_len
        # Right-align (left-pad)
        input_ids[i, max_len - seq_len:] = req.input_ids[0]
        attention_mask[i, max_len - seq_len:] = True
    
    batch.input_ids = input_ids
    batch.attention_mask = attention_mask
    batch.positions = [max_len for _ in batch.requests]
    
    return input_ids, attention_mask, prompt_lens
