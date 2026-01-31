"""
Benchmarking and profiling utilities.
For Tracking token speed, latency, memory, and kernel timings.
"""

import time
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from contextlib import contextmanager


@dataclass
class TokenMetrics:
    """Metrics for a single generation."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    
    # Timing (seconds)
    prefill_time: float = 0.0
    decode_times: List[float] = field(default_factory=list)
    total_time: float = 0.0
    
    # Memory (bytes)
    peak_memory: int = 0
    
    @property
    def time_to_first_token(self) -> float:
        """TTFT - Time from request to first token."""
        return self.prefill_time
    
    @property
    def decode_time(self) -> float:
        """Total decode time."""
        return sum(self.decode_times)
    
    @property
    def avg_decode_time(self) -> float:
        """Average time per decode step."""
        if not self.decode_times:
            return 0.0
        return sum(self.decode_times) / len(self.decode_times)
    
    @property
    def tokens_per_second(self) -> float:
        """Generation throughput (excluding prefill)."""
        if self.decode_time == 0:
            return 0.0
        return self.generated_tokens / self.decode_time
    
    @property
    def total_tokens_per_second(self) -> float:
        """Overall throughput including prefill."""
        if self.total_time == 0:
            return 0.0
        return self.generated_tokens / self.total_time
    
    def to_dict(self) -> Dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "prefill_time_ms": self.prefill_time * 1000,
            "decode_time_ms": self.decode_time * 1000,
            "total_time_ms": self.total_time * 1000,
            "ttft_ms": self.time_to_first_token * 1000,
            "avg_token_latency_ms": self.avg_decode_time * 1000,
            "tokens_per_second": self.tokens_per_second,
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
        }
    
    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"BENCHMARK RESULTS\n"
            f"{'='*50}\n"
            f"  Prompt tokens:      {self.prompt_tokens}\n"
            f"  Generated tokens:   {self.generated_tokens}\n"
            f"{'─'*50}\n"
            f"LATENCY\n"
            f"  TTFT (prefill):     {self.prefill_time*1000:.2f} ms\n"
            f"  Avg token latency:  {self.avg_decode_time*1000:.2f} ms\n"
            f"  Total time:         {self.total_time*1000:.2f} ms\n"
            f"{'─'*50}\n"
            f"THROUGHPUT\n"
            f"  Decode speed:       {self.tokens_per_second:.2f} tok/s\n"
            f"  Overall speed:      {self.total_tokens_per_second:.2f} tok/s\n"
            f"{'─'*50}\n"
            f"MEMORY\n"
            f"  Peak GPU memory:    {self.peak_memory / (1024**2):.2f} MB\n"
            f"{'='*50}\n"
        )


class Benchmark:
    """
    Benchmarking context manager for generation.
    
    Usage:
        bench = Benchmark(device=0)
        with bench.track_prefill():
            # prefill code
        for token in tokens:
            with bench.track_decode():
                # decode code
        print(bench.metrics)
    """
    
    def __init__(self, device: int = 0, enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self.metrics = TokenMetrics()
        self._start_time: Optional[float] = None
        
    def start(self, prompt_tokens: int = 0):
        """Start benchmarking a generation."""
        self.metrics = TokenMetrics(prompt_tokens=prompt_tokens)
        self._start_time = time.perf_counter()
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def stop(self):
        """Stop benchmarking and finalize metrics."""
        if self._start_time:
            self.metrics.total_time = time.perf_counter() - self._start_time
        if self.enabled and torch.cuda.is_available():
            self.metrics.peak_memory = torch.cuda.max_memory_allocated(self.device)
    
    @contextmanager
    def track_prefill(self):
        """Track prefill time."""
        if not self.enabled:
            yield
            return
        
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        yield
        torch.cuda.synchronize(self.device)
        self.metrics.prefill_time = time.perf_counter() - start
    
    @contextmanager
    def track_decode(self):
        """Track single decode step time."""
        if not self.enabled:
            yield
            return
        
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        yield
        torch.cuda.synchronize(self.device)
        self.metrics.decode_times.append(time.perf_counter() - start)
        self.metrics.generated_tokens += 1
    
    def add_generated_token(self):
        """Manually increment token count (if not using track_decode)."""
        self.metrics.generated_tokens += 1


class CUDATimer:
    """
    Precise CUDA kernel timing using CUDA events.
    More accurate than CPU timing for GPU operations.
    
    Usage:
        timer = CUDATimer()
        timer.start()
        # GPU operations
        elapsed = timer.stop()  # Returns milliseconds
    """
    
    def __init__(self, device: int = 0):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        """Record start event."""
        self.start_event.record(torch.cuda.current_stream(self.device))
    
    def stop(self) -> float:
        """Record end event and return elapsed time in milliseconds."""
        self.end_event.record(torch.cuda.current_stream(self.device))
        torch.cuda.synchronize(self.device)
        return self.start_event.elapsed_time(self.end_event)


@contextmanager
def cuda_timer(device: int = 0):
    """
    Context manager for CUDA timing.
    
    Usage:
        with cuda_timer() as t:
            # GPU operations
        print(f"Elapsed: {t.elapsed_ms:.2f} ms")
    """
    class TimerResult:
        elapsed_ms: float = 0.0
    
    result = TimerResult()
    timer = CUDATimer(device)
    timer.start()
    yield result
    result.elapsed_ms = timer.stop()