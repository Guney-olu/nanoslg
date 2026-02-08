"""
Parallelism infrastructure for Tensor and Pipeline parallelism.
Handles communication groups, primitives, and coordination.
"""

import os
import torch
import torch.distributed as dist
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from contextlib import contextmanager


class ParallelMode(Enum):
    """Supported parallelism modes."""
    PIPELINE = "pipeline"      # PP only - split layers across GPUs
    TENSOR = "tensor"          # TP only - split tensors across GPUs
    HYBRID = "hybrid"          # TP + PP combined
    DATA = "data"              # DP - replicate model


@dataclass
class ParallelConfig:
    """
    Configuration for distributed parallelism.
    
    For N GPUs:
    - PP only: pp_size=N, tp_size=1
    - TP only: tp_size=N, pp_size=1  
    - Hybrid: tp_size * pp_size = N (e.g., 2 TP x 2 PP = 4 GPUs)
    """
    mode: ParallelMode
    world_size: int
    tp_size: int = 1          # Tensor parallel degree
    pp_size: int = 1          # Pipeline parallel degree
    
    # Layer distribution for PP
    pp_layer_splits: Dict[int, List[int]] = None  # pp_rank -> layer indices
    
    def __post_init__(self):
        if self.mode == ParallelMode.PIPELINE:
            self.tp_size = 1
            self.pp_size = self.world_size
        elif self.mode == ParallelMode.TENSOR:
            self.tp_size = self.world_size
            self.pp_size = 1
        elif self.mode == ParallelMode.HYBRID:
            assert self.tp_size * self.pp_size == self.world_size, \
                f"TP({self.tp_size}) * PP({self.pp_size}) != world_size({self.world_size})"
    
    def get_tp_rank(self, global_rank: int) -> int:
        """Get tensor parallel rank within TP group."""
        return global_rank % self.tp_size
    
    def get_pp_rank(self, global_rank: int) -> int:
        """Get pipeline parallel rank within PP group."""
        return global_rank // self.tp_size
    
    def get_tp_group_ranks(self, global_rank: int) -> List[int]:
        """Get all ranks in same TP group."""
        pp_rank = self.get_pp_rank(global_rank)
        return [pp_rank * self.tp_size + i for i in range(self.tp_size)]
    
    def get_pp_group_ranks(self, global_rank: int) -> List[int]:
        """Get all ranks in same PP group."""
        tp_rank = self.get_tp_rank(global_rank)
        return [i * self.tp_size + tp_rank for i in range(self.pp_size)]


class ParallelContext:
    """
    Singleton managing parallel groups and communication.
    
    For hybrid parallelism with 4 GPUs (2 TP x 2 PP):
        GPU Layout:
            PP Stage 0: GPU 0, GPU 1 (TP group)
            PP Stage 1: GPU 2, GPU 3 (TP group)
        
        TP Groups: [0,1], [2,3]
        PP Groups: [0,2], [1,3]
    """
    _instance: Optional['ParallelContext'] = None
    
    def __init__(self, config: ParallelConfig, rank: int):
        self.config = config
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        
        self.tp_rank = config.get_tp_rank(rank)
        self.pp_rank = config.get_pp_rank(rank)
        
        self.tp_group: Optional[dist.ProcessGroup] = None
        self.pp_group: Optional[dist.ProcessGroup] = None
        
        self._initialized = False
    
    @classmethod
    def initialize(cls, config: ParallelConfig, rank: int) -> 'ParallelContext':
        """Initialize the parallel context singleton."""
        cls._instance = cls(config, rank)
        return cls._instance
    
    @classmethod
    def get(cls) -> Optional['ParallelContext']:
        """Get the parallel context singleton."""
        return cls._instance
    
    def setup_groups(self):
        """Create communication groups for TP and PP."""
        if self._initialized:
            return
        
        world_size = self.config.world_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        
        # Create TP groups (GPUs that share tensor shards)
        if tp_size > 1:
            for pp_rank in range(pp_size):
                ranks = [pp_rank * tp_size + i for i in range(tp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.tp_group = group
                    if self.rank == ranks[0]:
                        print(f"[Rank {self.rank}] TP group: {ranks}")
        
        # Create PP groups (GPUs that form a pipeline)
        if pp_size > 1:
            for tp_rank in range(tp_size):
                ranks = [i * tp_size + tp_rank for i in range(pp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.pp_group = group
                    if self.rank == ranks[0]:
                        print(f"[Rank {self.rank}] PP group: {ranks}")
        
        self._initialized = True
        print(f"[Rank {self.rank}] Context ready: tp_rank={self.tp_rank}, pp_rank={self.pp_rank}")
    
    @property
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0
    
    @property
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == self.config.pp_size - 1
    
    @property
    def is_tp_master(self) -> bool:
        """Is this the first rank in its TP group?"""
        return self.tp_rank == 0
    
    def get_pp_prev_rank(self) -> Optional[int]:
        """Get rank of previous pipeline stage (same TP position)."""
        if self.pp_rank == 0:
            return None
        return (self.pp_rank - 1) * self.config.tp_size + self.tp_rank
    
    def get_pp_next_rank(self) -> Optional[int]:
        """Get rank of next pipeline stage (same TP position)."""
        if self.pp_rank == self.config.pp_size - 1:
            return None
        return (self.pp_rank + 1) * self.config.tp_size + self.tp_rank


def all_reduce_tp(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce across tensor parallel group."""
    ctx = ParallelContext.get()
    if ctx and ctx.config.tp_size > 1 and ctx.tp_group is not None:
        dist.all_reduce(tensor, op=op, group=ctx.tp_group)
    return tensor


def all_gather_tp(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather across tensor parallel group."""
    ctx = ParallelContext.get()
    if ctx is None or ctx.config.tp_size == 1:
        return tensor
    
    tp_size = ctx.config.tp_size
    gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=ctx.tp_group)
    return torch.cat(gathered, dim=dim)


def reduce_scatter_tp(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reduce-scatter across tensor parallel group."""
    ctx = ParallelContext.get()
    if ctx is None or ctx.config.tp_size == 1:
        return tensor
    
    tp_size = ctx.config.tp_size
    # Split tensor along dimension
    chunks = tensor.chunk(tp_size, dim=dim)
    output = torch.empty_like(chunks[0])
    
    # Flatten chunks for reduce_scatter
    input_list = [chunk.contiguous() for chunk in chunks]
    dist.reduce_scatter(output, input_list, group=ctx.tp_group)
    
    return output


def broadcast_tp(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast within TP group from local rank."""
    ctx = ParallelContext.get()
    if ctx and ctx.config.tp_size > 1 and ctx.tp_group is not None:
        # Convert local src to global rank
        global_src = ctx.pp_rank * ctx.config.tp_size + src
        dist.broadcast(tensor, src=global_src, group=ctx.tp_group)
    return tensor


def pp_send(tensor: torch.Tensor, dst_pp_rank: int):
    """Send to next pipeline stage."""
    ctx = ParallelContext.get()
    dst = dst_pp_rank * ctx.config.tp_size + ctx.tp_rank
    dist.send(tensor.contiguous(), dst=dst)


def pp_recv(tensor: torch.Tensor, src_pp_rank: int):
    """Receive from previous pipeline stage."""
    ctx = ParallelContext.get()
    src = src_pp_rank * ctx.config.tp_size + ctx.tp_rank
    dist.recv(tensor, src=src)

#UTILITY
def get_parallel_context() -> ParallelContext:
    """Get parallel context, raise if not initialized."""
    ctx = ParallelContext.get()
    if ctx is None:
        raise RuntimeError("ParallelContext not initialized")
    return ctx


def split_tensor_for_tp(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Split tensor for current TP rank."""
    ctx = ParallelContext.get()
    if ctx is None or ctx.config.tp_size == 1:
        return tensor
    
    chunks = tensor.chunk(ctx.config.tp_size, dim=dim)
    return chunks[ctx.tp_rank].contiguous()


def get_layers_for_pp_rank(total_layers: int, pp_size: int, pp_rank: int) -> List[int]:
    """Get layer indices for a PP rank (roughly equal distribution)."""
    layers_per_stage = total_layers // pp_size
    remainder = total_layers % pp_size
    
    start = pp_rank * layers_per_stage + min(pp_rank, remainder)
    end = start + layers_per_stage + (1 if pp_rank < remainder else 0)
    
    return list(range(start, end))


@contextmanager
def cuda_sync_context(device: int = None):
    """Context manager that syncs CUDA at entry and exit."""
    if device is None:
        ctx = ParallelContext.get()
        device = ctx.rank if ctx else 0
    torch.cuda.synchronize(device)
    yield
    torch.cuda.synchronize(device)
