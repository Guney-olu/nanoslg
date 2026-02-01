"""
Tensor Parallel layer implementations.

Column Parallel: Output dimension is partitioned
    Y = XW^T where W is [out_features, in_features]
    Each GPU holds W_i of shape [out_features/tp_size, in_features]
    
Row Parallel: Input dimension is partitioned
    Y = XW^T where W is [out_features, in_features]  
    Each GPU holds W_i of shape [out_features, in_features/tp_size]
    Input must already be partitioned
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

from .parallel import (
    ParallelContext, 
    all_reduce_tp, 
    all_gather_tp,
    get_parallel_context,
)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.
    
    Weight shape: [out_features_per_partition, in_features]
    Output shape: [batch, seq, out_features_per_partition] (if not gathered)
                  [batch, seq, out_features] (if gathered)
    
    Use for: Q, K, V projections, gate_proj, up_proj
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        init_method=None,
    ):
        super().__init__()
        ctx = get_parallel_context()
        tp_size = ctx.config.tp_size
        
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.gather_output = gather_output
        
        assert out_features % tp_size == 0, \
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        
        self.out_features_per_partition = out_features // tp_size
        
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights(init_method)
    
    def _init_weights(self, init_method):
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, in_features]
        # out: [batch, seq, out_features_per_partition]
        output = F.linear(x, self.weight, self.bias)
        
        if self.gather_output and self.tp_size > 1:
            output = all_gather_tp(output, dim=-1)
        
        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.
    
    Weight shape: [out_features, in_features_per_partition]
    Input must be pre-split along last dimension.
    Output is all-reduced across TP group.
    
    Use for: O projection, down_proj
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        input_is_parallel: bool = True,
        init_method=None,
    ):
        super().__init__()
        ctx = get_parallel_context()
        tp_size = ctx.config.tp_size
        
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.input_is_parallel = input_is_parallel
        
        assert in_features % tp_size == 0, \
            f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        
        self.in_features_per_partition = in_features // tp_size
        
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        
        if bias:
            # Bias is not split - only add after all-reduce
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights(init_method)
    
    def _init_weights(self, init_method):
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, in_features_per_partition]
        output = F.linear(x, self.weight)  # [batch, seq, out_features]
        
        if self.tp_size > 1:
            output = all_reduce_tp(output)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class VocabParallelEmbedding(nn.Module):
    """
    Embedding with vocabulary parallelism.
    
    Vocabulary is split across TP ranks.
    Each rank holds vocab_size/tp_size embeddings.
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: Optional[int] = None):
        super().__init__()
        ctx = get_parallel_context()
        tp_size = ctx.config.tp_size
        tp_rank = ctx.tp_rank
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        assert vocab_size % tp_size == 0, \
            f"vocab_size ({vocab_size}) must be divisible by tp_size ({tp_size})"
        
        self.vocab_per_partition = vocab_size // tp_size
        self.vocab_start = tp_rank * self.vocab_per_partition
        self.vocab_end = self.vocab_start + self.vocab_per_partition
        
        self.embedding = nn.Embedding(
            self.vocab_per_partition, 
            hidden_size,
            padding_idx=padding_idx if padding_idx is not None and 
                       self.vocab_start <= padding_idx < self.vocab_end else None
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x >= self.vocab_start) & (x < self.vocab_end)
        
        local_x = x - self.vocab_start
        local_x = local_x.clamp(0, self.vocab_per_partition - 1)
        
        output = self.embedding(local_x)
        
        output = output * mask.unsqueeze(-1).to(output.dtype)
        
        if self.tp_size > 1:
            output = all_reduce_tp(output)
        
        return output


class ParallelLMHead(nn.Module):
    """
    Language model head with vocabulary parallelism.
    
    Each rank computes logits for vocab_size/tp_size tokens.
    Final logits are gathered if needed for loss computation.
    """
    
    def __init__(self, hidden_size: int, vocab_size: int, gather_output: bool = True):
        super().__init__()
        ctx = get_parallel_context()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = ctx.config.tp_size
        self.gather_output = gather_output
        
        self.lm_head = ColumnParallelLinear(
            hidden_size,
            vocab_size,
            bias=False,
            gather_output=gather_output,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)