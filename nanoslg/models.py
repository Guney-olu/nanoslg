"""
Model architectures.
SUPPORT:
1. Llama3-family models
"""
#TODO ADD Qwen, GLM, Mistral

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoConfig
from safetensors.torch import load_file
from glob import glob

from .parallel import (
    ParallelContext, ParallelMode, ParallelConfig,
    get_parallel_context, all_reduce_tp,
)
from .tp_layers import (
    ColumnParallelLinear, RowParallelLinear,
    VocabParallelEmbedding, ParallelLMHead,
)


class RMSNorm(nn.Module):    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_fp32.to(x.dtype)


def precompute_rope_frequencies(
    dim: int, 
    max_seq_len: int, 
    theta: float,
    config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary position embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Llama 3 RoPE scaling
    if hasattr(config, "rope_scaling") and config.rope_scaling:
        if config.rope_scaling.get("rope_type") == "llama3":
            scale_config = config.rope_scaling
            factor = scale_config.get("factor", 8.0)
            low_freq_factor = scale_config.get("low_freq_factor", 1.0)
            high_freq_factor = scale_config.get("high_freq_factor", 4.0)
            old_context_len = scale_config.get("original_max_position_embeddings", 8192)
            
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            
            new_freqs = []
            for freq in freqs:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / factor)
                else:
                    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
            freqs = torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)
    
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs = torch.cat((freqs, freqs), dim=-1)
    
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings."""
    cos = cos[None, :, None, :].to(x.device).type_as(x)
    sin = sin[None, :, None, :].to(x.device).type_as(x)
    
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    
    return (x * cos) + (rotated * sin)


# Tensor Parallel Attention IMPLEMENTATION
class TPAttention(nn.Module):
    """
    Multi-head attention with tensor parallelism.
    
    - Q, K, V: Column parallel (split heads across GPUs)
    - O: Row parallel (all-reduce after projection)
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        ctx = get_parallel_context()
        tp_size = ctx.config.tp_size
        
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx
        
        # Validate head distribution
        assert self.n_heads % tp_size == 0, \
            f"n_heads ({self.n_heads}) must be divisible by tp_size ({tp_size})"
        assert self.n_kv_heads % tp_size == 0, \
            f"n_kv_heads ({self.n_kv_heads}) must be divisible by tp_size ({tp_size})"
        
        self.n_heads_per_tp = self.n_heads // tp_size
        self.n_kv_heads_per_tp = self.n_kv_heads // tp_size
        
        # Column parallel QKV (each GPU gets subset of heads)
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.n_heads * self.head_dim,
            gather_output=False,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            gather_output=False,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            gather_output=False,
        )
        
        # Row parallel output (all-reduce combines head outputs)
        self.o_proj = RowParallelLinear(
            self.n_heads * self.head_dim,
            self.hidden_size,
            input_is_parallel=True,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        B, Seq, _ = x.shape
        
        # QKV projection - each GPU gets partial heads
        q = self.q_proj(x).view(B, Seq, self.n_heads_per_tp, self.head_dim)
        k = self.k_proj(x).view(B, Seq, self.n_kv_heads_per_tp, self.head_dim)
        v = self.v_proj(x).view(B, Seq, self.n_kv_heads_per_tp, self.head_dim)
        
        # Get start position from cache
        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0:
            if len(kv_caches[0]) > self.layer_idx:
                cache = kv_caches[0][self.layer_idx]
                if cache.get('k') is not None:
                    start_pos = cache['k'].shape[1]
        
        # Apply RoPE
        curr_cos = cos[start_pos:start_pos + Seq]
        curr_sin = sin[start_pos:start_pos + Seq]
        q = apply_rotary_emb(q, curr_cos, curr_sin)
        k = apply_rotary_emb(k, curr_cos, curr_sin)
        
        # KV cache update
        if kv_caches is not None:
            new_k_list, new_v_list = [], []
            
            for b in range(B):
                while len(kv_caches[b]) <= self.layer_idx:
                    kv_caches[b].append({'k': None, 'v': None})
                
                cache = kv_caches[b][self.layer_idx]
                k_b, v_b = k[b:b+1], v[b:b+1]
                
                if cache['k'] is not None:
                    k_b = torch.cat([cache['k'], k_b], dim=1)
                    v_b = torch.cat([cache['v'], v_b], dim=1)
                
                cache['k'], cache['v'] = k_b, v_b
                new_k_list.append(k_b)
                new_v_list.append(v_b)
            
            max_len = max(kk.shape[1] for kk in new_k_list)
            k_padded, v_padded = [], []
            for kk, vv in zip(new_k_list, new_v_list):
                pad = max_len - kk.shape[1]
                if pad > 0:
                    kk = F.pad(kk, (0, 0, 0, 0, pad, 0))
                    vv = F.pad(vv, (0, 0, 0, 0, pad, 0))
                k_padded.append(kk)
                v_padded.append(vv)
            
            k = torch.cat(k_padded, dim=0)
            v = torch.cat(v_padded, dim=0)
        
        # GQA expansion
        if self.n_kv_heads_per_tp != self.n_heads_per_tp:
            expand = self.n_heads_per_tp // self.n_kv_heads_per_tp
            k = k.repeat_interleave(expand, dim=2)
            v = v.repeat_interleave(expand, dim=2)
        
        # Attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            is_causal=(Seq > 1 and start_pos == 0)
        )
        
        # Output projection with all-reduce
        out = out.transpose(1, 2).contiguous().view(B, Seq, -1)
        return self.o_proj(out)


class TPMLP(nn.Module):
    """
    SwiGLU MLP with tensor parallelism.
    
    - gate_proj, up_proj: Column parallel
    - down_proj: Row parallel
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            input_is_parallel=True,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TPTransformerBlock(nn.Module):
    """Transformer block with tensor parallelism."""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = TPAttention(config, layer_idx)
        self.mlp = TPMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_caches: Optional[List[Dict]],
    ) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, kv_caches)
        return h + self.mlp(self.post_attention_layernorm(h))


# Pipeline Parallel Component (INITIAL)
class PPAttention(nn.Module):
    """Original attention for pipeline parallelism."""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx
        
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        B, Seq, _ = x.shape
        
        q = self.q_proj(x).view(B, Seq, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, Seq, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, Seq, self.n_kv_heads, self.head_dim)

        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0:
            if len(kv_caches[0]) > self.layer_idx and kv_caches[0][self.layer_idx].get('k') is not None:
                start_pos = kv_caches[0][self.layer_idx]['k'].shape[1]
        
        curr_cos = cos[start_pos:start_pos + Seq]
        curr_sin = sin[start_pos:start_pos + Seq]
        q = apply_rotary_emb(q, curr_cos, curr_sin)
        k = apply_rotary_emb(k, curr_cos, curr_sin)

        if kv_caches is not None:
            new_k_list, new_v_list = [], []
            for b in range(B):
                while len(kv_caches[b]) <= self.layer_idx:
                    kv_caches[b].append({'k': None, 'v': None})
                cache = kv_caches[b][self.layer_idx]
                k_b, v_b = k[b:b+1], v[b:b+1]
                if cache['k'] is not None:
                    k_b = torch.cat([cache['k'], k_b], dim=1)
                    v_b = torch.cat([cache['v'], v_b], dim=1)
                cache['k'], cache['v'] = k_b, v_b
                new_k_list.append(k_b)
                new_v_list.append(v_b)
            
            max_len = max(kk.shape[1] for kk in new_k_list)
            k_padded, v_padded = [], []
            for kk, vv in zip(new_k_list, new_v_list):
                pad = max_len - kk.shape[1]
                if pad > 0:
                    kk = F.pad(kk, (0, 0, 0, 0, pad, 0))
                    vv = F.pad(vv, (0, 0, 0, 0, pad, 0))
                k_padded.append(kk)
                v_padded.append(vv)
            k = torch.cat(k_padded, dim=0)
            v = torch.cat(v_padded, dim=0)

        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(Seq > 1 and start_pos == 0))
        
        return self.o_proj(output.transpose(1, 2).contiguous().view(B, Seq, -1))


class PPMLP(nn.Module):
    """Original MLP for pipeline parallelism."""
    
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PPTransformerBlock(nn.Module):
    """Transformer block for pipeline parallelism."""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = PPAttention(config, layer_idx)
        self.mlp = PPMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_caches: Optional[List[Dict]],
    ) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, kv_caches)
        return h + self.mlp(self.post_attention_layernorm(h))



class LlamaTP(nn.Module):
    """
    Llama model with Tensor Parallelism only.
    All layers on all GPUs, weights sharded.
    """
    
    def __init__(self, hf_config):
        super().__init__()
        ctx = get_parallel_context()
        self.ctx = ctx
        self.device = ctx.device
        
        self.n_layers = hf_config.num_hidden_layers
        
        # Replicated embedding (simpler than vocab parallel for small models)
        self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size)
        
        # TP layers
        self.layers = nn.ModuleList([
            TPTransformerBlock(hf_config, i) for i in range(self.n_layers)
        ])
        
        self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        
        # Parallel LM head
        self.lm_head = ParallelLMHead(
            hf_config.hidden_size,
            hf_config.vocab_size,
            gather_output=True,
        )
        
        # RoPE
        theta = getattr(hf_config, "rope_theta", 500000.0)
        max_seq = getattr(hf_config, "max_position_embeddings", 8192)
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        self.cos, self.sin = precompute_rope_frequencies(head_dim, max_seq, theta, hf_config)
        self.cos = self.cos.to(self.device)
        self.sin = self.sin.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_caches: List[List[Dict]],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(x) if x.dtype == torch.long else x
        
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, mask, kv_caches)
        
        return self.lm_head(self.norm(h))


class LlamaPP(nn.Module):
    """
    Llama model with Pipeline Parallelism only (Original LlamaStage).
    Layers split across GPUs.
    """
    
    def __init__(self, hf_config, layer_indices: List[int], is_first: bool, is_last: bool):
        super().__init__()
        ctx = get_parallel_context()
        self.device = ctx.device
        self.layer_indices = layer_indices
        self.is_first = is_first
        self.is_last = is_last
        
        self.layers = nn.ModuleList([
            PPTransformerBlock(hf_config, i) for i in range(len(layer_indices))
        ])
        
        if is_first:
            self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size)
        
        if is_last:
            self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
            self.lm_head = nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False)
        
        theta = getattr(hf_config, "rope_theta", 500000.0)
        max_seq = getattr(hf_config, "max_position_embeddings", 8192)
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        self.cos, self.sin = precompute_rope_frequencies(head_dim, max_seq, theta, hf_config)
        self.cos = self.cos.to(self.device)
        self.sin = self.sin.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: List[List[Dict]],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(x) if self.is_first and x.dtype == torch.long else x
        
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, mask, kv_caches)
        
        if self.is_last:
            h = self.lm_head(self.norm(h))
        
        return h


class LlamaHybrid(nn.Module):
    """
    Llama model with Hybrid TP+PP parallelism.
    
    Layers are split across PP stages.
    Within each stage, tensors are split across TP ranks.
    """
    
    def __init__(self, hf_config, layer_indices: List[int], is_first: bool, is_last: bool):
        super().__init__()
        ctx = get_parallel_context()
        self.device = ctx.device
        self.layer_indices = layer_indices
        self.is_first = is_first
        self.is_last = is_last
        
        # Use TP layers within PP stage
        self.layers = nn.ModuleList([
            TPTransformerBlock(hf_config, i) for i in range(len(layer_indices))
        ])
        
        if is_first:
            # Replicated embedding
            self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size)
        
        if is_last:
            self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
            self.lm_head = ParallelLMHead(
                hf_config.hidden_size,
                hf_config.vocab_size,
                gather_output=True,
            )
        
        theta = getattr(hf_config, "rope_theta", 500000.0)
        max_seq = getattr(hf_config, "max_position_embeddings", 8192)
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        self.cos, self.sin = precompute_rope_frequencies(head_dim, max_seq, theta, hf_config)
        self.cos = self.cos.to(self.device)
        self.sin = self.sin.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: List[List[Dict]],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(x) if self.is_first and x.dtype == torch.long else x
        
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, mask, kv_caches)
        
        if self.is_last:
            h = self.lm_head(self.norm(h))
        
        return h



def create_model(hf_config, parallel_config: ParallelConfig):
    """Create appropriate model based on parallelism mode."""
    ctx = get_parallel_context()
    
    if parallel_config.mode == ParallelMode.TENSOR:
        return LlamaTP(hf_config)
    
    elif parallel_config.mode == ParallelMode.PIPELINE:
        layer_indices = parallel_config.pp_layer_splits[ctx.pp_rank]
        is_first = ctx.pp_rank == 0
        is_last = ctx.pp_rank == parallel_config.pp_size - 1
        return LlamaPP(hf_config, layer_indices, is_first, is_last)
    
    elif parallel_config.mode == ParallelMode.HYBRID:
        layer_indices = parallel_config.pp_layer_splits[ctx.pp_rank]
        is_first = ctx.pp_rank == 0
        is_last = ctx.pp_rank == parallel_config.pp_size - 1
        return LlamaHybrid(hf_config, layer_indices, is_first, is_last)
    
    else:
        raise ValueError(f"Unknown parallel mode: {parallel_config.mode}")


def load_weights_tp(model: LlamaTP, model_path: str, hf_config, dtype: torch.dtype):
    """Load and shard weights for tensor parallel model."""
    ctx = get_parallel_context()
    tp_rank = ctx.tp_rank
    tp_size = ctx.config.tp_size
    
    files = sorted(glob(f"{model_path}/*.safetensors"))
    
    for filepath in files:
        state_dict = load_file(filepath)
        
        for key, tensor in state_dict.items():
            key = key.replace("model.", "")
            
            try:
                # Embedding
                if "embed_tokens" in key:
                    model.embed_tokens.weight.data.copy_(tensor.to(model.device).to(dtype))
                
                # Transformer layers
                elif key.startswith("layers."):
                    parts = key.split(".")
                    layer_idx = int(parts[1])
                    rest = ".".join(parts[2:])
                    
                    layer = model.layers[layer_idx]
                    
                    # Column parallel weights (shard output dim)
                    if "q_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.self_attn.q_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "k_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.self_attn.k_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "v_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.self_attn.v_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "gate_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.mlp.gate_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "up_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.mlp.up_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    
                    # Row parallel weights (shard input dim)
                    elif "o_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=1)[tp_rank]
                        layer.self_attn.o_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "down_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=1)[tp_rank]
                        layer.mlp.down_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    
                    # Replicated weights
                    elif "input_layernorm" in rest:
                        layer.input_layernorm.weight.data.copy_(tensor.to(model.device).to(dtype))
                    elif "post_attention_layernorm" in rest:
                        layer.post_attention_layernorm.weight.data.copy_(tensor.to(model.device).to(dtype))
                
                # Output
                elif "norm.weight" in key:
                    model.norm.weight.data.copy_(tensor.to(model.device).to(dtype))
                elif "lm_head.weight" in key:
                    shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                    model.lm_head.lm_head.weight.data.copy_(shard.to(model.device).to(dtype))
                
            except Exception as e:
                print(f"[TP Load] Error loading {key}: {e}")
        
        del state_dict
        torch.cuda.empty_cache()
    
    print(f"[TP Rank {tp_rank}] Weights loaded")


def load_weights_pp(model: LlamaPP, model_path: str, hf_config, dtype: torch.dtype):
    """Load weights for pipeline parallel model."""
    ctx = get_parallel_context()
    
    def map_key(key: str):
        key = key.replace("model.", "")
        
        if key.startswith("layers."):
            parts = key.split(".")
            global_idx = int(parts[1])
            if global_idx in model.layer_indices:
                local_idx = model.layer_indices.index(global_idx)
                return f"layers.{local_idx}." + ".".join(parts[2:])
        
        if model.is_first and "embed_tokens" in key:
            return "embed_tokens.weight"
        
        if model.is_last:
            if "norm.weight" in key:
                return "norm.weight"
            if "lm_head" in key:
                return "lm_head.weight"
        
        return None
    
    files = sorted(glob(f"{model_path}/*.safetensors"))
    for filepath in files:
        state_dict = load_file(filepath)
        for key, tensor in state_dict.items():
            local_key = map_key(key)
            if local_key:
                module = model
                parts = local_key.split(".")
                for part in parts[:-1]:
                    module = getattr(module, part)
                param = getattr(module, parts[-1])
                param.data.copy_(tensor.to(model.device).to(dtype))
        del state_dict
        torch.cuda.empty_cache()
    
    print(f"[PP Rank {ctx.pp_rank}] Loaded layers {model.layer_indices}")


def load_weights_hybrid(model: LlamaHybrid, model_path: str, hf_config, dtype: torch.dtype):
    """Load and shard weights for hybrid TP+PP model."""
    ctx = get_parallel_context()
    tp_rank = ctx.tp_rank
    tp_size = ctx.config.tp_size
    
    files = sorted(glob(f"{model_path}/*.safetensors"))
    
    for filepath in files:
        state_dict = load_file(filepath)
        
        for key, tensor in state_dict.items():
            key = key.replace("model.", "")
            
            try:
                # Embedding (first PP stage only)
                if model.is_first and "embed_tokens" in key:
                    model.embed_tokens.weight.data.copy_(tensor.to(model.device).to(dtype))
                
                # Transformer layers
                elif key.startswith("layers."):
                    parts = key.split(".")
                    global_idx = int(parts[1])
                    
                    if global_idx not in model.layer_indices:
                        continue
                    
                    local_idx = model.layer_indices.index(global_idx)
                    layer = model.layers[local_idx]
                    rest = ".".join(parts[2:])
                    
                    # Column parallel (shard output dim)
                    if "q_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.self_attn.q_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "k_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.self_attn.k_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "v_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.self_attn.v_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "gate_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.mlp.gate_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "up_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                        layer.mlp.up_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    
                    # Row parallel (shard input dim)
                    elif "o_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=1)[tp_rank]
                        layer.self_attn.o_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    elif "down_proj.weight" in rest:
                        shard = tensor.chunk(tp_size, dim=1)[tp_rank]
                        layer.mlp.down_proj.weight.data.copy_(shard.to(model.device).to(dtype))
                    
                    # Replicated
                    elif "input_layernorm" in rest:
                        layer.input_layernorm.weight.data.copy_(tensor.to(model.device).to(dtype))
                    elif "post_attention_layernorm" in rest:
                        layer.post_attention_layernorm.weight.data.copy_(tensor.to(model.device).to(dtype))
                
                # Output (last PP stage only)
                elif model.is_last and "norm.weight" in key:
                    model.norm.weight.data.copy_(tensor.to(model.device).to(dtype))
                elif model.is_last and "lm_head.weight" in key:
                    shard = tensor.chunk(tp_size, dim=0)[tp_rank]
                    model.lm_head.lm_head.weight.data.copy_(shard.to(model.device).to(dtype))
                    
            except Exception as e:
                print(f"[Hybrid Load] Error {key}: {e}")
        
        del state_dict
        torch.cuda.empty_cache()
    
    print(f"[Hybrid Rank {ctx.rank}] PP={ctx.pp_rank} TP={tp_rank} loaded layers {model.layer_indices}")


def load_weights(model, model_path: str, hf_config, dtype: torch.dtype, parallel_config: ParallelConfig):
    """Unified weight loading dispatcher."""
    if parallel_config.mode == ParallelMode.TENSOR:
        load_weights_tp(model, model_path, hf_config, dtype)
    elif parallel_config.mode == ParallelMode.PIPELINE:
        load_weights_pp(model, model_path, hf_config, dtype)
    elif parallel_config.mode == ParallelMode.HYBRID:
        load_weights_hybrid(model, model_path, hf_config, dtype)