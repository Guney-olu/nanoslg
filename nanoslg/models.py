"""
Model architectures with paged KV cache support.
Supports: Llama3, Qwen2, GLM, Mistral, etc.
"""
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
from .kv_cache import CacheContext


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return self.weight * (x32 * torch.rsqrt(
            x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


def precompute_rope_frequencies(
    dim: int, max_seq_len: int, theta: float, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))

    if hasattr(config, "rope_scaling") and config.rope_scaling:
        if config.rope_scaling.get("rope_type") == "llama3":
            sc = config.rope_scaling
            factor = sc.get("factor", 8.0)
            lo = sc.get("low_freq_factor", 1.0)
            hi = sc.get("high_freq_factor", 4.0)
            old_ctx = sc.get("original_max_position_embeddings", 8192)
            lo_wl = old_ctx / lo
            hi_wl = old_ctx / hi
            new = []
            for f in freqs:
                wl = 2 * math.pi / f
                if wl < hi_wl:
                    new.append(f)
                elif wl > lo_wl:
                    new.append(f / factor)
                else:
                    s = (old_ctx / wl - lo) / (hi - lo)
                    new.append((1 - s) * f / factor + s * f)
            freqs = torch.tensor(new, dtype=freqs.dtype)

    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x: [B,S,H,D], cos/sin: [B,S,D] or [S,D]."""
    if cos.dim() == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif cos.dim() == 3:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    cos = cos.to(x.device, dtype=x.dtype)
    sin = sin.to(x.device, dtype=x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _attention_forward(
    q: torch.Tensor,           # [B, S, Hq, D]
    k_new: torch.Tensor,       # [B, S, Hkv, D]
    v_new: torch.Tensor,       # [B, S, Hkv, D]
    layer_idx: int,
    n_heads: int,
    n_kv_heads: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_ctx: CacheContext,
) -> torch.Tensor:
    B, Seq = q.shape[0], q.shape[1]

    pos_ids = cache_ctx.get_position_ids(Seq)       # [B, S]
    cos_pos = cos[pos_ids]                           # [B, S, D]
    sin_pos = sin[pos_ids]
    q = apply_rotary_emb(q, cos_pos, sin_pos)
    k_new = apply_rotary_emb(k_new, cos_pos, sin_pos)

    return cache_ctx.attend(layer_idx, q, k_new, v_new, n_heads, n_kv_heads)


class TPAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        ctx = get_parallel_context()
        tp = ctx.config.tp_size
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = getattr(config, "num_key_value_heads",
                         getattr(config, "multi_query_group_num", self.n_heads))
        self.head_dim = config.hidden_size // self.n_heads
        self.layer_idx = layer_idx

        assert self.n_heads % tp == 0
        assert self.n_kv_heads % tp == 0
        self.n_heads_tp = self.n_heads // tp
        self.n_kv_heads_tp = self.n_kv_heads // tp

        self.q_proj = ColumnParallelLinear(
            self.hidden_size, self.n_heads * self.head_dim, gather_output=False)
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, self.n_kv_heads * self.head_dim, gather_output=False)
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, self.n_kv_heads * self.head_dim, gather_output=False)
        self.o_proj = RowParallelLinear(
            self.n_heads * self.head_dim, self.hidden_size, input_is_parallel=True)

    def forward(self, x, cos, sin, cache_ctx: CacheContext, mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads_tp, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads_tp, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads_tp, self.head_dim)
        attn_out = _attention_forward(
            q, k, v, self.layer_idx,
            self.n_heads_tp, self.n_kv_heads_tp,
            cos, sin, cache_ctx)
        return self.o_proj(attn_out)


class TPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, gather_output=False)
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, gather_output=False)
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, input_is_parallel=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TPTransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = TPAttention(config, layer_idx)
        self.mlp = TPMLP(config)
        eps = getattr(config, "rms_norm_eps", 1e-5)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def forward(self, x, cos, sin, cache_ctx, mask=None):
        h = x + self.self_attn(
            self.input_layernorm(x), cos, sin, cache_ctx, mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class PPAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = getattr(config, "num_key_value_heads",
                         getattr(config, "multi_query_group_num", self.n_heads))
        self.head_dim = config.hidden_size // self.n_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, cos, sin, cache_ctx: CacheContext, mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        attn_out = _attention_forward(
            q, k, v, self.layer_idx,
            self.n_heads, self.n_kv_heads,
            cos, sin, cache_ctx)
        return self.o_proj(attn_out)


# ═══════════════════════════════════════════════════════════════════
#  PP MLP
# ═══════════════════════════════════════════════════════════════════

class PPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════════════
#  PP Transformer Block
# ═══════════════════════════════════════════════════════════════════

class PPTransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = PPAttention(config, layer_idx)
        self.mlp = PPMLP(config)
        eps = getattr(config, "rms_norm_eps", 1e-5)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def forward(self, x, cos, sin, cache_ctx, mask=None):
        h = x + self.self_attn(
            self.input_layernorm(x), cos, sin, cache_ctx, mask)
        return h + self.mlp(self.post_attention_layernorm(h))


# ═══════════════════════════════════════════════════════════════════
#  Full Models
# ═══════════════════════════════════════════════════════════════════

class LlamaTP(nn.Module):
    """Full model for Tensor Parallel mode (all layers on every GPU)."""

    def __init__(self, hf_config):
        super().__init__()
        ctx = get_parallel_context()
        self.device = ctx.device
        self.n_layers = hf_config.num_hidden_layers

        self.embed_tokens = nn.Embedding(
            hf_config.vocab_size, hf_config.hidden_size)
        self.layers = nn.ModuleList(
            [TPTransformerBlock(hf_config, i) for i in range(self.n_layers)])
        self.norm = RMSNorm(
            hf_config.hidden_size,
            eps=getattr(hf_config, "rms_norm_eps", 1e-5))
        self.lm_head = ParallelLMHead(
            hf_config.hidden_size, hf_config.vocab_size, gather_output=True)

        theta = getattr(hf_config, "rope_theta", 500000.0)
        mx = getattr(hf_config, "max_position_embeddings", 8192)
        hd = hf_config.hidden_size // hf_config.num_attention_heads
        self.cos, self.sin = precompute_rope_frequencies(hd, mx, theta, hf_config)
        self.cos = self.cos.to(self.device)
        self.sin = self.sin.to(self.device)

    def forward(self, x, cache_ctx: CacheContext, mask=None):
        h = self.embed_tokens(x) if x.dtype == torch.long else x
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, cache_ctx, mask)
        return self.lm_head(self.norm(h))


class LlamaPP(nn.Module):
    """Partial model for Pipeline Parallel mode (subset of layers per GPU)."""

    def __init__(self, hf_config, layer_indices: List[int],
                 is_first: bool, is_last: bool):
        super().__init__()
        ctx = get_parallel_context()
        self.device = ctx.device
        self.layer_indices = layer_indices
        self.is_first = is_first
        self.is_last = is_last

        self.layers = nn.ModuleList(
            [PPTransformerBlock(hf_config, i) for i in range(len(layer_indices))])

        if is_first:
            self.embed_tokens = nn.Embedding(
                hf_config.vocab_size, hf_config.hidden_size)
        if is_last:
            self.norm = RMSNorm(
                hf_config.hidden_size,
                eps=getattr(hf_config, "rms_norm_eps", 1e-5))
            self.lm_head = nn.Linear(
                hf_config.hidden_size, hf_config.vocab_size, bias=False)

        theta = getattr(hf_config, "rope_theta", 500000.0)
        mx = getattr(hf_config, "max_position_embeddings", 8192)
        hd = hf_config.hidden_size // hf_config.num_attention_heads
        self.cos, self.sin = precompute_rope_frequencies(hd, mx, theta, hf_config)
        self.cos = self.cos.to(self.device)
        self.sin = self.sin.to(self.device)

    def forward(self, x, cache_ctx: CacheContext, mask=None):
        h = (self.embed_tokens(x)
             if self.is_first and x.dtype == torch.long else x)
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, cache_ctx, mask)
        if self.is_last:
            h = self.lm_head(self.norm(h))
        return h


class LlamaHybrid(nn.Module):
    """Partial model for Hybrid TP+PP mode (TP layers, PP layer split)."""

    def __init__(self, hf_config, layer_indices: List[int],
                 is_first: bool, is_last: bool):
        super().__init__()
        ctx = get_parallel_context()
        self.device = ctx.device
        self.layer_indices = layer_indices
        self.is_first = is_first
        self.is_last = is_last

        self.layers = nn.ModuleList(
            [TPTransformerBlock(hf_config, i) for i in range(len(layer_indices))])

        if is_first:
            self.embed_tokens = nn.Embedding(
                hf_config.vocab_size, hf_config.hidden_size)
        if is_last:
            self.norm = RMSNorm(
                hf_config.hidden_size,
                eps=getattr(hf_config, "rms_norm_eps", 1e-5))
            self.lm_head = ParallelLMHead(
                hf_config.hidden_size, hf_config.vocab_size, gather_output=True)

        theta = getattr(hf_config, "rope_theta", 500000.0)
        mx = getattr(hf_config, "max_position_embeddings", 8192)
        hd = hf_config.hidden_size // hf_config.num_attention_heads
        self.cos, self.sin = precompute_rope_frequencies(hd, mx, theta, hf_config)
        self.cos = self.cos.to(self.device)
        self.sin = self.sin.to(self.device)

    def forward(self, x, cache_ctx: CacheContext, mask=None):
        h = (self.embed_tokens(x)
             if self.is_first and x.dtype == torch.long else x)
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, cache_ctx, mask)
        if self.is_last:
            h = self.lm_head(self.norm(h))
        return h

def create_model(hf_config, parallel_config: ParallelConfig):
    ctx = get_parallel_context()

    if parallel_config.mode == ParallelMode.TENSOR:
        return LlamaTP(hf_config)

    elif parallel_config.mode == ParallelMode.PIPELINE:
        li = parallel_config.pp_layer_splits[ctx.pp_rank]
        return LlamaPP(
            hf_config, li,
            is_first=(ctx.pp_rank == 0),
            is_last=(ctx.pp_rank == parallel_config.pp_size - 1))

    elif parallel_config.mode == ParallelMode.HYBRID:
        li = parallel_config.pp_layer_splits[ctx.pp_rank]
        return LlamaHybrid(
            hf_config, li,
            is_first=(ctx.pp_rank == 0),
            is_last=(ctx.pp_rank == parallel_config.pp_size - 1))

    else:
        raise ValueError(f"Unknown mode: {parallel_config.mode}")


def load_weights_tp(model: LlamaTP, model_path: str, hf_config, dtype):
    ctx = get_parallel_context()
    tp_rank, tp_size = ctx.tp_rank, ctx.config.tp_size
    files = sorted(glob(f"{model_path}/*.safetensors"))
    lm_head_found = False

    for fp in files:
        sd = load_file(fp)
        for key, tensor in sd.items():
            key = key.replace("model.", "")
            try:
                if "embed_tokens" in key:
                    model.embed_tokens.weight.data.copy_(
                        tensor.to(model.device, dtype))

                elif key.startswith("layers."):
                    parts = key.split(".")
                    li = int(parts[1])
                    rest = ".".join(parts[2:])
                    layer = model.layers[li]

                    if "q_proj.weight" in rest:
                        layer.self_attn.q_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(model.device, dtype))
                    elif "k_proj.weight" in rest:
                        layer.self_attn.k_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(model.device, dtype))
                    elif "v_proj.weight" in rest:
                        layer.self_attn.v_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(model.device, dtype))
                    elif "gate_proj.weight" in rest:
                        layer.mlp.gate_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(model.device, dtype))
                    elif "up_proj.weight" in rest:
                        layer.mlp.up_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(model.device, dtype))
                    elif "o_proj.weight" in rest:
                        layer.self_attn.o_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 1)[tp_rank].to(model.device, dtype))
                    elif "down_proj.weight" in rest:
                        layer.mlp.down_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 1)[tp_rank].to(model.device, dtype))
                    elif "input_layernorm" in rest:
                        layer.input_layernorm.weight.data.copy_(
                            tensor.to(model.device, dtype))
                    elif "post_attention_layernorm" in rest:
                        layer.post_attention_layernorm.weight.data.copy_(
                            tensor.to(model.device, dtype))

                elif "norm.weight" in key:
                    model.norm.weight.data.copy_(
                        tensor.to(model.device, dtype))

                elif "lm_head.weight" in key:
                    model.lm_head.lm_head.weight.data.copy_(
                        tensor.chunk(tp_size, 0)[tp_rank].to(model.device, dtype))
                    lm_head_found = True

            except Exception as e:
                print(f"[TP Load] Error {key}: {e}")
        del sd
        torch.cuda.empty_cache()

    if not lm_head_found and getattr(hf_config, "tie_word_embeddings", False):
        full = model.embed_tokens.weight.data
        model.lm_head.lm_head.weight.data.copy_(
            full.chunk(tp_size, 0)[tp_rank])
    print(f"[TP Rank {tp_rank}] Weights loaded")


def load_weights_pp(model: LlamaPP, model_path: str, hf_config, dtype):
    ctx = get_parallel_context()

    def _map(key):
        key = key.replace("model.", "")
        if key.startswith("layers."):
            p = key.split(".")
            gi = int(p[1])
            if gi in model.layer_indices:
                return (f"layers.{model.layer_indices.index(gi)}."
                        + ".".join(p[2:]))
        if model.is_first and "embed_tokens" in key:
            return "embed_tokens.weight"
        if model.is_last:
            if "norm.weight" in key:
                return "norm.weight"
            if "lm_head" in key:
                return "lm_head.weight"
        return None

    for fp in sorted(glob(f"{model_path}/*.safetensors")):
        sd = load_file(fp)
        for key, tensor in sd.items():
            lk = _map(key)
            if lk:
                mod = model
                parts = lk.split(".")
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                getattr(mod, parts[-1]).data.copy_(
                    tensor.to(model.device, dtype))
        del sd
        torch.cuda.empty_cache()
    print(f"[PP Rank {ctx.pp_rank}] Loaded layers {model.layer_indices}")


def load_weights_hybrid(model: LlamaHybrid, model_path: str,
                        hf_config, dtype):
    ctx = get_parallel_context()
    tp_rank, tp_size = ctx.tp_rank, ctx.config.tp_size

    for fp in sorted(glob(f"{model_path}/*.safetensors")):
        sd = load_file(fp)
        for key, tensor in sd.items():
            key = key.replace("model.", "")
            try:
                if model.is_first and "embed_tokens" in key:
                    model.embed_tokens.weight.data.copy_(
                        tensor.to(model.device, dtype))

                elif key.startswith("layers."):
                    parts = key.split(".")
                    gi = int(parts[1])
                    if gi not in model.layer_indices:
                        continue
                    li = model.layer_indices.index(gi)
                    rest = ".".join(parts[2:])
                    layer = model.layers[li]

                    if "q_proj.weight" in rest:
                        layer.self_attn.q_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(
                                model.device, dtype))
                    elif "k_proj.weight" in rest:
                        layer.self_attn.k_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(
                                model.device, dtype))
                    elif "v_proj.weight" in rest:
                        layer.self_attn.v_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(
                                model.device, dtype))
                    elif "gate_proj.weight" in rest:
                        layer.mlp.gate_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(
                                model.device, dtype))
                    elif "up_proj.weight" in rest:
                        layer.mlp.up_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 0)[tp_rank].to(
                                model.device, dtype))
                    elif "o_proj.weight" in rest:
                        layer.self_attn.o_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 1)[tp_rank].to(
                                model.device, dtype))
                    elif "down_proj.weight" in rest:
                        layer.mlp.down_proj.weight.data.copy_(
                            tensor.chunk(tp_size, 1)[tp_rank].to(
                                model.device, dtype))
                    elif "input_layernorm" in rest:
                        layer.input_layernorm.weight.data.copy_(
                            tensor.to(model.device, dtype))
                    elif "post_attention_layernorm" in rest:
                        layer.post_attention_layernorm.weight.data.copy_(
                            tensor.to(model.device, dtype))

                elif model.is_last and "norm.weight" in key:
                    model.norm.weight.data.copy_(
                        tensor.to(model.device, dtype))

                elif model.is_last and "lm_head.weight" in key:
                    model.lm_head.lm_head.weight.data.copy_(
                        tensor.chunk(tp_size, 0)[tp_rank].to(
                            model.device, dtype))

            except Exception as e:
                print(f"[Hybrid Load] Error {key}: {e}")
        del sd
        torch.cuda.empty_cache()
    print(f"[Hybrid Rank {ctx.rank}] PP={ctx.pp_rank} TP={tp_rank} "
          f"loaded {model.layer_indices}")


def load_weights(model, model_path, hf_config, dtype, parallel_config):
    if parallel_config.mode == ParallelMode.TENSOR:
        load_weights_tp(model, model_path, hf_config, dtype)
    elif parallel_config.mode == ParallelMode.PIPELINE:
        load_weights_pp(model, model_path, hf_config, dtype)
    elif parallel_config.mode == ParallelMode.HYBRID:
        load_weights_hybrid(model, model_path, hf_config, dtype)