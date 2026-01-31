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

from .config import ModelConfig

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


def apply_rotary_emb(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embeddings."""
    cos = cos[None, :, None, :].to(x.device).type_as(x)
    sin = sin[None, :, None, :].to(x.device).type_as(x)
    
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    
    return (x * cos) + (rotated * sin)

class Attention(nn.Module):
    """Multi-head attention with batched KV cache support."""
    
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.layer_idx = None  # Set by TransformerBlock

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Dict]] = None,  # List of caches, one per batch item
    ) -> torch.Tensor:
        B, Seq, _ = x.shape
        
        q = self.q_proj(x).view(B, Seq, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, Seq, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, Seq, self.n_kv_heads, self.head_dim)

        if kv_caches is not None and len(kv_caches) > 0:
            # TO Check if this layer has cached values
            if len(kv_caches[0]) > self.layer_idx and kv_caches[0][self.layer_idx].get('k') is not None:
                # TO Get start positions per batch item (they should all be the same for simplicity)
                start_pos = kv_caches[0][self.layer_idx]['k'].shape[1]
            else:
                start_pos = 0
        else:
            start_pos = 0
        
        # Position embeddings
        curr_cos = cos[start_pos:start_pos + Seq]
        curr_sin = sin[start_pos:start_pos + Seq]
        
        q = apply_rotary_emb(q, curr_cos, curr_sin)
        k = apply_rotary_emb(k, curr_cos, curr_sin)

        # Update KV cache per batch item
        if kv_caches is not None:
            new_k_list = []
            new_v_list = []
            
            for b in range(B):
                # Ensure cache list is long enough
                while len(kv_caches[b]) <= self.layer_idx:
                    kv_caches[b].append({'k': None, 'v': None})
                
                cache = kv_caches[b][self.layer_idx]
                k_b = k[b:b+1]  # [1, seq, n_kv_heads, head_dim]
                v_b = v[b:b+1]
                
                if cache['k'] is not None:
                    k_b = torch.cat([cache['k'], k_b], dim=1)
                    v_b = torch.cat([cache['v'], v_b], dim=1)
                
                cache['k'] = k_b
                cache['v'] = v_b
                new_k_list.append(k_b)
                new_v_list.append(v_b)
            
            max_cache_len = max(kk.shape[1] for kk in new_k_list)
            
            # Pad to same length for batched attention
            k_padded = []
            v_padded = []
            for kk, vv in zip(new_k_list, new_v_list):
                pad_len = max_cache_len - kk.shape[1]
                if pad_len > 0:
                    kk = F.pad(kk, (0, 0, 0, 0, pad_len, 0))  # left pad
                    vv = F.pad(vv, (0, 0, 0, 0, pad_len, 0))
                k_padded.append(kk)
                v_padded.append(vv)
            
            k = torch.cat(k_padded, dim=0)  # [B, max_cache_len, n_kv_heads, head_dim]
            v = torch.cat(v_padded, dim=0)

        # GQA expansion
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        # Attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(Seq > 1 and start_pos == 0))
        
        return self.o_proj(output.transpose(1, 2).contiguous().view(B, Seq, -1))

class MLP(nn.Module):
    """SwiGLU MLP."""
    
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer layer."""
    
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.self_attn.layer_idx = layer_id  # Set layer index for KV cache
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_caches: Optional[List[Dict]],  # List of caches per batch item
    ) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, kv_caches)
        return h + self.mlp(self.post_attention_layernorm(h))

class LlamaStage(nn.Module):
    """Pipeline stage with batched KV cache support."""
    
    def __init__(self, hf_config, rank: int, device_map: Dict[int, List[int]]):
        super().__init__()
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.my_layers = device_map[rank]
        self.is_first = (rank == 0)
        self.is_last = (rank == max(device_map.keys()))
        
        # Create layers with correct local indices
        self.layers = nn.ModuleList()
        for i, global_idx in enumerate(self.my_layers):
            block = TransformerBlock(hf_config, i)  # Use local index for KV cache
            block.self_attn.layer_idx = i
            self.layers.append(block)
        
        if self.is_first:
            self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size)
        
        if self.is_last:
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
        kv_caches: List[List[Dict]],  # [batch_size][layer_idx] -> {'k': ..., 'v': ...}
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Embedding
        h = self.embed_tokens(x) if self.is_first and x.dtype == torch.long else x
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(h, self.cos, self.sin, mask, kv_caches)
        
        # Output projection
        if self.is_last and self.lm_head is not None:
            h = self.lm_head(self.norm(h))
        
        return h

def load_weights_into_stage(
    model: LlamaStage,
    model_path: str,
    hf_config,
    dtype: torch.dtype,
):    
    def map_key(key: str) -> Optional[str]:
        """Map HF weight key to local model key."""
        key = key.replace("model.", "")
        
        # Transformer layers
        if key.startswith("layers."):
            parts = key.split(".")
            layer_idx = int(parts[1])
            if layer_idx in model.my_layers:
                local_idx = model.my_layers.index(layer_idx)
                return f"layers.{local_idx}." + ".".join(parts[2:])
        
        # Embeddings (first stage)
        if model.is_first and "embed_tokens" in key:
            return "embed_tokens.weight"
        
        # Output head (last stage)
        if model.is_last:
            if "norm.weight" in key:
                return "norm.weight"
            if "lm_head" in key:
                return "lm_head.weight"
            if getattr(hf_config, "tie_word_embeddings", False) and "embed_tokens" in key:
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
    
    print(f"[Rank {model.rank}] Loaded weights for layers {model.my_layers}")
