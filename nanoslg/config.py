"""
config for nanoslg server.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

from .parallel import ParallelMode, ParallelConfig, get_layers_for_pp_rank

STORE_PATH = "/tmp/nanoslg_store"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@dataclass
class ModelConfig:
    name: str
    path: str
    dtype: str = "bfloat16"
    max_seq_len: int = 8192

    parallel_mode: str = "pipeline"
    tp_size: int = 1
    pp_size: int = 1
    device_map: Dict[int, List[int]] = None

    # KV cache settings
    page_size: int = 16
    max_kv_pages: int = 0              # 0 = auto

    kv_memory_fraction: float = 0.30
    kv_backend: str = "auto"      

    enable_prefix_caching: bool = True

    chat_template: str = "llama3"

    def __post_init__(self):
        if self.device_map is None and self.parallel_mode in ("pipeline", "hybrid"):
            total = self._infer_total_layers()
            self.device_map = {}
            for pp in range(self.pp_size):
                self.device_map[pp] = get_layers_for_pp_rank(total, self.pp_size, pp)

    def _infer_total_layers(self) -> int:
        n = self.name.lower()
        if "3b" in n:   return 28
        if "8b" in n:   return 32
        if "70b" in n:  return 80
        if "14b" in n:  return 40   # Qwen2-14B
        if "7b" in n:   return 32   # Qwen2-7B, Mistral-7B
        return 32

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size

    def get_parallel_config(self, total_layers: int = None) -> ParallelConfig:
        mode = ParallelMode(self.parallel_mode)
        config = ParallelConfig(mode=mode, world_size=self.world_size,
                                tp_size=self.tp_size, pp_size=self.pp_size)
        if mode in (ParallelMode.PIPELINE, ParallelMode.HYBRID):
            if self.device_map:
                config.pp_layer_splits = self.device_map
            elif total_layers:
                config.pp_layer_splits = {}
                for pp in range(self.pp_size):
                    config.pp_layer_splits[pp] = get_layers_for_pp_rank(total_layers, self.pp_size, pp)
        return config


_MODEL_REGISTRY: Dict[str, ModelConfig] = {}

def register_model(c: ModelConfig):
    _MODEL_REGISTRY[c.name] = c
    print(f"[Registry] {c.name} ({c.parallel_mode})")

def get_model_config(name: str) -> ModelConfig:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"'{name}' not found. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]

def list_models() -> List[str]:
    return list(_MODEL_REGISTRY.keys())


register_model(ModelConfig(
    name="llama-3.1-8b-tp",
    path="./models/Llama-3.1-8B-Instruct",
    parallel_mode="tensor", tp_size=2, pp_size=1,
    page_size=16, enable_prefix_caching=True,
))

# register_model(ModelConfig(
#     name="llama-3.2-3b",
#     path="./models/Llama-3.2-3B-Instruct",
#     dtype="bfloat16",
#     max_seq_len=8192,
#     parallel_mode="tensor",   # or "tensor" if you prefer
#     tp_size=2,
#     page_size=16,
#     kv_memory_fraction=0.30,
#     enable_prefix_caching=True,
#     chat_template="llama3",
# ))


# ── Chat Templates ──

CHAT_TEMPLATES = {
    "llama3": {
        "message": "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "chatml": {     
        "message": "<|im_start|>{role}\n{content}<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
    },
    "chatglm": {
        "message": "[gMASK]sop<|{role}|>\n{content}",
        "assistant_start": "<|assistant|>\n",
    },
}


def format_chat(messages: List[dict], template_name: str = "llama3") -> str:
    t = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["llama3"])
    prompt = ""
    for m in messages:
        prompt += t["message"].format(role=m["role"], content=m["content"])
    prompt += t["assistant_start"]
    return prompt