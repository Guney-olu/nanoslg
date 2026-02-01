"""
Configuration with parallel mode support.
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
    """Configuration for a model deployment."""
    name: str
    path: str
    dtype: str = "bfloat16"
    max_seq_len: int = 8192
    
    # Parallelism settings
    parallel_mode: str = "pipeline"  # "pipeline", "tensor", "hybrid"
    tp_size: int = 1
    pp_size: int = 1
    
    # Layer distribution (for PP/hybrid)
    device_map: Dict[int, List[int]] = None
    
    chat_template: str = "llama3"
    
    def __post_init__(self):
        # Auto-configure device_map if not provided
        if self.device_map is None and self.parallel_mode in ("pipeline", "hybrid"):
            total_layers = self._infer_total_layers()
            self.device_map = {}
            for pp_rank in range(self.pp_size):
                self.device_map[pp_rank] = get_layers_for_pp_rank(
                    total_layers, self.pp_size, pp_rank
                )
    
    def _infer_total_layers(self) -> int:
        """Infer total layers from model path (common configurations)."""
        name_lower = self.name.lower()
        if "3b" in name_lower or "3.2-3b" in name_lower:
            return 28
        elif "8b" in name_lower or "3.1-8b" in name_lower:
            return 32
        elif "70b" in name_lower:
            return 80
        return 32  # Default
    
    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size
    
    def get_parallel_config(self, total_layers: int = None) -> ParallelConfig:
        """Create ParallelConfig from ModelConfig."""
        mode = ParallelMode(self.parallel_mode)
        
        config = ParallelConfig(
            mode=mode,
            world_size=self.world_size,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
        )
        
        # Set layer splits for PP modes
        if mode in (ParallelMode.PIPELINE, ParallelMode.HYBRID):
            if self.device_map:
                config.pp_layer_splits = self.device_map
            elif total_layers:
                config.pp_layer_splits = {}
                for pp_rank in range(self.pp_size):
                    config.pp_layer_splits[pp_rank] = get_layers_for_pp_rank(
                        total_layers, self.pp_size, pp_rank
                    )
        
        return config


# Model Registry
_MODEL_REGISTRY: Dict[str, ModelConfig] = {}


def register_model(config: ModelConfig):
    _MODEL_REGISTRY[config.name] = config
    print(f"[Registry] Registered: {config.name} ({config.parallel_mode} mode)")


def get_model_config(name: str) -> ModelConfig:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]


def list_models() -> List[str]:
    return list(_MODEL_REGISTRY.keys())



# Pipeline Parallel (2 GPUs)
# register_model(ModelConfig(
#     name="llama-3.2-3b-pp",
#     path="/path/to/Llama-3.2-3B-Instruct",
#     parallel_mode="pipeline",
#     pp_size=2,
#     tp_size=1,
#     device_map={0: list(range(14)), 1: list(range(14, 28))},
# ))

# Tensor Parallel (2 GPUs)
register_model(ModelConfig(
    name="llama-3.1-8b-tp",
    path="./models/Llama-3.1-8B-Instruct",
    parallel_mode="tensor",
    tp_size=2,
    pp_size=1,
))

# Hybrid (4 GPUs: 2 TP x 2 PP)
# register_model(ModelConfig(
#     name="llama-3.1-8b-hybrid",
#     path="/path/to/Llama-3.1-8B-Instruct",
#     parallel_mode="hybrid",
#     tp_size=2,
#     pp_size=2,
#     device_map={0: list(range(16)), 1: list(range(16, 32))},
# ))


# Chat Templates
CHAT_TEMPLATES = {
    "llama3": {
        "message": "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "eos_token": "<|eot_id|>",
    },
    "chatml": {
        "message": "<|im_start|>{role}\n{content}<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "eos_token": "<|im_end|}",
    },
}


def format_chat(messages: List[dict], template_name: str = "llama3") -> str:
    template = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["llama3"])
    prompt = ""
    for msg in messages:
        prompt += template["message"].format(role=msg["role"], content=msg["content"])
    prompt += template["assistant_start"]
    return prompt