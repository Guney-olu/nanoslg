"""
Configuration management and model registry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


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
    device_map: Dict[int, List[int]] = None  
    chat_template: str = "llama3"       
    
    def __post_init__(self):
        if self.device_map is None:
            self.device_map = {0: list(range(0, 14)), 1: list(range(14, 28))}
    
    @property
    def world_size(self) -> int:
        return len(self.device_map)
    
    @property
    def total_layers(self) -> int:
        return sum(len(layers) for layers in self.device_map.values())


_MODEL_REGISTRY: Dict[str, ModelConfig] = {}


def register_model(config: ModelConfig):
    """Register a model configuration."""
    _MODEL_REGISTRY[config.name] = config
    print(f"[Registry] Registered model: {config.name}")


def get_model_config(name: str) -> ModelConfig:
    """Get a registered model configuration."""
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available}")
    return _MODEL_REGISTRY[name]


def list_models() -> List[str]:
    """List all registered models."""
    return list(_MODEL_REGISTRY.keys())


# Llama 3.2 3B - 2 GPU split
# register_model(ModelConfig(
#     name="llama-3.2-3b",
#     path="/home/Aryan/models/Llama-3.2-3B-Instruct",
#     device_map={0: list(range(0, 14)), 1: list(range(14, 28))},
#     chat_template="llama3",
# ))

# register_model(ModelConfig(
#     name="llama-3.1-8b",
#     path="/path/to/Llama-3.1-8B-Instruct",
#     device_map={0: list(range(0, 16)), 1: list(range(16, 32))},
#     chat_template="llama3",
# ))


# CHAT TEMPLATES
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
    "alpaca": {
        "message": "### {role}:\n{content}\n\n",
        "assistant_start": "### Assistant:\n",
        "eos_token": "",
    },
}


def format_chat(messages: List[dict], template_name: str = "llama3") -> str:
    """Format messages using specified chat template."""
    template = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["llama3"])
    
    prompt = ""
    for msg in messages:
        prompt += template["message"].format(role=msg["role"], content=msg["content"])
    prompt += template["assistant_start"]
    
    return prompt