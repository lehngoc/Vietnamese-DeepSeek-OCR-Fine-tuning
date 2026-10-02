# IMPORTANT: Unsloth must be imported before transformers / torch to patch optimizations
import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from typing import Any, Dict, List, Optional, Tuple

try:
    import unsloth
    from unsloth import FastVisionModel
except Exception:
    FastVisionModel = None  # Fallback when running in environments without GPU acceleration

import torch
from transformers import AutoModel


def setup_env_variables() -> None:
    """
    Configure PyTorch and Unsloth memory allocation variables.
    """
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_base_model_and_tokenizer(
    model_config: Dict[str, Any]
) -> Tuple[Any, Any]:
    """
    Load base DeepSeek-OCR model and tokenizer using Unsloth FastVisionModel.
    
    Args:
        model_config (Dict[str, Any]): Dictionary containing:
            - model_name_or_path (str)
            - load_in_4bit (bool)
            - trust_remote_code (bool)
            - unsloth_force_compile (bool)
            - use_gradient_checkpointing (str or bool)
            
    Returns:
        Tuple[Any, Any]: (model, tokenizer)
    """
    setup_env_variables()
    
    if FastVisionModel is None:
        raise RuntimeError(
            "FastVisionModel from Unsloth is not available. Please ensure you are running in an environment with a supported GPU."
        )

    model_name_or_path = model_config.get("model_name_or_path", "unsloth/DeepSeek-OCR")
    load_in_4bit = model_config.get("load_in_4bit", False)
    trust_remote_code = model_config.get("trust_remote_code", True)
    unsloth_force_compile = model_config.get("unsloth_force_compile", True)
    use_gradient_checkpointing = model_config.get("use_gradient_checkpointing", "unsloth")

    print(f"[*] Loading base model from: {model_name_or_path} (4-bit: {load_in_4bit})")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        auto_model=AutoModel,
        trust_remote_code=trust_remote_code,
        unsloth_force_compile=unsloth_force_compile,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    
    return model, tokenizer


def apply_peft_lora(
    model: Any,
    lora_config: Dict[str, Any],
) -> Any:
    """
    Attach Low-Rank Adaptation (LoRA) adapters to specified linear projection layers.
    
    Args:
        model (Any): Base model instance.
        lora_config (Dict[str, Any]): Dictionary containing:
            - r (int): LoRA rank
            - lora_alpha (int): Scaling factor
            - lora_dropout (float): Dropout probability
            - bias (str): Bias training mode
            - random_state (int): Seed
            - use_rslora (bool): Rank-stabilized LoRA flag
            - target_modules (List[str]): Linear projection layer names
            
    Returns:
        Any: Model with LoRA adapters attached and set for training.
    """
    if FastVisionModel is None:
        raise RuntimeError("FastVisionModel is not available. A GPU environment is required.")

    r = lora_config.get("r", 16)
    lora_alpha = lora_config.get("lora_alpha", 16)
    lora_dropout = lora_config.get("lora_dropout", 0.0)
    bias = lora_config.get("bias", "none")
    random_state = lora_config.get("random_state", 3407)
    use_rslora = lora_config.get("use_rslora", False)
    target_modules = lora_config.get(
        "target_modules",
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print(f"[*] Applying LoRA: r={r}, alpha={lora_alpha}, target_modules={target_modules}")

    model = FastVisionModel.get_peft_model(
        model,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=None,
    )

    # Enable Unsloth fast training mode
    FastVisionModel.for_training(model)

    return model


def setup_model_and_lora(
    model_config: Dict[str, Any],
    lora_config: Dict[str, Any],
) -> Tuple[Any, Any]:
    """
    Convenience function to load base model, tokenizer, and apply LoRA adapters in one step.
    
    Args:
        model_config (Dict[str, Any]): Model configuration.
        lora_config (Dict[str, Any]): LoRA configuration.
        
    Returns:
        Tuple[Any, Any]: (peft_model, tokenizer)
    """
    model, tokenizer = load_base_model_and_tokenizer(model_config)
    model = apply_peft_lora(model, lora_config)
    return model, tokenizer
