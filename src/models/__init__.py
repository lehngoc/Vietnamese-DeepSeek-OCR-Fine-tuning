"""
Model and PEFT setup modules for DeepSeek-OCR
"""
from src.models.peft_setup import (
    load_base_model_and_tokenizer,
    apply_peft_lora,
    setup_model_and_lora,
)

__all__ = [
    "load_base_model_and_tokenizer",
    "apply_peft_lora",
    "setup_model_and_lora",
]
