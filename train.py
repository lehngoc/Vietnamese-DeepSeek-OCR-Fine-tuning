import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
from typing import Any, Dict

try:
    import unsloth
    from unsloth import FastVisionModel, is_bf16_supported
except Exception:
    FastVisionModel = None
    def is_bf16_supported():
        return False

import torch
import yaml
from transformers import Trainer, TrainingArguments

from src.data import DeepSeekOCRDataCollator, load_and_prepare_dataset
from src.models import setup_model_and_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSeek-OCR for Vietnamese Handwriting Recognition using LoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of train/val samples for quick testing",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides if provided
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.epochs:
        config["training"]["num_train_epochs"] = args.epochs
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate

    print("=" * 70)
    print("🚀 STARTING DEEPSEEK-OCR VIETNAMESE FINE-TUNING PIPELINE")
    print("=" * 70)

    # 1. Load Model & Tokenizer with PEFT LoRA
    print("\n[Step 1/5] Initializing Model & LoRA Adapters...")
    model, tokenizer = setup_model_and_lora(
        model_config=config["model"],
        lora_config=config["lora"],
    )

    # 2. Load & Prepare Datasets
    print("\n[Step 2/5] Preparing Train and Validation Datasets...")
    data_cfg = config["data"]
    
    train_dataset = load_and_prepare_dataset(
        json_path=data_cfg["train_json"],
        images_dir=data_cfg["images_dir"],
        instruction=data_cfg.get("instruction", "<image>\nFree OCR. "),
        max_samples=args.max_samples,
    )
    print(f"  -> Loaded {len(train_dataset)} training samples.")

    val_dataset = None
    if os.path.exists(data_cfg.get("val_json", "")):
        val_dataset = load_and_prepare_dataset(
            json_path=data_cfg["val_json"],
            images_dir=data_cfg["images_dir"],
            instruction=data_cfg.get("instruction", "<image>\nFree OCR. "),
            max_samples=args.max_samples,
        )
        print(f"  -> Loaded {len(val_dataset)} validation samples.")

    # 3. Create Custom Data Collator
    print("\n[Step 3/5] Setting up DeepSeekOCR Data Collator (Gundam Cropping)...")
    data_collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=data_cfg.get("image_size", 640),
        base_size=data_cfg.get("base_size", 1024),
        crop_mode=data_cfg.get("crop_mode", True),
        train_on_responses_only=data_cfg.get("train_on_responses_only", True),
    )

    # 4. Setup Training Arguments
    print("\n[Step 4/5] Configuring Trainer Arguments...")
    train_cfg = config["training"]
    output_dir = train_cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    use_bf16 = is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        eval_accumulation_steps=train_cfg.get("eval_accumulation_steps", 1),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.001),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=train_cfg.get("warmup_steps", 10),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        optim=train_cfg.get("optim", "adamw_8bit"),
        seed=train_cfg.get("seed", 3407),
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps") if val_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 25) if val_dataset else None,
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 25),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True) if val_dataset else False,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss") if val_dataset else None,
        greater_is_better=train_cfg.get("greater_is_better", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        report_to=train_cfg.get("report_to", "none"),
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    # 5. Execute Training
    print("\n[Step 5/5] Executing Training Loop...")
    start_time = time.time()
    trainer_stats = trainer.train()
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"🎉 Training Completed in {total_time:.2f}s ({total_time / 60:.2f} mins)")
    print(f"   Train Loss: {trainer_stats.training_loss:.4f}")
    print("=" * 70)

    # Save final model and tokenizer
    final_save_dir = os.path.join(output_dir, "final_model")
    print(f"[*] Saving final PEFT model and tokenizer to: {final_save_dir}")
    trainer.save_model(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print("✅ Model successfully saved!")


if __name__ == "__main__":
    main()
