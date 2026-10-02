import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import time
from typing import Dict, List

from tqdm import tqdm

try:
    import unsloth
    from unsloth import FastVisionModel
except Exception:
    FastVisionModel = None

import torch
from transformers import AutoModel

from src.data.dataset import load_json_labels
from src.utils.metrics import calculate_cer, calculate_wer, compute_metrics_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-OCR performance on Vietnamese handwriting test dataset"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="data/test.json",
        help="Path to test set labels JSON file",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/final_model",
        help="Path to model checkpoint to evaluate",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\nFree OCR.",
        help="Prompt used for evaluation",
    )
    parser.add_argument(
        "--output_log",
        type=str,
        default="eval_results.json",
        help="Path to save detailed predictions and metrics log",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of test samples to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if FastVisionModel is None:
        raise RuntimeError("FastVisionModel is not available. A GPU environment is required for evaluation.")

    print("=" * 70)
    print("📊 EVALUATING DEEPSEEK-OCR BENCHMARK (CER / WER)")
    print("=" * 70)

    # 1. Load Labels
    print(f"[*] Loading test labels from: {args.test_json}")
    labels = load_json_labels(args.test_json)
    items = list(labels.items())
    if args.max_samples:
        items = items[: args.max_samples]
    print(f"[*] Total samples to evaluate: {len(items)}")

    # 2. Load Model
    print(f"[*] Loading model from: {args.model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
    )
    FastVisionModel.for_inference(model)

    predictions: List[str] = []
    references: List[str] = []
    detailed_results = []

    print("\n[*] Running inference on evaluation dataset...")
    start_time = time.time()

    for idx, (img_name, ref_text) in enumerate(tqdm(items, desc="Evaluating")):
        img_path = os.path.join(args.images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[Warning] Missing image: {img_path}, skipping.")
            continue

        try:
            pred_text = model.infer(
                tokenizer,
                prompt=args.prompt,
                image_file=img_path,
                output_path=None,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False,
            )
            pred_text = str(pred_text).strip() if pred_text else ""
        except Exception as e:
            print(f"[Error] Failed inference on {img_name}: {e}")
            pred_text = ""

        ref_clean = ref_text.strip()
        sample_cer = calculate_cer(pred_text, ref_clean)
        sample_wer = calculate_wer(pred_text, ref_clean)

        predictions.append(pred_text)
        references.append(ref_clean)

        detailed_results.append(
            {
                "image": img_name,
                "ground_truth": ref_clean,
                "prediction": pred_text,
                "cer": sample_cer,
                "wer": sample_wer,
            }
        )

    total_time = time.time() - start_time
    summary = compute_metrics_summary(predictions, references)

    print("\n" + "=" * 70)
    print("📈 FINAL EVALUATION REPORT")
    print("=" * 70)
    print(f"  • Total Evaluated Samples: {summary['num_samples']}")
    print(f"  • Character Error Rate (CER): {summary['cer_percent']:.2f}%")
    print(f"  • Word Error Rate (WER):      {summary['wer_percent']:.2f}%")
    print(f"  • Total Evaluation Time:      {total_time:.2f}s ({total_time / len(predictions):.3f}s / image)")
    print("=" * 70)

    # Save detailed JSON log
    log_data = {
        "summary": summary,
        "total_time_seconds": total_time,
        "details": detailed_results,
    }
    with open(args.output_log, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"[+] Detailed evaluation log saved to: {args.output_log}")


if __name__ == "__main__":
    main()
