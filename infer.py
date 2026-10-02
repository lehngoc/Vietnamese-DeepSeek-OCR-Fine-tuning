import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
from typing import Optional
from PIL import Image

try:
    import unsloth
    from unsloth import FastVisionModel
except Exception:
    FastVisionModel = None

import torch
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR Inference on handwritten Vietnamese images using DeepSeek-OCR"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single input image file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to directory containing input images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/final_model",
        help="Path to fine-tuned model checkpoint or base model ('unsloth/DeepSeek-OCR')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\nFree OCR.",
        help="Text prompt for OCR task",
    )
    parser.add_argument(
        "--base_size",
        type=int,
        default=1024,
        help="Base size for global image view",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Size for local patch crops",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save recognition results (JSON or TXT)",
    )
    return parser.parse_args()


def predict_single_image(
    model: any,
    tokenizer: any,
    image_path: str,
    prompt: str = "<image>\nFree OCR.",
    base_size: int = 1024,
    image_size: int = 640,
) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=None,
        base_size=base_size,
        image_size=image_size,
        crop_mode=True,
        save_results=False,
        test_compress=False,
    )
    return res if isinstance(res, str) else str(res)


def main() -> None:
    args = parse_args()

    if not args.image and not args.image_dir:
        print("[!] Please provide either --image or --image_dir.")
        return

    if FastVisionModel is None:
        raise RuntimeError("FastVisionModel is not available. A GPU environment is required for inference.")

    print("=" * 60)
    print("🔍 DEEPSEEK-OCR VIETNAMESE INFERENCE")
    print("=" * 60)
    print(f"[*] Loading model from: {args.model_path}")

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
    )
    FastVisionModel.for_inference(model)

    if args.image:
        print(f"\n[*] Processing image: {args.image}")
        start_time = time.time()
        result = predict_single_image(
            model=model,
            tokenizer=tokenizer,
            image_path=args.image,
            prompt=args.prompt,
            base_size=args.base_size,
            image_size=args.image_size,
        )
        latency = time.time() - start_time

        print("-" * 60)
        print(f"📝 OCR Output: {result}")
        print(f"⏱️ Latency: {latency:.3f}s")
        print("-" * 60)

    elif args.image_dir:
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        image_files = [
            f for f in os.listdir(args.image_dir) if f.lower().endswith(image_extensions)
        ]
        print(f"\n[*] Found {len(image_files)} images in directory: {args.image_dir}")

        results = []
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(args.image_dir, img_file)
            pred = predict_single_image(
                model=model,
                tokenizer=tokenizer,
                image_path=img_path,
                prompt=args.prompt,
                base_size=args.base_size,
                image_size=args.image_size,
            )
            print(f"[{idx+1}/{len(image_files)}] {img_file} -> {pred}")
            results.append(f"{img_file}\t{pred}")

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(results))
            print(f"\n[+] Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
