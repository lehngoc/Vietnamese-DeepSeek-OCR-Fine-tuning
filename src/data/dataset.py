import json
import os
from typing import Dict, List, Any, Optional
from PIL import Image


def load_json_labels(json_path: str) -> Dict[str, str]:
    """
    Load ground-truth labels from a JSON file.
    
    Args:
        json_path (str): Path to JSON file containing {"img_name.jpg": "ground_truth_text"}.
        
    Returns:
        Dict[str, str]: Dictionary mapping image filename to text label.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Label file not found at: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels


def convert_dataset_to_conversation(
    raw_labels: Dict[str, str],
    images_dir: str,
    instruction: str = "<image>\nFree OCR. ",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Format raw image-text pairs into multi-turn conversation format required by DeepSeek-OCR.
    
    Format:
        [
            {
                "messages": [
                    {
                        "role": "<|User|>",
                        "content": "<image>\nFree OCR. ",
                        "images": [<PIL.Image.Image>]
                    },
                    {
                        "role": "<|Assistant|>",
                        "content": "Text transcription..."
                    }
                ]
            },
            ...
        ]
        
    Args:
        raw_labels (Dict[str, str]): Mapping from image filename to transcription.
        images_dir (str): Root directory containing the image files.
        instruction (str): Text prompt preceding the image token.
        max_samples (Optional[int]): Limit number of samples to load (useful for testing/debugging).
        
    Returns:
        List[Dict[str, Any]]: Formatted conversation samples ready for DataCollator.
    """
    dataset_samples = []
    
    for idx, (img_name, text_content) in enumerate(raw_labels.items()):
        if max_samples is not None and idx >= max_samples:
            break
            
        image_path = os.path.join(images_dir, img_name)
        if not os.path.exists(image_path):
            print(f"[Warning] Image not found at {image_path}, skipping sample.")
            continue
            
        try:
            img_obj = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to open image {image_path}: {e}, skipping.")
            continue
            
        conversation = [
            {
                "role": "<|User|>",
                "content": instruction,
                "images": [img_obj],
            },
            {
                "role": "<|Assistant|>",
                "content": text_content,
            },
        ]
        
        dataset_samples.append({"messages": conversation})
        
    return dataset_samples


def load_and_prepare_dataset(
    json_path: str,
    images_dir: str,
    instruction: str = "<image>\nFree OCR. ",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Helper function to load labels from JSON and build the conversation dataset in one step.
    
    Args:
        json_path (str): Path to JSON label file.
        images_dir (str): Directory containing image files.
        instruction (str): Prompt instruction string.
        max_samples (Optional[int]): Max number of samples to load.
        
    Returns:
        List[Dict[str, Any]]: Prepared dataset list.
    """
    labels = load_json_labels(json_path)
    return convert_dataset_to_conversation(
        raw_labels=labels,
        images_dir=images_dir,
        instruction=instruction,
        max_samples=max_samples,
    )
