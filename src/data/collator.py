import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence


# ------------------------------------------------------------------------------
# Image Preprocessing & Helper Transforms (Gundam Dynamic Cropping)
# ------------------------------------------------------------------------------
class BasicImageTransform:
    """
    Standard image transformation: Converts PIL Image to Tensor and normalizes.
    """
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        
        transforms = [T.ToTensor()]
        if normalize:
            transforms.append(T.Normalize(mean=mean, std=std))
        self.transform = T.Compose(transforms)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """
    Find the closest grid aspect ratio to fit the input image dimensions.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 2,
    max_num: int = 9,
    image_size: int = 640,
    use_thumbnail: bool = False,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    Dynamically crop a high-resolution image into a grid of local patches (Gundam method).
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def text_encode(tokenizer: Any, text: str, bos: bool = False, eos: bool = False) -> List[int]:
    """
    Tokenize input string into list of token IDs with optional BOS/EOS tokens.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        tokens = [bos_id] + tokens
    if eos:
        tokens = tokens + [eos_id]
    return tokens

# ------------------------------------------------------------------------------
# DeepSeek-OCR Custom Data Collator
# ------------------------------------------------------------------------------
@dataclass
class DeepSeekOCRDataCollator:
    """
    Custom DataCollator for DeepSeek-OCR fine-tuning.
    
    Responsibilities:
    1. Multi-scale vision processing: global view (1024x1024) + dynamic local patches (640x640).
    2. Visual token placeholder insertion (<image> token expansion).
    3. Prompt Masking: Masks user instruction and vision tokens with -100 so loss is computed
       strictly on the assistant's transcription response.
    4. Batch creation: Dynamic sequence padding and tensor stacking for GPU execution.
    """
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.train_on_responses_only = train_on_responses_only
        self.dtype = getattr(model, "dtype", torch.float16)

        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True,
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0

    def deserialize_image(self, image_data: Any) -> Image.Image:
        """
        Convert raw image input (bytes dict or PIL Image) to standard PIL RGB image.
        """
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, dict) and "bytes" in image_data:
            return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(self, image: Image.Image, crop_ratio: Tuple[int, int]) -> int:
        """
        Calculate number of vision tokens generated for the given image and crop ratio.
        """
        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)
        else:
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(
        self, image: Image.Image
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[int]], List[int], Tuple[int, int]]:
        """
        Process an image into global view, local patch crops, and vision token placeholders.
        """
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num=2, max_num=9, image_size=self.image_size, use_thumbnail=False
                )

            # Global view with padding
            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Local patch views (crops)
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img).to(self.dtype))

            # Build token placeholder sequence
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = (
                [self.image_token_id] * num_queries_base + [self.image_token_id]
            ) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += (
                    [self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]
                ) * (num_queries * height_crop_num)

        else:
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            if self.base_size <= 640:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image,
                    (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean),
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a single conversation item into model input tensors.
        """
        images = []
        for message in messages:
            if "images" in message and message["images"]:
                for img_data in message["images"]:
                    if img_data is not None:
                        images.append(self.deserialize_image(img_data))

        if not images:
            raise ValueError("No images found in sample. All samples must contain an image.")

        tokenized_str = []
        images_seq_mask = []
        images_list, images_crop_list, images_spatial_crop = [], [], []

        prompt_token_count = -1
        assistant_started = False
        image_idx = 0

        # Prepend BOS token
        tokenized_str.append(self.bos_id)
        images_seq_mask.append(False)

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "<|Assistant|>":
                if not assistant_started:
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True
                content = f"{content.strip()} {self.tokenizer.eos_token}"

            text_splits = content.split("<image>")

            for i, text_sep in enumerate(text_splits):
                tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                tokenized_str.extend(tokenized_sep)
                images_seq_mask.extend([False] * len(tokenized_sep))

                if i < len(text_splits) - 1:
                    if image_idx >= len(images):
                        raise ValueError("Mismatch: '<image>' tag found without corresponding image object.")

                    image = images[image_idx]
                    img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial_crop)

                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))
                    image_idx += 1

        if image_idx != len(images):
            raise ValueError(f"Mismatch: {len(images)} images provided but only {image_idx} tags processed.")

        if not assistant_started:
            prompt_token_count = len(tokenized_str)

        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)

        return {
            "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "prompt_token_count": prompt_token_count,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate and pad a batch of samples for model forward pass.
        """
        batch_data = []

        for feature in features:
            try:
                processed = self.process_single_sample(feature["messages"])
                batch_data.append(processed)
            except Exception as e:
                print(f"[Warning] Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch.")

        input_ids_list = [item["input_ids"] for item in batch_data]
        images_seq_mask_list = [item["images_seq_mask"] for item in batch_data]
        prompt_token_counts = [item["prompt_token_count"] for item in batch_data]

        # Dynamic padding
        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        images_seq_mask = pad_sequence(
            images_seq_mask_list, batch_first=True, padding_value=False
        )

        labels = input_ids.clone()

        # 1. Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 2. Mask image tokens (model does not predict vision embeddings)
        labels[images_seq_mask] = -100

        # 3. Mask prompt tokens (train strictly on assistant response)
        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        images_batch = []
        for item in batch_data:
            images_batch.append((item["images_crop"], item["images_ori"]))

        images_spatial_crop = torch.cat([item["images_spatial_crop"] for item in batch_data], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }
