import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from tqdm import tqdm

import sys
sys.path.append(".../ArtiMuse/src")
sys.path.append(".../ArtiMuse/src/artimuse")
from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


INPUT_DIR = "AesFormer_ICML2026/test/Stage2/reconstructed_images"
OUTPUT_PATH = "AesFormer_ICML2026/test/Stage2/evaluate/ArtiMuse_Score.json"


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0)  # [1, C, H, W]


@torch.no_grad()
def score_one(model, tokenizer, pixel_values, generation_config, device="cuda:0"):
    pixel_values = pixel_values.to(torch.bfloat16).to(device)
    score = model.score(device, tokenizer, pixel_values, generation_config)
    return score


def process_images(model, tokenizer, input_dir, generation_config, device, all_results):
    img_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ]

    if not img_paths:
        return {}

    for idx, img_path in enumerate(tqdm(img_paths, desc=f"Processing {input_dir}"), 1):
        try:
            pixel_values = load_image(img_path)
            score = score_one(model, tokenizer, pixel_values, generation_config, device=device)
            
            score = round(score, 0)
            
            img_name = os.path.basename(img_path)
            all_results[img_name] = score

        except Exception as e:
            img_name = os.path.basename(img_path)
            print(f"[ERROR] {img_name}:{e}")

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Scoring completed. Results saved to: {OUTPUT_PATH}")


def main(args):
    model_path = "AesFormer_ICML2026/pretrained_weights/ArtiMuse"

    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    generation_config = dict(
        max_new_tokens=8192,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    all_results = {}
    process_images(model, tokenizer, INPUT_DIR, generation_config, device=args.device, all_results=all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ArtiMuse",
                        help="Model directory name (under checkpoints/)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device, e.g. cuda:0 / cuda:1 or cpu")
    args = parser.parse_args()
    main(args)