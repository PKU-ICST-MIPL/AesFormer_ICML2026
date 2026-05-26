import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from PIL import Image
from tqdm import tqdm
import requests
from transformers import AutoModelForCausalLM

INPUT_DIR = "AesFormer_ICML2026/test/Stage2/reconstructed_images"
OUTPUT_PATH = "AesFormer_ICML2026/test/Stage2/evaluate/Q-align_Score.json"

def process_images(
    model,
    input_dir,
    all_results
):
    img_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ]

    if not img_paths:
        print(f"[WARN] No PNG files: {input_dir}")
        return

    for img_path in tqdm(img_paths, desc=f"Processing {input_dir}"):
        score = model.score([Image.open(img_path)], task_="aesthetics", input_="image") # task_ : quality | aesthetics; # input_: image | video
        score = score[0]
        score = score.item()
        score = round(score, 5)
        img_name = os.path.basename(img_path)
        all_results[img_name] = score

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)


def main():
    model = AutoModelForCausalLM.from_pretrained("/home/dutianxiang/67/ckpt/q-future_one-align", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    all_results = {}
    process_images(
        model,
        INPUT_DIR,
        all_results
    )
    print(f"Scoring completed. Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
