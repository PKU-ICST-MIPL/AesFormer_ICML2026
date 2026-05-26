import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
import subprocess
from tqdm import tqdm
import time

PROJECT_ROOT = "AesFormer_ICML2026/musubi-tuner"
SCRIPT = "src/musubi_tuner/qwen_image_generate_image.py"

BENCH_PATH = "AesFormer_ICML2026/test/Stage1/Editing_Actions.json"
POOR_IMAGE_ROOT = "AesFormer_ICML2026/AesRecon_dataset/images/poor_images"
SAVE_ROOT = "AesFormer_ICML2026/test/Stage2/reconstructed_images"
TMP_ROOT = os.path.join(SAVE_ROOT, "_tmp")

LORA_PATH = "/path/to/your/Stage2_model/output_dir"

DIT_PATH = "AesFormer_ICML2026/pretrained_weights/qwen_image_edit_2511_bf16.safetensors"
VAE_PATH = "AesFormer_ICML2026/pretrained_weights/qwen_image_vae.safetensors"
TEXT_ENCODER_PATH = "AesFormer_ICML2026/pretrained_weights/qwen_2.5_vl_7b.safetensors"


def move_and_rename_latest_image(tmp_dir: str, poor_image_path: str, final_dir: str):
    os.makedirs(final_dir, exist_ok=True)

    base = os.path.basename(poor_image_path)      # 5062_poor.jpg
    stem = os.path.splitext(base)[0]              # 5062_poor
    target_path = os.path.join(final_dir, stem + "_APR.png")

    imgs = [
        f for f in os.listdir(tmp_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not imgs:
        raise RuntimeError(f"{tmp_dir} 中未找到生成图片")

    imgs.sort(key=lambda f: os.path.getmtime(os.path.join(tmp_dir, f)))
    src_path = os.path.join(tmp_dir, imgs[-1])

    os.replace(src_path, target_path)
    return target_path


def main():
    os.chdir(PROJECT_ROOT)

    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(TMP_ROOT, exist_ok=True)

    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    for idx, (poor_image_name, prompt) in enumerate(tqdm(bench_data.items(), desc="Qwen-Image-Edit Benchmark")):
        poor_image = os.path.join(POOR_IMAGE_ROOT, poor_image_name)

        base = os.path.basename(poor_image)
        stem = os.path.splitext(base)[0]
        final_dir = SAVE_ROOT
        final_img_path = os.path.join(final_dir, stem + "_APR.png")

        if os.path.exists(final_img_path):
            continue

        tmp_dir = os.path.join(TMP_ROOT, f"{idx:05d}")
        os.makedirs(tmp_dir, exist_ok=True)

        cmd = [
            "python", SCRIPT,
            "--dit", DIT_PATH,
            "--vae", VAE_PATH,
            "--text_encoder", TEXT_ENCODER_PATH,
            "--model_version", "edit-2511",
            "--control_image_path", poor_image,
            "--prompt", prompt,
            "--negative_prompt", " ",
            "--infer_steps", "40",
            "--guidance_scale", "4.0",
            "--resize_control_to_official_size",
            "--image_size", "1440", "1088",
            "--save_path", tmp_dir,
            "--seed", "1234",
            "--lora_weight", LORA_PATH,
            "--lora_multiplier", "1.0",
        ]

        subprocess.run(cmd, check=True)
        time.sleep(3)

        final_dir = SAVE_ROOT
        move_and_rename_latest_image(tmp_dir, poor_image, final_dir)

    print(f"\nBenchmark completed. Processed {len(bench_data)} samples.")


if __name__ == "__main__":
    main()
