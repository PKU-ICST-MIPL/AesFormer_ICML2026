import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from PIL import Image
from tqdm import tqdm
import clip
import numpy as np

INPUT_DIR = "AesFormer_ICML2026/test/Stage2/reconstructed_images"
OUTPUT_PATH = "AesFormer_ICML2026/test/Stage2/evaluate/LAION-V2_Score.json"

MLP_CKPT_PATH = "AesFormer_ICML2026/pretrained_weights/sac+logos+ava1-l14-linearMSE.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(pl.LightningModule):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


@torch.no_grad()
def score_one_image(
    img_path,
    clip_model,
    clip_preprocess,
    aesthetic_model,
    device
):
    pil_image = Image.open(img_path)
    image = clip_preprocess(pil_image).unsqueeze(0).to(device)

    # CLIP image embedding
    image_features = clip_model.encode_image(image)
    image_features = image_features.cpu().numpy()

    # Normalize
    image_features = normalized(image_features)

    # Aesthetic score
    image_features = torch.from_numpy(image_features).to(device).float()
    score = aesthetic_model(image_features)

    print(score.item())
    return score.item()


def process_images(
    input_dir,
    clip_model,
    clip_preprocess,
    aesthetic_model,
    device,
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
        score = score_one_image(
            img_path,
            clip_model,
            clip_preprocess,
            aesthetic_model,
            device
        )
        score = round(score, 3)
        img_name = os.path.basename(img_path)
        all_results[img_name] = score

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)


def main():
    # 1. CLIP
    print("Loading CLIP ViT-L/14...")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
    clip_model.eval()

    # 2. LAION aesthetic MLP
    print("Loading LAION aesthetic MLP...")
    aesthetic_model = MLP(input_size=768)
    state_dict = torch.load(MLP_CKPT_PATH)
    aesthetic_model.load_state_dict(state_dict)
    aesthetic_model.to(DEVICE)
    aesthetic_model.eval()
    
    all_results = {}
    process_images(
        INPUT_DIR,
        clip_model,
        clip_preprocess,
        aesthetic_model,
        DEVICE,
        all_results
    )
    print(f"Scoring completed. Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
