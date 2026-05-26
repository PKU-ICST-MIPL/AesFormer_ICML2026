#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

cd AesFormer_ICML2026/musubi-tuner

DATASET_CONFIG="AesFormer_ICML2026/train/Stage2/dataset.toml"
VAE="AesFormer_ICML2026/pretrained_weights/qwen_image_vae.safetensors"

python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config $DATASET_CONFIG \
    --vae $VAE \
    --model_version edit-2511 \
