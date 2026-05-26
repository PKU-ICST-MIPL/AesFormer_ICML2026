#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

cd AesFormer_ICML2026/musubi-tuner

DATASET_CONFIG="AesFormer_ICML2026/train/Stage2/dataset.toml"
TEXT_ENCODER="AesFormer_ICML2026/pretrained_weights/qwen_2.5_vl_7b.safetensors"

python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config $DATASET_CONFIG \
    --text_encoder $TEXT_ENCODER \
    --batch_size 1 \
    --model_version edit-2511 \
