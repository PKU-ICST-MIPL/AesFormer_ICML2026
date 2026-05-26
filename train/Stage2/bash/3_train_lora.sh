#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd AesFormer_ICML2026/musubi-tuner

DIT="AesFormer_ICML2026/pretrained_weights/qwen_image_edit_2511_bf16.safetensors"
VAE="AesFormer_ICML2026/pretrained_weights/qwen_image_vae.safetensors"
TEXT_ENCODER="AesFormer_ICML2026/pretrained_weights/qwen_2.5_vl_7b.safetensors"
DATASET_CONFIG="AesFormer_ICML2026/train/Stage2/dataset.toml"

OUTPUT="/path/to/your/Stage2_model/output_dir"
NAME="edit2511_lora"

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/qwen_image_train_network.py \
  --dit $DIT \
  --vae $VAE \
  --text_encoder $TEXT_ENCODER \
  \
  --dataset_config $DATASET_CONFIG \
  \
  --model_version edit-2511 \
  \
  --network_module networks.lora_qwen_image \
  --network_dim 16 \
  \
  --timestep_sampling qwen_shift \
  \
  --sdpa \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --gradient_checkpointing_cpu_offload \
  \
  --optimizer_type adamw8bit \
  --learning_rate 1e-5 \
  \
  --max_train_epochs 2 \
  --save_every_n_epochs 1 \
  --seed 42 \
  \
  --max_data_loader_n_workers 2 \
  --persistent_data_loader_workers \
  \
  --output_dir $OUTPUT \
  --output_name $NAME
