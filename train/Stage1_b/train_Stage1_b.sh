#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH=/path/to/your/Stage1_a_model

python3 -m verl.trainer.main \
    config=AesFormer_ICML2026/train/Stage1_b/train_Stage1_b.yaml \
    data.train_files=AesFormer_ICML2026/AesRecon_dataset/jsons/train/Stage1_b/train_Stage1_b.json \
    data.val_files=/path/to/your/val_files.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=AesFormer_Stage1_b \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=8 \
    data.filter_overlong_prompts=false \
