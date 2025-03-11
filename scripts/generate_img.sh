#!/bin/bash

## Step 0: Load the environment
# 建议使用本地conda或virtualenv环境路径
source /home/pengrui/mambaforge/bin/activate morphodiff
unset LD_LIBRARY_PATH

## 定义参数（根据本地环境调整路径）
EXPERIMENT="pr_AIVCdiff_morphodiff_inference"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CKPT_PATH="/home/pengrui/work_space_pengrui/project/AIVCdiff/scripts/checkpoint/pr_AIVCdiff/checkpoint-140000"
VAE_PATH="/home/pengrui/work_space_pengrui/huggingface_model/stable-diffusion-v1-4"
GEN_IMG_PATH="$SCRIPT_DIR/generated_imgs/"
NUM_GEN_IMG=5
MODEL_NAME="SD"
MODEL_TYPE="conditional"
TEST_DATA_PATH="/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/adata_test_updated_cleaned.h5ad"

## 生成图像（添加CUDA_VISIBLE_DEVICES指定GPU）
CUDA_VISIBLE_DEVICES=7

/home/pengrui/mambaforge/envs/morphodiff/bin/python -m debugpy --listen 9999 --wait-for-client ../evaluation/generate_img.py \
    --experiment $EXPERIMENT \
    --test_data_path $TEST_DATA_PATH \
    --adata_ctrl_path "/home/pengrui/work_space_pengrui/project/RNA图像合成/1_3_rna_ctrl_data.h5ad" \
    --drug_embed_path "/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/molecule_embeddings_rdkit_所有位点.csv" \
    --model_checkpoint $CKPT_PATH \
    --model_name $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --vae_path $VAE_PATH \
    --gen_img_path $GEN_IMG_PATH \
    --num_imgs $NUM_GEN_IMG
