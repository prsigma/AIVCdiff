#!/bin/bash

# 激活环境
source /home/pengrui/mambaforge/bin/activate morphodiff

## 固定参数 ##
export CKPT_NUMBER=0
export TRAINED_STEPS=0

# 设置训练类型 - "conditional"用于训练MorphoDiff, "naive"用于训练Stable Diffusion
export SD_TYPE="conditional"

# 设置预训练VAE模型路径
export VAE_DIR="/home/pengrui/work_space_pengrui/huggingface_model/stable-diffusion-v1-4"

# 设置日志目录
export LOG_DIR="log/"
# 检查并创建日志目录
if [ ! -d "$LOG_DIR" ]; then                                                                                                                                                                            
  mkdir -p $LOG_DIR
fi

# 设置实验名称
export EXPERIMENT="pr_AIVCdiff"

#验证的药物列表
export VALID_PROMPT="alsterpaullone,cisplatin"

# 设置预训练模型路径
export MODEL_NAME="/home/pengrui/work_space_pengrui/huggingface_model/stable-diffusion-v1-4"

# 设置数据路径
export TRAIN_DATA_PATH="/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/adata_train_cleaned.h5ad"
export VALID_DATA_PATH="/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/adata_valid_updated_cleaned.h5ad"
export ADATA_CTRL_PATH="/home/pengrui/work_space_pengrui/project/RNA图像合成/1_3_rna_ctrl_data.h5ad"
export DRUG_EMBED_PATH="/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/molecule_embeddings_rdkit_所有位点.csv"

# 增加checkpoint编号
export CKPT_NUMBER=$((${CKPT_NUMBER}+1))
export OUTPUT_DIR="checkpoint/${EXPERIMENT}"

echo "Checkpoint编号: $CKPT_NUMBER"
echo "模型目录: $MODEL_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "数据目录: $TRAIN_DIR"
echo "已训练步数: $TRAINED_STEPS"


unset LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=3,4
/home/pengrui/mambaforge/envs/morphodiff/bin/accelerate-launch --num_processes 2 --main_process_port=29873 --mixed_precision="fp16" ../train.py \
    --train_data=$TRAIN_DATA_PATH \
    --valid_data=$VALID_DATA_PATH \
    --ctrl_data=$ADATA_CTRL_PATH \
    --resume_from_checkpoint='latest' \
    --pretrained_model_path=$MODEL_NAME \
    --perturbation_embedding_path=$DRUG_EMBED_PATH \
    --naive_conditional=$SD_TYPE \
    --enable_xformers_memory_efficient_attention \
    --use_ema \
    --train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-05 \
    --lr_scheduler="constant" \
    --rna_loss_types='GUSS' \
    --rna_loss_weight=0.1 \
    --checkpoint_number=$CKPT_NUMBER \
    --output_dir=$OUTPUT_DIR \
    --cache_dir="tmp/" \
    --logging_dir="${LOG_DIR}${EXPERIMENT}_log" \
    --report_to="wandb" \
    --seed=42 \
    --dataloader_num_workers 32 \
    --max_train_steps=140000    \
    --checkpointing_steps=10000