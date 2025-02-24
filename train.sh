#!/bin/bash

# 激活环境
source /home/pengrui/mambaforge/bin/activate morphodiff

## 固定参数 ##
export CKPT_NUMBER=0
export TRAINED_STEPS=0

# 设置预训练模型路径
export MODEL_NAME="/home/pengrui/work_space_pengrui/huggingface_model/stable-diffusion-v1-4"

# 设置数据路径
export TRAIN_DATA_PATH="/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/adata_train_cleaned.h5ad"
export VALID_DATA_PATH="/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/adata_valid_updated_cleaned.h5ad"
export ADATA_CTRL_PATH="/home/pengrui/work_space_pengrui/project/RNA图像合成/1_3_rna_ctrl_data.h5ad"
export DRUG_EMBED_PATH="/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/molecule_embeddings_rdkit_所有位点.csv"

# export VALID_DRUG='calcitriol'

# 设置实验名称
export EXPERIMENT="pr_AIVC_所有位点"

# 设置日志目录
export LOG_DIR="log/"
# 检查并创建日志目录
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p $LOG_DIR
fi

# 设置checkpoint日志文件
export CKPT_LOG_FILE="${LOG_DIR}${EXPERIMENT}_checkpoints.csv"

# checkpoint日志文件的表头
export HEADER="dataset_id,log_dir,pretrained_model_dir,checkpoint_dir,seed,trained_steps,checkpoint_number"

# 获取列索引的函数
get_column_index() {
    local header_line=$1
    local column_name=$2
    echo $(echo "$header_line" | tr ',' '\n' | nl -v 0 | grep "$column_name" | awk '{print $1}')
}

# 检查checkpoint日志文件
if [ ! -f "$CKPT_LOG_FILE" ]; then
    echo "$HEADER" > "$CKPT_LOG_FILE"
    echo "创建了CSV checkpoint日志文件,表头为: $HEADER"
elif [ $(wc -l < "$CKPT_LOG_FILE") -eq 1 ]; then
    echo "$HEADER" > "$CKPT_LOG_FILE"
    echo "覆盖了CSV checkpoint日志文件表头为: $HEADER"
else
    echo "CSV checkpoint日志文件存在于 $CKPT_LOG_FILE"
    echo "读取最后一行以继续训练"
    
    LAST_LINE=$(tail -n 1 "$CKPT_LOG_FILE")
    HEADER_LINE=$(head -n 1 "$CKPT_LOG_FILE")
    CHECKPOINT_DIR_INDEX=$(get_column_index "$HEADER_LINE" "checkpoint_dir")
    MODEL_NAME=$(echo "$LAST_LINE" | cut -d',' -f$(($CHECKPOINT_DIR_INDEX + 1)))
    LAST_COLUMN=$(echo "$LAST_LINE" | awk -F',' '{print $NF}')
    CKPT_NUMBER=$((LAST_COLUMN))
    TRAINED_STEPS_INDEX=$(get_column_index "$HEADER_LINE" "trained_steps")
    TRAINED_STEPS=$(echo "$LAST_LINE" | cut -d',' -f$(($TRAINED_STEPS_INDEX + 1)))
fi

# 增加checkpoint编号
export CKPT_NUMBER=$((${CKPT_NUMBER}+1))
export OUTPUT_DIR="checkpoint/${EXPERIMENT}"

echo "Checkpoint编号: $CKPT_NUMBER"
echo "模型目录: $MODEL_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "已训练步数: $TRAINED_STEPS"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,3,4,7
unset LD_LIBRARY_PATH

#accelerate launch
# python -m debugpy --listen 9999 --wait-for-client /home/pengrui/mambaforge/envs/morphodiff/bin/accelerate-launch \
# 启动训练命令
accelerate launch --num_processes 4 --main_process_port=29873 --mixed_precision="fp16"   \
    ./train.py \
    --train_data=$TRAIN_DATA_PATH \
    --valid_data=$VALID_DATA_PATH \
    --ctrl_data=$ADATA_CTRL_PATH \
    --pretrained_model_path=$MODEL_NAME \
    --perturbation_embedding_path=$DRUG_EMBED_PATH \
    --naive_conditional=$SD_TYPE \
    --enable_xformers_memory_efficient_attention \
    --use_ema \
    --train_batch_size=256 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=100 \
    --rna_loss_types='GUSS' \
    --rna_loss_weight=0.1 \
    --num_train_epochs=100 \
    --checkpointing_log_file=$CKPT_LOG_FILE \
    --checkpoint_number=$CKPT_NUMBER \
    --output_dir=$OUTPUT_DIR \
    --cache_dir="tmp/" \
    --logging_dir="${LOG_DIR}${EXPERIMENT}_log" \
    --report_to="wandb" \
    --seed=42 \
    --dataloader_num_workers 32 \
    --save_epoch 100