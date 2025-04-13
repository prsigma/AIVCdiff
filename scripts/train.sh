#!/bin/bash

## 固定参数 ##
export CKPT_NUMBER=0
export TRAINED_STEPS=0

# 设置训练类型 - "conditional"用于训练MorphoDiff, "naive"用于训练Stable Diffusion
export SD_TYPE="conditional"

# 设置预训练VAE模型路径
export VAE_DIR="/data/pr/stable-diffusion-v1-4"

# 设置日志目录
export LOG_DIR="log/"
# 检查并创建日志目录
if [ ! -d "$LOG_DIR" ]; then                                                                                                                                                                            
  mkdir -p $LOG_DIR
fi

# 设置实验名称
export EXPERIMENT="pr_morphodiff_Morphodiff"

#验证的药物列表
export VALID_PROMPT="alsterpaullone,cisplatin"

# 设置预训练模型路径
export MODEL_NAME="/data/pr/stable-diffusion-v1-4"

# 设置训练数据目录
export TRAIN_DIR="/home/pr/MorphoDiff/train_data"

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
echo "数据目录: $TRAIN_DIR"
echo "已训练步数: $TRAINED_STEPS"

export CUDA_VISIBLE_DEVICES=0,1
unset LD_LIBRARY_PATH

# /home/pengrui/mambaforge/envs/morphodiff/bin/python -m debugpy --listen 9999 --wait-for-client

# 跑morphodiff

# /home/pengrui/mambaforge/envs/morphodiff/bin/accelerate-launch --num_processes 4 --main_process_port=29874 --mixed_precision="fp16" ../train.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --naive_conditional=$SD_TYPE \
#   --train_data_dir=$TRAIN_DIR \
#   --dataset_id=$EXPERIMENT \
#   --enable_xformers_memory_efficient_attention \
#   --random_flip \
#   --use_ema \
#   --resume_from_checkpoint='latest' \
#   --train_batch_size=32 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=161000 \
#   --learning_rate=1e-05 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --validation_epochs=20 \
#   --validation_prompts=$VALID_PROMPT  \
#   --checkpointing_steps=20 \
#   --output_dir=$OUTPUT_DIR \
#   --image_column="image" \
#   --caption_column='perturb_id' \
#   --pretrained_vae_path=$VAE_DIR \
#   --cache_dir="/tmp/" \
#   --report_to="wandb" \
#   --logging_dir="${LOG_DIR}${EXPERIMENT}_log" \
#   --seed=42 \
#   --checkpointing_log_file=$CKPT_LOG_FILE \
#   --checkpoint_number=$CKPT_NUMBER \
#   --trained_steps=$TRAINED_STEPS \
#   --dataloader_num_workers 32

#naive模型
# --resume_from_checkpoint='latest' \
accelerate launch --config_file /home/pr/.cache/huggingface/accelerate/default_config.yaml --num_processes 2 --main_process_port=29874 --mixed_precision="fp16" /data/pr/morphodiff/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --naive_conditional=$SD_TYPE \
  --train_data_dir=$TRAIN_DIR \
  --dataset_id=$EXPERIMENT \
  --enable_xformers_memory_efficient_attention \
  --random_flip \
  --use_ema \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=140000 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_epochs=20 \
  --validation_prompts=$VALID_PROMPT  \
  --checkpointing_steps=20 \
  --output_dir=$OUTPUT_DIR \
  --image_column="image" \
  --caption_column='perturb_id' \
  --pretrained_vae_path=$VAE_DIR \
  --cache_dir="/tmp/" \
  --report_to="wandb" \
  --logging_dir="${LOG_DIR}${EXPERIMENT}_log" \
  --seed=42 \
  --trained_steps=$TRAINED_STEPS \
  --dataloader_num_workers 32 \
  --checkpointing_log_file=$CKPT_LOG_FILE \
  --checkpoint_number=$CKPT_NUMBER \