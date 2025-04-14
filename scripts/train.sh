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

# 设置实验名称 (例如: 使用数据集和日期)
# export EXPERIMENT="pr_aivcdiff_BBBC021" # 旧名称
export EXPERIMENT="pr_aivcdiff_BBBC021"

# 验证的药物列表 (扰动ID)
export VALID_PROMPT="alsterpaullone,cisplatin" # 保持不变

# 设置预训练模型路径 (SD基础模型)
export MODEL_NAME="/data/pr/stable-diffusion-v1-4"

# --- 修改: 数据输入 --- 
# 移除 TRAIN_DIR
# export TRAIN_DIR="/home/pr/MorphoDiff/train_data"
# 添加 H5AD 文件路径
export TRAIN_H5AD_PATH="/home/pr/BBBC021_train_data.h5ad" # <--- !!! 修改这里: 指定你的h5ad文件路径 !!!
# 添加 H5AD 中的列名
export IMAGE_PATH_COL="merged_image" # h5ad中包含图像相对路径的列名
export PERTURB_ID_COL="compound" # h5ad中包含扰动ID的列名
# 添加图像根目录 (如果h5ad中的路径是相对的)
export IMAGE_ROOT_DIR=" " # <--- !!! 修改这里: 如果h5ad中的路径是相对此目录的, 否则留空 (" ") !!!
# --- 结束修改 --- 

# --- 添加: 控制 RNA h5ad 文件路径 ---
export RNA_CTRL_H5AD_PATH="/home/pr/1_3_rna_ctrl_data.h5ad" # <--- !!! 修改这里: 指定你的对照组h5ad文件路径 !!!
# --- 结束添加 ---

# --- 添加: 新的必需参数 ---
# 设置RNA维度（比如基因数量）
export RNA_INPUT_DIM=977  # <--- !!! 修改这里: 根据你的RNA数据设置正确的维度 !!!
# 设置扰动（药物）嵌入文件路径
export PERTURBATION_EMBEDDING_PATH="/data/pr/molecule_embeddings_rdkit_BBBC021.csv"  # <--- !!! 修改这里: 指定你的药物嵌入文件路径 !!!
# --- 结束添加 ---

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
echo "数据目录: $TRAIN_H5AD_PATH"
echo "已训练步数: $TRAINED_STEPS"
echo "RNA输入维度: $RNA_INPUT_DIM"
echo "扰动嵌入路径: $PERTURBATION_EMBEDDING_PATH"

export CUDA_VISIBLE_DEVICES=0,1
unset LD_LIBRARY_PATH

# --resume_from_checkpoint='latest' \
export WANDB_API_KEY=06c4efa240357498f66106088907a3c775e74b58
accelerate launch --config_file /home/pr/.cache/huggingface/accelerate/default_config.yaml --num_processes 2 --main_process_port=29874 --mixed_precision="fp16" /data/pr/morphodiff/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --naive_conditional=$SD_TYPE \
  --train_data_path=$TRAIN_H5AD_PATH \
  --image_path_column=$IMAGE_PATH_COL \
  --perturb_id_column=$PERTURB_ID_COL \
  --image_root_dir="$IMAGE_ROOT_DIR" \
  --dataset_id=$EXPERIMENT \
  --enable_xformers_memory_efficient_attention \
  --rna_ctrl_data_path=$RNA_CTRL_H5AD_PATH \
  --random_flip \
  --use_ema \
  --train_batch_size=32 \
  --gradient_accumulation_steps=8 \
  --max_train_steps=140000 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_epochs=20 \
  --validation_prompts="$VALID_PROMPT"  \
  --checkpointing_steps=10000 \
  --output_dir=$OUTPUT_DIR \
  --pretrained_vae_path=$VAE_DIR \
  --cache_dir="/tmp/" \
  --report_to="wandb" \
  --logging_dir="${LOG_DIR}${EXPERIMENT}_log" \
  --seed=42 \
  --trained_steps=$TRAINED_STEPS \
  --dataloader_num_workers=32 \
  --checkpointing_log_file=$CKPT_LOG_FILE \
  --checkpoint_number=$CKPT_NUMBER \
  --rna_input_dim=$RNA_INPUT_DIM \
  --perturbation_embedding_path=$PERTURBATION_EMBEDDING_PATH  \
  --use_dynamic_loss_scaling \
  --dynamic_weight_alpha=0.01 \
  --kl_annealing \
  --kl_annealing_steps=1000 \
  --max_kl_weight=1.0 \
  --loss_log_scale