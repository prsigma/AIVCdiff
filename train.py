import argparse
import logging
import math
import os
import json
import time
from typing import Optional
import shutil
from pathlib import Path
from datetime import datetime
from perturbation_encoder import PerturbationEncoder, PerturbationEncoderInference
from transformers import AutoFeatureExtractor

import accelerate
import anndata as ad
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
import wandb
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import random

# import debugpy
# # 设置调试服务器
# debugpy.listen(("localhost", 9999))
# print("等待调试器连接...")
# debugpy.wait_for_client()  # 暂停执行直到调试器连接
# print("调试器已连接，继续执行...")
    

if is_wandb_available():
    import wandb
    os.environ['WANDB_DIR'] = "tmp/"
    # os.environ["WANDB_MODE"] = "dryrun"

check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class RNAEncoder(nn.Module):
    """ Simple MLP encoder for RNA data. """
    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class RNADecoder(nn.Module):
    """ Simple MLP decoder for RNA data. """
    def __init__(self, latent_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_out_logvar = nn.Linear(hidden_dim, output_dim) # For GaussianNLLLoss

    def forward(self, z):
        h = F.relu(self.fc1(z))
        mu = self.fc_out_mu(h)
        logvar = self.fc_out_logvar(h) # Predict log variance
        return mu, logvar

class MorphoDiffRNA(nn.Module):
    """ Main model integrating UNet, RNA VAE, and fusion logic. """
    def __init__(self, args):
        super().__init__()
        self.args = args

        try:
            import pandas as pd
            self.drug_embeddings_df = pd.read_csv(args.perturbation_embedding_path, index_col=0) 
            self.drug_embedding_dim = self.drug_embeddings_df.shape[1]
            logger.info(f"从{args.perturbation_embedding_path}加载药物嵌入，维度为{self.drug_embedding_dim}")
        except Exception as e:
            logger.error(f"无法从{args.perturbation_embedding_path}加载药物嵌入：{e}")
            raise
        
        self.register_buffer("drug_embeddings_tensor", torch.tensor(self.drug_embeddings_df.values, dtype=torch.float32))
        self.drug_id_to_idx = {name: i for i, name in enumerate(self.drug_embeddings_df.index)}

        
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_path,
            subfolder="vae", revision=args.revision, variant=args.variant
        )
        if args.pretrained_model_name_or_path == args.pretrained_vae_path:
            self.unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet", revision=args.non_ema_revision
            )
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet_ema", revision=args.non_ema_revision
            )
        
        self.vae.requires_grad_(False)
        logger.info("图像VAE已冻结。")
        
        
        self.rna_encoder = RNAEncoder(args.rna_input_dim + self.drug_embedding_dim + 2, args.rna_latent_dim)
        self.rna_decoder = RNADecoder(args.rna_latent_dim, args.rna_input_dim)
        logger.info(f"初始化RNA编码器/解码器，输入维度{args.rna_input_dim}，潜在维度{args.rna_latent_dim}")

        # 将RNA潜在空间投影到UNet条件维度
        self.rna_latent_proj = nn.Linear(args.rna_latent_dim, args.condition_dim)
        logger.info(f"初始化RNA潜在投影层：从{args.rna_latent_dim}到{args.condition_dim}")
        
        if args.enable_xformers_memory_efficient_attention:
            self.enable_xformers_memory_efficient_attention()
           
        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            logger.info("为UNet启用梯度检查点。")

    def train(self, mode=True):
        """
        重写train方法，确保VAE保持在eval模式，即使模型设置为train模式
        """
    
        # 先将整个模型设为训练/评估模式
        super().train(mode)

        # 冻结图像VAE
        self.vae.requires_grad_(False)
        logger.info("图像VAE已冻结。")
        
        self.vae.eval()
        logger.info("VAE保持在评估模式，即使模型切换到训练模式。")
            
        return self

    # 用于递归应用梯度检查点的辅助函数
    def _set_gradient_checkpointing(self, module, value=True):
        if isinstance(module, (nn.Linear)): # 如果需要，添加其他类型
            module.gradient_checkpointing = value
            
    def enable_xformers_memory_efficient_attention(self):
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16不能用于训练...请更新xFormers..."
                )
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                logger.info("为UNet启用xFormers内存高效注意力。")
            except Exception as e:
                logger.warning(f"无法为UNet启用xformers：{e}")
        else:
            logger.warning("xFormers不可用。")
           
    def get_drug_embedding(self, perturb_ids):
        """ 通过ID检索预计算的药物嵌入。 """
        indices = [self.drug_id_to_idx.get(pid) for pid in perturb_ids]
        if None in indices:
            missing_ids = [pid for pid, idx in zip(perturb_ids, indices) if idx is None]
            raise ValueError(f"在嵌入文件中未找到扰动ID：{missing_ids}")
        indices = torch.tensor(indices, device=self.drug_embeddings_tensor.device, dtype=torch.long)
        return self.drug_embeddings_tensor[indices]
    
    def reparameterize(self, mu, logvar):
        """ VAE的重参数化技巧。 """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch, noise_scheduler):
        """ 用于训练的完整前向传递。 """
        images = batch["pixel_values"] # 图像张量
        rna_ctrl = batch["rna_ctrl"]   # 对照RNA表达
        rna_perturb = batch["rna_perturb"]  # 扰动后的RNA表达（目标）
        perturb_ids = batch["perturb_id"]  # 扰动ID列表
        
        # 获取剂量和时间信息
        pert_dose = batch.get("pert_dose", torch.ones(len(perturb_ids), 1, device=images.device))
        pert_time = batch.get("pert_time", torch.ones(len(perturb_ids), 1, device=images.device))
        
        # 1. 编码图像到潜在空间
        latents = self.vae.encode(images.to(dtype=self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # 2. 为扩散采样噪声和时间步
        noise = torch.randn_like(latents)
        # 如果指定，添加噪声偏移
        if self.args.noise_offset:
            noise += self.args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                device=latents.device
            )
        # 如果指定，输入扰动
        if self.args.input_perturbation:
            effective_noise = noise + self.args.input_perturbation * torch.randn_like(noise)
        else:
            effective_noise = noise
            
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        
        # 3. 向潜在变量添加噪声（前向扩散）
        noisy_latents = noise_scheduler.add_noise(latents, effective_noise, timesteps)

        # 4. 准备条件向量
        # 4a. 获取药物嵌入
        drug_embed = self.get_drug_embedding(perturb_ids).to(latents.device)
        
        # 4b. 将对照RNA、药物嵌入、剂量和时间连接起来作为RNA编码器的输入
        rna_encoder_input = torch.cat([
            rna_ctrl.to(latents.device),
            drug_embed,
            pert_dose.to(latents.device),
            pert_time.to(latents.device)
        ], dim=1)
        
        # 4c. 通过RNA编码器获取潜在表示
        rna_latent_mu, rna_latent_logvar = self.rna_encoder(rna_encoder_input)
        rna_latent = self.reparameterize(rna_latent_mu, rna_latent_logvar)
        
        # 4d. 解码RNA潜在表示以获得预测的扰动RNA
        rna_perturb_pred_mu, rna_perturb_pred_logvar = self.rna_decoder(rna_latent)
        
        # 4e. 将RNA潜在表示投影为UNet的条件
        condition_embed = self.rna_latent_proj(rna_latent)
        
        # 4f. 扩展为UNet（需要序列长度，例如77）
        # 假设UNet期望[batch_size, seq_len, condition_dim]
        seq_len = 77
                
        # 重塑condition_embed: [batch_size, condition_dim] -> [batch_size, 1, condition_dim]
        condition_embed = condition_embed.unsqueeze(1)
        # 沿序列长度维度重复
        encoder_hidden_states = condition_embed.repeat(1, seq_len, 1)

        # 5. UNet预测
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 6. 计算图像损失目标
        if noise_scheduler.config.prediction_type == "epsilon":
            image_target = effective_noise # 预测添加的噪声
        elif noise_scheduler.config.prediction_type == "v_prediction":
            image_target = noise_scheduler.get_velocity(latents, effective_noise, timesteps)
        else:
            raise ValueError(f"不支持的预测类型：{noise_scheduler.config.prediction_type}")

        return {
            "image_pred": model_pred,
            "image_target": image_target,
            "rna_perturb_pred_mu": rna_perturb_pred_mu,
            "rna_perturb_pred_logvar": rna_perturb_pred_logvar,
            "rna_perturb_target": rna_perturb.to(latents.device), # RNA损失的真实值
            "rna_latent_mu": rna_latent_mu,     # 用于KL的潜在分布的mu
            "rna_latent_logvar": rna_latent_logvar # 用于KL的潜在分布的logvar
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="model/stable-diffusion-v1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default="model/stable-diffusion-v1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the training h5ad file.",
    )
    parser.add_argument(
        "--image_path_column",
        type=str,
        default="image_path",
        help="Column name in h5ad obs containing the relative image path.",
    )
    parser.add_argument(
        "--perturb_id_column",
        type=str,
        default="perturb_id",
        help="Column name in h5ad obs containing the perturbation identifier.",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        default="",
        help="Root directory for images if paths in h5ad are relative to this directory.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help=("The name of the Dataset."),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default="cytochalasin-d,docetaxel,epothilone-b",
        nargs="+",
        help=("A set of perturbation ids evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='tmp/',
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "directory in which training log will be saved."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1000,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None,
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--checkpointing_log_file",
        type=str,
        default="",
        help=(
            "File address that stores checkpoint information."
        ),
    )
    parser.add_argument(
        "--checkpoint_number",
        type=str,
        default=None,
        help=(
            "Number for tracking checkpoint models."
        ),
    )
    parser.add_argument(
        "--naive_conditional",
        type=str,
        default='conditional',
        help=(
            "If the SD be trained with naive setting or conditional."
        ),
    )
    parser.add_argument(
        "--trained_steps",
        type=int,
        default=0,
        help=(
            "The number of trained steps so far."
        ),
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=200000,
        help=(
            "The total number of steps to train."
        ),
    )
    parser.add_argument(
        "--rna_input_dim", type=int, required=True,
        help="Input dimension of RNA expression data (number of genes)."
    )
    parser.add_argument(
        "--rna_latent_dim", type=int, default=128,
        help="Latent dimension for the RNA VAE."
    )
    parser.add_argument(
        "--fusion_proj_dim", type=int, default=256,
        help="Dimension to project drug and RNA embeddings before fusion."
    )
    parser.add_argument(
        "--condition_dim", type=int, default=768,
        help="Output dimension of the fusion layer (input dim for UNet cross-attention)."
    )
    parser.add_argument(
        "--perturbation_embedding_path", type=str, required=True,
        help="Path to the precomputed perturbation (drug) embeddings CSV file."
    )
    parser.add_argument(
        "--rna_ctrl_data_path", type=str, required=True,
        help="Path to the h5ad file containing control RNA profiles."
    )
    parser.add_argument(
        "--rna_loss_weight", type=float, default=0.1,
        help="Weight for the RNA reconstruction loss term."
    )
    parser.add_argument(
        "--kl_loss_weight", type=float, default=0.01,
        help="Weight for the KL divergence loss term in RNA VAE."
    )
    parser.add_argument(
        "--image_loss_weight", type=float, default=1.0,
        help="Weight for the image diffusion loss term."
    )
    parser.add_argument(
        "--use_dynamic_loss_scaling",
        action="store_true",
        help="是否使用动态损失权重缩放"
    )
    parser.add_argument(
        "--dynamic_weight_alpha",
        type=float,
        default=0.01,
        help="动态权重学习率"
    )
    parser.add_argument(
        "--kl_annealing",
        action="store_true",
        help="是否使用KL退火策略"
    )
    parser.add_argument(
        "--kl_annealing_steps",
        type=int,
        default=1000,
        help="KL退火步数"
    )
    parser.add_argument(
        "--max_kl_weight",
        type=float,
        default=1.0,
        help="KL退火最大权重"
    )
    parser.add_argument(
        "--loss_log_scale",
        action="store_true",
        help="是否对损失值进行对数变换"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if not args.train_data_path:
        raise ValueError("Need --train_data_path argument specifying the h5ad file.")
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data file not found: {args.train_data_path}")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    args.validation_prompts = args.validation_prompts[0].split(',')
    args.trained_steps = int(args.trained_steps)

    return args

# +++ Updated H5adDataset ++
class H5adDataset(Dataset):
    def __init__(self, adata_path, image_path_column, perturb_id_column, adata_ctrl_path, transform=None, image_root_dir="", rna_input_dim=None):
        self.adata_path = adata_path
        self.adata_ctrl_path = adata_ctrl_path
        self.image_path_column = image_path_column
        self.perturb_id_column = perturb_id_column
        self.transform = transform
        self.image_root_dir = image_root_dir

        # --- 加载主要AnnData --- 
        try:
            self.adata = ad.read_h5ad(adata_path)
            # 确保X是numpy数组
            if not isinstance(self.adata.X, np.ndarray):
                 logger.info("Converting main AnnData .X to dense numpy array.")
                 self.adata.X = self.adata.X.toarray()
            logger.info(f"Loaded main AnnData from {adata_path} with {len(self.adata)} obs.")
        except Exception as e:
            logger.error(f"Failed to load main AnnData file {adata_path}: {e}")
            raise
            
        # --- 加载对照组AnnData ---
        try:
            self.adata_ctrl = ad.read_h5ad(adata_ctrl_path)
            # 确保X是numpy数组
            if not isinstance(self.adata_ctrl.X, np.ndarray):
                logger.info("Converting control AnnData .X to dense numpy array.")
                self.adata_ctrl.X = self.adata_ctrl.X.toarray()
            logger.info(f"Loaded control AnnData from {adata_ctrl_path} with {len(self.adata_ctrl)} obs.")
        except Exception as e:
            logger.error(f"Failed to load control AnnData file {adata_ctrl_path}: {e}")
            raise

        # 验证主要adata中的列
        if self.image_path_column not in self.adata.obs.columns:
            raise ValueError(f"Image path column '{self.image_path_column}' not found in main AnnData obs.")
        if self.perturb_id_column not in self.adata.obs.columns:
             raise ValueError(f"Perturbation ID column '{self.perturb_id_column}' not found in main AnnData obs.")

        # 检查pert_dose和pert_time列是否存在
        self.has_pert_dose = 'pert_dose' in self.adata.obs.columns
        self.has_pert_time = 'pert_time' in self.adata.obs.columns
        
        if not self.has_pert_dose:
            logger.warning("'pert_dose' column not found in main AnnData obs. Will use default values.")
        if not self.has_pert_time:
            logger.warning("'pert_time' column not found in main AnnData obs. Will use default values.")

        # 存储RNA输入维度
        self.rna_input_dim = self.adata.X.shape[1]
        if rna_input_dim is not None and self.rna_input_dim != rna_input_dim:
            raise ValueError(f"Expected RNA dimension {rna_input_dim} but got {self.rna_input_dim}")
            
        # 为每种药物创建对照RNA映射
        self.perturb_to_ctrl = {}
        unique_perturbs = self.adata.obs[self.perturb_id_column].unique()
        
        # 获取对照组的总行数
        ctrl_total_rows = len(self.adata_ctrl)
        if ctrl_total_rows == 0:
            raise ValueError("对照组数据为空，无法创建映射")
            
        # 为每种药物随机分配一个不同的对照RNA
        used_ctrl_indices = set()  
        
        for perturb in unique_perturbs:
            # 找到一个未使用的随机对照组索引
            available_indices = list(set(range(ctrl_total_rows)) - used_ctrl_indices)
            
            # 如果所有对照组样本都已使用，则重新开始
            if not available_indices:
                used_ctrl_indices.clear()
                available_indices = list(range(ctrl_total_rows))
                
            # 随机选择一个可用的对照组索引
            random_ctrl_idx = np.random.choice(available_indices)
            used_ctrl_indices.add(random_ctrl_idx)
            
            # 将药物映射到随机选择的对照组RNA
            self.perturb_to_ctrl[perturb] = self.adata_ctrl.X[random_ctrl_idx]
        
        logger.info(f"为{len(self.perturb_to_ctrl)}种不同的扰动创建了随机对照RNA映射，每种扰动对应不同的对照组样本。")

    def __len__(self):
        return len(self.adata)
        
    def __getitem__(self, idx):
        # 获取图像路径
        image_path = self.adata.obs[self.image_path_column].iloc[idx]
        if self.image_root_dir:
            image_path = os.path.join(self.image_root_dir, image_path)
            
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}. Skipping sample index {idx}.")
            return None  

        # 获取扰动ID和对应的RNA数据
        perturb_id = str(self.adata.obs[self.perturb_id_column].iloc[idx])  # 确保是字符串
        rna_perturbed = self.adata.X[idx]  # 获取扰动后的RNA表达
        rna_ctrl = self.perturb_to_ctrl[perturb_id]  # 获取对应的对照RNA

        # 获取pert_dose和pert_time
        pert_dose = float(self.adata.obs['pert_dose'].iloc[idx]) if self.has_pert_dose else 0.0
        pert_time = float(self.adata.obs['pert_time'].iloc[idx]) if self.has_pert_time else 0.0

        sample = {
            "image": image, 
            "perturb_id": perturb_id,
            "rna_perturb": torch.tensor(rna_perturbed, dtype=torch.float32),
            "rna_ctrl": torch.tensor(rna_ctrl, dtype=torch.float32),
            "pert_dose": torch.tensor(pert_dose, dtype=torch.float32).unsqueeze(0),
            "pert_time": torch.tensor(pert_time, dtype=torch.float32).unsqueeze(0)
        }

        if self.transform:
            sample["image"] = self.transform(sample["image"])  # 现在image是一个张量

        return sample
# --- End Added H5adDataset ---

def main():
    args = parse_args()

    global dataset_id
    dataset_id = args.dataset_id

    global naive_conditional
    naive_conditional = args.naive_conditional

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir,
                                                      logging_dir=args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler")

    def deepspeed_zero_init_disabled_context_manager():
        """
        返回一个上下文列表，其中包含一个将禁用zero.Init的上下文，或一个空上下文列表
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # 创建MorphoDiffRNA模型实例
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        model = MorphoDiffRNA(args)
    
    # 创建EMA模型
    if args.use_ema:
        ema_model_path = args.pretrained_model_name_or_path
        ema_subfolder = "unet" if args.pretrained_model_name_or_path == args.pretrained_vae_path else "unet_ema"
        try:
            ema_unet = UNet2DConditionModel.from_pretrained(
                ema_model_path, subfolder=ema_subfolder, revision=args.revision, variant=args.variant
            )
            ema_unet = EMAModel(
                 ema_unet.parameters(), 
                 model_cls=UNet2DConditionModel,
                 model_config=ema_unet.config
            )
            logger.info("EMA模型创建成功。")
        except Exception as e:
             logger.error(f"从{ema_model_path}/{ema_subfolder}创建EMA模型失败: {e}")
             args.use_ema = False
             ema_unet = None
             logger.warning("由于初始化过程中出错，EMA已禁用。")
    else:
         ema_unet = None

    
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema and ema_unet is not None:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema and ema_unet is not None:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "请安装bitsandbytes以使用8位Adam。您可以通过运行`pip install bitsandbytes`来安装。"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    with accelerator.main_process_first():
        logger.info(f"创建H5adDataset，使用以下参数：")
        logger.info(f"  adata_path: {args.train_data_path}")
        logger.info(f"  adata_ctrl_path: {args.rna_ctrl_data_path}")
        logger.info(f"  image_path_column: {args.image_path_column}")
        logger.info(f"  perturb_id_column: {args.perturb_id_column}")
        logger.info(f"  image_root_dir: {args.image_root_dir}")
        logger.info(f"  rna_input_dim: {args.rna_input_dim}")

        try:
            train_dataset = H5adDataset(
                adata_path=args.train_data_path,
                image_path_column=args.image_path_column,
                perturb_id_column=args.perturb_id_column,
                adata_ctrl_path=args.rna_ctrl_data_path,  
                transform=train_transforms,
                image_root_dir=args.image_root_dir,
                rna_input_dim=args.rna_input_dim  
            )
            logger.info(f"成功创建H5adDataset，包含{len(train_dataset)}个样本。")
        except (ValueError, FileNotFoundError) as e:
             logger.error(f"创建H5adDataset时出错: {e}")
             exit(1)

    def collate_fn(examples):
        original_count = len(examples)
        examples = [e for e in examples if e is not None]
        filtered_count = len(examples)
        
        if original_count != filtered_count:
             logger.warning(f"由于此批次中的加载错误，过滤掉了{original_count - filtered_count}个样本。")

        if not examples:
            return None 

        pixel_values = torch.stack([example["image"] for example in examples]) 
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # 获取扰动ID
        perturb_ids = [example["perturb_id"] for example in examples]
        
        # 获取RNA数据
        rna_perturb = torch.stack([example["rna_perturb"] for example in examples])
        rna_ctrl = torch.stack([example["rna_ctrl"] for example in examples])
        
        # 获取剂量和时间
        pert_dose = torch.stack([example["pert_dose"] for example in examples])
        pert_time = torch.stack([example["pert_time"] for example in examples])
        
        # 创建批次字典
        batch = {
            "pixel_values": pixel_values,
            "perturb_id": perturb_ids,
            "rna_perturb": rna_perturb,
            "rna_ctrl": rna_ctrl,
            "pert_dose": pert_dose,
            "pert_time": pert_time
        }
        
        return batch

    # 创建DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # 调度器和训练步数的计算
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # 使用`accelerator`准备所有内容
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    # 重新计算总训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # 重新计算训练轮数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        if args.dataset_id and (args.report_to == "wandb" or args.report_to == "all"):
            if args.tracker_project_name is None: 
                 args.tracker_project_name = args.dataset_id
                 logger.info(f"使用dataset_id '{args.dataset_id}'作为wandb项目名称。")
            else:
                 logger.info(f"使用明确提供的wandb项目名称: '{args.tracker_project_name}'")
        elif (args.report_to == "wandb" or args.report_to == "all") and args.tracker_project_name is None:
            args.tracker_project_name = "morphodiff-default-project"
            logger.warning(f"未提供dataset_id且未设置tracker_project_name。使用默认wandb项目: '{args.tracker_project_name}'")

        tracker_config = dict(vars(args))
        if "validation_prompts" in tracker_config:
            tracker_config.pop("validation_prompts")
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except Exception as e:
             logger.error(f"初始化跟踪器失败: {e}")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 开始训练！
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** 开始训练 *****")
    logger.info(f"  样本数量 = {len(train_dataset)}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  每个设备的即时批次大小 = {args.train_batch_size}")
    logger.info(f"  总训练批次大小（包括并行、分布式和累积） = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # 可能从之前的保存加载权重和状态
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # 获取最近的检查点
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"检查点'{args.resume_from_checkpoint}'不存在。开始新的训练运行。"
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"从检查点{path}恢复")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    
    # 初始化动态权重和KL退火
    if args.use_dynamic_loss_scaling:
        # 初始化损失权重的对数值，便于动态调整
        log_image_weight = torch.tensor(math.log(args.image_loss_weight), device=accelerator.device, requires_grad=True)
        log_rna_weight = torch.tensor(math.log(args.rna_loss_weight), device=accelerator.device, requires_grad=True)
        log_kl_weight = torch.tensor(math.log(args.kl_loss_weight), device=accelerator.device, requires_grad=True)
        
        # 添加到优化器
        optimizer.add_param_group({'params': [log_image_weight, log_rna_weight, log_kl_weight], 'lr': args.dynamic_weight_alpha})
        
        logger.info("启用动态损失权重平衡，初始权重：图像=%.4f，RNA=%.4f，KL=%.4f", 
                   args.image_loss_weight, args.rna_loss_weight, args.kl_loss_weight)
    else:
        logger.info("使用固定损失权重：图像=%.4f，RNA=%.4f，KL=%.4f", 
                   args.image_loss_weight, args.rna_loss_weight, args.kl_loss_weight)
    
    # 如果传入，现在设置训练种子
    if args.seed is not None:
        set_seed(args.seed)
    total_trained_steps = args.trained_steps

    # initial checkpoint save 
    if not args.resume_from_checkpoint: 
        accelerator.wait_for_everyone() 
        if accelerator.is_main_process:
            initial_save_step = 0 
            save_path = os.path.join(args.output_dir, f"checkpoint-{initial_save_step}")
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Saving initial state at step {initial_save_step} to {save_path}...")

            if args.checkpoints_total_limit is not None:
                existing_initial_ckpt = os.path.join(args.output_dir, "checkpoint-0")
                if os.path.exists(existing_initial_ckpt):
                    logger.warning(f"Removing existing initial checkpoint {existing_initial_ckpt} before saving new one.")
                    shutil.rmtree(existing_initial_ckpt, ignore_errors=True)

            optimizer_path = os.path.join(save_path, "optimizer.bin")
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info(f"优化器状态已保存到 {optimizer_path}")
            
            # 保存学习率调度器
            lr_scheduler_path = os.path.join(save_path, "scheduler.bin")
            torch.save(lr_scheduler.state_dict(), lr_scheduler_path)
            logger.info(f"学习率调度器已保存到 {lr_scheduler_path}")
            
            # 保存随机状态
            rng_path = os.path.join(save_path, "rng_state.pth")
            rng_states = {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "random_rng_state": random.getstate(),
            }
            torch.save(rng_states, rng_path)
            logger.info(f"随机状态已保存到 {rng_path}")
            
            logger.info(f"已完成模型状态保存到 {save_path}")

            logger.info("保存模型组件用于推理管道...")
            model_unwrapped = accelerator.unwrap_model(model)
            unet_ckpt = model_unwrapped.unet

            # 如果使用EMA，将EMA权重应用到保存的模型
            if args.use_ema and ema_unet is not None:
                logger.info("将EMA权重应用到保存的模型。")
                ema_unet.copy_to(unet_ckpt.parameters())
            
            # 保存UNet
            unet_save_path = os.path.join(save_path, "unet")
            unet_ckpt.save_pretrained(unet_save_path)
            logger.info(f"UNet模型已保存到 {unet_save_path}")
            
            # 保存RNA编码器和解码器
            rna_encoder_save_path = os.path.join(save_path, "rna_encoder")
            os.makedirs(rna_encoder_save_path, exist_ok=True)
            torch.save(model_unwrapped.rna_encoder.state_dict(), os.path.join(rna_encoder_save_path, "pytorch_model.bin"))
            logger.info(f"RNA编码器已保存到 {rna_encoder_save_path}")
            
            rna_decoder_save_path = os.path.join(save_path, "rna_decoder")
            os.makedirs(rna_decoder_save_path, exist_ok=True)
            torch.save(model_unwrapped.rna_decoder.state_dict(), os.path.join(rna_decoder_save_path, "pytorch_model.bin"))
            logger.info(f"RNA解码器已保存到 {rna_decoder_save_path}")
            
            # 保存RNA潜在投影层
            rna_latent_proj_save_path = os.path.join(save_path, "rna_latent_proj")
            os.makedirs(rna_latent_proj_save_path, exist_ok=True)
            torch.save(model_unwrapped.rna_latent_proj.state_dict(), os.path.join(rna_latent_proj_save_path, "pytorch_model.bin"))
            logger.info(f"RNA潜在投影层已保存到 {rna_latent_proj_save_path}")
            
            # 保存调度器
            noise_scheduler.save_pretrained(os.path.join(save_path, "scheduler"))
            logger.info(f"调度器配置已保存到 {os.path.join(save_path, 'scheduler')}")

            # 保存VAE配置（VAE是冻结的，除非微调，否则无需保存权重）
            vae_config_save_path = os.path.join(save_path, "vae")
            os.makedirs(vae_config_save_path, exist_ok=True)
            try:
                    # 保存配置文件
                    model_unwrapped.vae.save_config(vae_config_save_path)
                    logger.info(f"VAE配置已保存到 {vae_config_save_path}")
                    # 可能还需要特征提取器配置
                    feature_extractor_save_path = os.path.join(save_path, "feature_extractor")
                    os.makedirs(feature_extractor_save_path, exist_ok=True)
                    try:
                        feature_extractor_orig_path = os.path.join(args.pretrained_vae_path, 'feature_extractor')
                        if os.path.isdir(feature_extractor_orig_path):
                            feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_orig_path)
                            feature_extractor.save_pretrained(feature_extractor_save_path)
                            logger.info(f"特征提取器配置已保存到 {feature_extractor_save_path}")
                        else:
                            logger.warning(f"在 {feature_extractor_orig_path} 未找到特征提取器目录，无法保存配置。")
                    except Exception as e:
                        logger.error(f"无法加载或保存特征提取器配置: {e}")
            except Exception as e:
                    logger.error(f"保存VAE/特征提取器配置时出错: {e}")

            # 保存训练参数
            try:
                args_dict = vars(args)
                # 将Path对象转换为字符串以进行JSON序列化
                for key, value in args_dict.items():
                    if isinstance(value, Path):
                        args_dict[key] = str(value)
                
                with open(os.path.join(save_path, 'training_args.json'), 'w') as f:
                    json.dump(args_dict, f, indent=2)
                logger.info(f"训练参数已保存到 {os.path.join(save_path, 'training_args.json')}")
            except Exception as e:
                logger.error(f"保存训练参数失败: {e}")

            # 将检查点信息写入日志文件
            try:
                # 确保文件存在并在需要时写入标题（幂等检查）
                if not os.path.exists(args.checkpointing_log_file):
                    with open(args.checkpointing_log_file, "w") as f:
                            # 根据预期列定义标题
                            header = "dataset_id,log_dir,pretrained_model_dir,checkpoint_dir,seed,trained_steps,checkpoint_number\n"
                            f.write(header)
                            logger.info(f"创建检查点日志文件: {args.checkpointing_log_file}")

                with open(args.checkpointing_log_file, "a") as f:
                    f.write(f"{args.dataset_id or 'N/A'}")
                    f.write(f"{args.logging_dir}")
                    f.write(f"{args.pretrained_model_name_or_path}")
                    f.write(f"{save_path}")
                    f.write(f"{args.seed}")
                    f.write(f"{total_trained_steps}")
                    f.write(f"{args.checkpoint_number or 'N/A'}")
                    logger.info(f"检查点信息已追加到 {args.checkpointing_log_file}")
            except Exception as e:
                    logger.error(f"写入检查点日志文件 {args.checkpointing_log_file} 失败: {e}")
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="步骤",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        train_image_loss = 0.0
        train_rna_loss = 0.0   
        train_kl_loss = 0.0    
        
        for step, batch in enumerate(train_dataloader):
            if batch is None: 
                logger.warning(f"由于批次中的数据加载错误，跳过第{epoch}轮的训练步骤{step}。")
                continue 
                
            with accelerator.accumulate(model):
                # 通过模型前向传播获取预测和目标
                outputs = model(batch, noise_scheduler)
                
                # 计算KL退火权重
                if args.kl_annealing:
                    kl_weight = min(1.0, global_step / args.kl_annealing_steps) * args.max_kl_weight
                else:
                    kl_weight = args.kl_loss_weight
                
                # 计算各个损失
                image_reconstruction_loss = F.mse_loss(
                    outputs["image_pred"].float(),
                    outputs["image_target"].float(),
                    reduction="mean")
                
                # 对RNA损失进行对数变换
                if args.loss_log_scale and outputs["rna_perturb_target"] is not None:
                    rna_reconstruction_loss = F.gaussian_nll_loss(
                        outputs["rna_perturb_pred_mu"], 
                        outputs["rna_perturb_target"], 
                        torch.exp(outputs["rna_perturb_pred_logvar"]),
                        full=True, 
                        reduction='mean'
                    )
                    rna_reconstruction_loss = torch.log1p(rna_reconstruction_loss)
                else:
                    rna_reconstruction_loss = torch.tensor(0.0, device=outputs["image_pred"].device)
                
                # 计算KL散度损失
                kl_div = -0.5 * torch.mean(1 + outputs["rna_latent_logvar"] - outputs["rna_latent_mu"].pow(2) - 
                                         outputs["rna_latent_logvar"].exp())
                
                # 动态权重或固定权重计算最终损失
                if args.use_dynamic_loss_scaling:
                    # 计算各损失的不确定性权重
                    image_weight = torch.exp(-log_image_weight)
                    rna_weight = torch.exp(-log_rna_weight)
                    kl_div_weight = torch.exp(-log_kl_weight) if not args.kl_annealing else torch.tensor(kl_weight, device=accelerator.device)
                    
                    # 使用不确定性权重计算总损失
                    loss = image_weight * image_reconstruction_loss + log_image_weight
                    loss = loss + rna_weight * rna_reconstruction_loss + log_rna_weight
                    loss = loss + kl_div_weight * kl_div + log_kl_weight
                    
                    # 每个step记录权重值
                    logger.info(f"当前动态权重：图像={image_weight.item():.4f}，RNA={rna_weight.item():.4f}，KL={kl_div_weight.item():.4f}")
                else:
                    # 使用固定权重
                    loss = args.image_loss_weight * image_reconstruction_loss + \
                           args.rna_loss_weight * rna_reconstruction_loss + \
                           kl_weight * kl_div
                
                train_image_loss += image_reconstruction_loss.detach().item()
                train_rna_loss += rna_reconstruction_loss.detach().item()
                train_kl_loss += kl_div.detach().item()
                train_loss += loss.detach().item()
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema and ema_unet is not None:
                     ema_unet.step(accelerator.unwrap_model(model).unet.parameters()) 

                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "train_image_loss": train_image_loss,
                    "train_rna_loss": train_rna_loss,
                    "train_kl_loss": train_kl_loss,
                    "lr": lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else 0
                }, step=global_step)
                
                train_loss = 0.0
                train_image_loss = 0.0
                train_rna_loss = 0.0
                train_kl_loss = 0.0
                total_trained_steps = global_step + args.trained_steps

                should_checkpoint = (total_trained_steps % args.checkpointing_steps == 0) or \
                                    (global_step >= args.max_train_steps)
                
                if should_checkpoint:
                    if accelerator.is_main_process:
                        logger.info(f"在总训练步骤{total_trained_steps}（全局步骤{global_step}）进行检查点")
                        logger.info(f"Checkpointing at total trained step {total_trained_steps} (global step {global_step})")
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        # remove old checkpoints if there are more than one saved checkpoints. Keep the latest one.
                        ckpt_files = os.listdir(args.output_dir)
                        ckpt_files = [f for f in ckpt_files if f.startswith("checkpoint")]
                        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("-")[1]))

                        # remove the folder with smaller number
                        if len(ckpt_files) > 1:
                            old_ckpt_path = os.path.join(
                                args.output_dir, ckpt_files[0])

                            # check if there is any folder in old_ckpt_path
                            if os.path.exists(old_ckpt_path):
                                shutil.rmtree(old_ckpt_path)
                                logger.info(f"Removed state from {old_ckpt_path}")
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{total_trained_steps}")
                        
                        # 手动保存优化器状态
                        os.makedirs(save_path, exist_ok=True)
                        optimizer_path = os.path.join(save_path, "optimizer.bin")
                        torch.save(optimizer.state_dict(), optimizer_path)
                        logger.info(f"优化器状态已保存到 {optimizer_path}")
                        
                        # 保存学习率调度器
                        lr_scheduler_path = os.path.join(save_path, "scheduler.bin")
                        torch.save(lr_scheduler.state_dict(), lr_scheduler_path)
                        logger.info(f"学习率调度器已保存到 {lr_scheduler_path}")
                        
                        # 保存随机状态
                        rng_path = os.path.join(save_path, "rng_state.pth")
                        rng_states = {
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state(),
                            "numpy_rng_state": np.random.get_state(),
                            "random_rng_state": random.getstate(),
                        }
                        torch.save(rng_states, rng_path)
                        logger.info(f"随机状态已保存到 {rng_path}")
                        
                        logger.info(f"已完成模型状态保存到 {save_path}")

                        logger.info("保存模型组件用于推理管道...")
                        model_unwrapped = accelerator.unwrap_model(model)
                        unet_ckpt = model_unwrapped.unet

                        if args.use_ema and ema_unet is not None:
                            logger.info("将EMA权重应用到保存的模型。")
                            ema_unet.copy_to(unet_ckpt.parameters())
                        
                        # 保存UNet
                        unet_save_path = os.path.join(save_path, "unet")
                        unet_ckpt.save_pretrained(unet_save_path)
                        logger.info(f"UNet模型已保存到 {unet_save_path}")
                        
                        # 保存RNA编码器和解码器
                        rna_encoder_save_path = os.path.join(save_path, "rna_encoder")
                        os.makedirs(rna_encoder_save_path, exist_ok=True)
                        torch.save(model_unwrapped.rna_encoder.state_dict(), os.path.join(rna_encoder_save_path, "pytorch_model.bin"))
                        logger.info(f"RNA编码器已保存到 {rna_encoder_save_path}")
                        
                        rna_decoder_save_path = os.path.join(save_path, "rna_decoder")
                        os.makedirs(rna_decoder_save_path, exist_ok=True)
                        torch.save(model_unwrapped.rna_decoder.state_dict(), os.path.join(rna_decoder_save_path, "pytorch_model.bin"))
                        logger.info(f"RNA解码器已保存到 {rna_decoder_save_path}")
                        
                        # 保存RNA潜在投影层
                        rna_latent_proj_save_path = os.path.join(save_path, "rna_latent_proj")
                        os.makedirs(rna_latent_proj_save_path, exist_ok=True)
                        torch.save(model_unwrapped.rna_latent_proj.state_dict(), os.path.join(rna_latent_proj_save_path, "pytorch_model.bin"))
                        logger.info(f"RNA潜在投影层已保存到 {rna_latent_proj_save_path}")

                        # 保存调度器
                        noise_scheduler.save_pretrained(os.path.join(save_path, "scheduler"))
                        logger.info(f"调度器配置已保存到 {os.path.join(save_path, 'scheduler')}")

                        # 保存VAE配置（VAE是冻结的，除非微调，否则无需保存权重）
                        vae_config_save_path = os.path.join(save_path, "vae")
                        os.makedirs(vae_config_save_path, exist_ok=True)
                        try:
                             # 保存配置文件
                             model_unwrapped.vae.save_config(vae_config_save_path)
                             logger.info(f"VAE配置已保存到 {vae_config_save_path}")
                             # 可能还需要特征提取器配置
                             feature_extractor_save_path = os.path.join(save_path, "feature_extractor")
                             os.makedirs(feature_extractor_save_path, exist_ok=True)
                             try:
                                 feature_extractor_orig_path = os.path.join(args.pretrained_vae_path, 'feature_extractor')
                                 if os.path.isdir(feature_extractor_orig_path):
                                     feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_orig_path)
                                     feature_extractor.save_pretrained(feature_extractor_save_path)
                                     logger.info(f"特征提取器配置已保存到 {feature_extractor_save_path}")
                                 else:
                                     logger.warning(f"在 {feature_extractor_orig_path} 未找到特征提取器目录，无法保存配置。")
                             except Exception as e:
                                 logger.error(f"无法加载或保存特征提取器配置: {e}")
                        except Exception as e:
                             logger.error(f"保存VAE/特征提取器配置时出错: {e}")

                        # 保存训练参数
                        try:
                            args_dict = vars(args)
                            # 将Path对象转换为字符串以进行JSON序列化
                            for key, value in args_dict.items():
                                if isinstance(value, Path):
                                    args_dict[key] = str(value)
                            
                            with open(os.path.join(save_path, 'training_args.json'), 'w') as f:
                                json.dump(args_dict, f, indent=2)
                            logger.info(f"训练参数已保存到 {os.path.join(save_path, 'training_args.json')}")
                        except Exception as e:
                            logger.error(f"保存训练参数失败: {e}")

                        # 将检查点信息写入日志文件
                        try:
                            # 确保文件存在并在需要时写入标题（幂等检查）
                            if not os.path.exists(args.checkpointing_log_file):
                                with open(args.checkpointing_log_file, "w") as f:
                                     # 根据预期列定义标题
                                     header = "dataset_id,log_dir,pretrained_model_dir,checkpoint_dir,seed,trained_steps,checkpoint_number\n"
                                     f.write(header)
                                     logger.info(f"创建检查点日志文件: {args.checkpointing_log_file}")

                            with open(args.checkpointing_log_file, "a") as f:
                                f.write(f"{args.dataset_id or 'N/A'}")
                                f.write(f"{args.logging_dir}")
                                f.write(f"{args.pretrained_model_name_or_path}")
                                f.write(f"{save_path}")
                                f.write(f"{args.seed}")
                                f.write(f"{total_trained_steps}")
                                f.write(f"{args.checkpoint_number or 'N/A'}")
                                logger.info(f"检查点信息已追加到 {args.checkpointing_log_file}")
                        except Exception as e:
                             logger.error(f"写入检查点日志文件 {args.checkpointing_log_file} 失败: {e}")

            # Check for termination based on total steps (moved from original code)
            if args.total_steps > 0 and total_trained_steps >= args.total_steps:
                 logger.info(f"Reached target total steps ({args.total_steps}). Stopping training.")
                 break # Break from inner step loop

            # Check for termination based on max_train_steps (steps in this run)
            if global_step >= args.max_train_steps:
                logger.info(f"Reached max_train_steps ({args.max_train_steps}) for this run. Stopping training.")
                break # Break from inner step loop
                
        # End of step loop
        if global_step >= args.max_train_steps or (args.total_steps > 0 and total_trained_steps >= args.total_steps):
             break # Break from outer epoch loop if training finished

    # End Training
    logger.info("Training finished.")
    accelerator.wait_for_everyone() # Ensure all processes finish cleanly
    
    # Clean up trackers
    accelerator.end_training()
    logger.info("Accelerator ended training.")

if __name__ == "__main__":
    main()