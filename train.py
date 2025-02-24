# 标准库导入
import argparse
import json
import logging
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Callable

# 第三方库导入
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb
import pandas as pd
import torch_fidelity
from torchvision.utils import save_image

# accelerate相关导入
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

# transformers相关导入
import transformers
from transformers import AutoFeatureExtractor
from transformers.utils import ContextManagers

# diffusers相关导入
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler, 
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DiffusionPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import (
    check_min_version,
    deprecate,
    is_wandb_available,
    make_image_grid
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# Hugging Face Hub相关导入
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm

# 本地模块导入
from models import AIVCdiff
from datasets import AIVCdiffDataset
from perturbation_encoder import (
    PerturbationEncoder,
    PerturbationEncoderInference
)
import datasets
from typing import Dict, Any, Union
from PIL import Image

# wandb配置
if is_wandb_available():
    import wandb
    os.environ['WANDB_DIR'] = "tmp/"
    # os.environ["WANDB_MODE"] = "dryrun"

# 版本检查
check_min_version("0.26.0.dev0")

# 日志配置
logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

class CustomStableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae,
        unet,
        scheduler,
        feature_extractor,
        rna_encoder,
        rna_decoder,
        drug_proj,
        rna_proj,
        fusion_layer,
        requires_safety_checker=False
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            rna_encoder=rna_encoder,
            rna_decoder=rna_decoder,
            drug_proj=drug_proj,
            rna_proj=rna_proj,
            fusion_layer=fusion_layer
        )
        
        self.requires_safety_checker = requires_safety_checker

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        """从预训练目录加载模型"""
        try:
            # 1. 加载标准组件
            vae = AutoencoderKL.from_pretrained(
                pretrained_model_path,
                subfolder="vae"
            )
            
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_path,
                subfolder="unet"
            )
            
            scheduler = DDPMScheduler.from_pretrained(
                pretrained_model_path,
                subfolder="scheduler"
            )
            
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_model_path,
                subfolder="feature_extractor"
            )
            
            # 2. 初始化自定义组件
            rna_encoder = RNAEncoder(
                input_dim=kwargs.get('rna_input_dim', 978),
                hidden_layer_sizes=[1024, 768],
                latent_dim=512,
                dropout_rate=0.1
            )
            rna_encoder.load_state_dict(
                torch.load(os.path.join(pretrained_model_path, "rna_encoder.pth"))
            )
            
            rna_decoder = RNADecoder(
                output_dim=kwargs.get('rna_input_dim', 978),
                hidden_layer_sizes=[768, 1024],
                latent_dim=512,
                dropout_rate=0.1
            )
            rna_decoder.load_state_dict(
                torch.load(os.path.join(pretrained_model_path, "rna_decoder.pth"))
            )
            
            drug_proj = nn.Linear(512, 768)
            drug_proj.load_state_dict(
                torch.load(os.path.join(pretrained_model_path, "drug_proj.pth"))
            )
            
            rna_proj = nn.Linear(512, 768)
            rna_proj.load_state_dict(
                torch.load(os.path.join(pretrained_model_path, "rna_proj.pth"))
            )
            
            fusion_layer = nn.Sequential(
                nn.Linear(768 * 2, 768),
                nn.LayerNorm(768),
                nn.ReLU(),
                nn.Linear(768, 768)
            )
            fusion_layer.load_state_dict(
                torch.load(os.path.join(pretrained_model_path, "fusion_layer.pth"))
            )
            
            # 3. 创建pipeline实例
            pipeline = cls(
                vae=vae,
                unet=unet,
                scheduler=scheduler,
                feature_extractor=feature_extractor,
                rna_encoder=rna_encoder,
                rna_decoder=rna_decoder,
                drug_proj=drug_proj,
                rna_proj=rna_proj,
                fusion_layer=fusion_layer,
                requires_safety_checker=False
            )
            
            return pipeline
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Attempting to load from: {pretrained_model_path}")
            raise

    def save_pretrained(self, save_directory: Union[str, os.PathLike], safe_serialization: bool = False):
        """保存所有模型组件到指定目录"""
        os.makedirs(save_directory, exist_ok=True)

        # 1. 保存模型配置
        pipeline_config = {
            "_class_name": self.__class__.__name__,
            "_diffusers_version": diffusers.__version__,
            "requires_safety_checker": self.requires_safety_checker,
        }
        
        with open(os.path.join(save_directory, "model_index.json"), "w") as f:
            json.dump(pipeline_config, f, indent=2)

        # 2. 保存各个组件
        if self.vae is not None:
            self.vae.save_pretrained(os.path.join(save_directory, "vae"))
            
        if self.unet is not None:
            self.unet.save_pretrained(os.path.join(save_directory, "unet"))
            
        if self.scheduler is not None:
            self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
            
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(os.path.join(save_directory, "feature_extractor"))
            
        # 3. 保存自定义组件
        if self.rna_encoder is not None:
            torch.save(self.rna_encoder.state_dict(), 
                      os.path.join(save_directory, "rna_encoder.pth"))
            
        if self.rna_decoder is not None:
            torch.save(self.rna_decoder.state_dict(), 
                      os.path.join(save_directory, "rna_decoder.pth"))
            
        if self.drug_proj is not None:
            torch.save(self.drug_proj.state_dict(), 
                      os.path.join(save_directory, "drug_proj.pth"))
            
        if self.rna_proj is not None:
            torch.save(self.rna_proj.state_dict(), 
                      os.path.join(save_directory, "rna_proj.pth"))
            
        if self.fusion_layer is not None:
            torch.save(self.fusion_layer.state_dict(), 
                      os.path.join(save_directory, "fusion_layer.pth"))
    
    def prepare_condition(self, drug_embedding, pre_rna):
        """准备条件向量"""
        # 1. 处理drug embedding
        drug_embed = self.drug_proj(drug_embedding)
        
        # 2. 处理RNA
        rna_z, _, _ = self.rna_encoder(pre_rna)
        rna_embed = self.rna_proj(rna_z)
        
        # 3. 融合
        combined_embed = torch.cat([drug_embed, rna_embed], dim=1)
        condition = self.fusion_layer(combined_embed)
        
        # 4. 标准化维度
        condition = condition.unsqueeze(1).repeat(1, 77, 1)
        
        return condition
    
    def encode_prompt(
        self,
        drug_embedding,
        pre_rna,
        device,
        num_images_per_prompt=1,
    ):
        """重写的prompt编码方法"""
        batch_size = drug_embedding.shape[0]
        
        # 准备条件向量
        condition = self.prepare_condition(drug_embedding, pre_rna)
        condition = condition.to(device)
            
        return condition

# def augment_images(images, num_augmented=50):
#     """对图像进行数据增强，保持图像尺寸不变
    
#     Args:
#         images: 原始图像列表，每个图像shape为(512, 512, 3)
#         num_augmented: 需要生成的增强图像数量
    
#     Returns:
#         augmented_images: 增强后的图像列表，每个图像shape保持(512, 512, 3)
#     """
#     augmented_images = []
    
#     while len(augmented_images) < num_augmented:
#         for img in images:
#             if len(augmented_images) >= num_augmented:
#                 break
                
#             # 转换为torch tensor便于处理
#             img_tensor = torch.from_numpy(img).float()
            
#             # 确保输入图像形状正确
#             if img_tensor.shape != (512, 512, 3):
#                 img_tensor = img_tensor.view(512, 512, 3)
            
#             # 随机水平翻转 (不会改变图像尺寸)
#             if torch.rand(1) > 0.5:
#                 img_tensor = torch.flip(img_tensor, [1])  # 在宽度维度上翻转
            
#             # 随机垂直翻转 (不会改变图像尺寸)
#             if torch.rand(1) > 0.5:
#                 img_tensor = torch.flip(img_tensor, [0])  # 在高度维度上翻转
            
#             # 随机旋转90度的倍数 (需要调整以保持尺寸)
#             k = torch.randint(4, (1,)).item()
#             if k > 0:
#                 # 转置操作以保持通道维度在最后
#                 img_tensor = img_tensor.permute(2, 0, 1)  # (3, 512, 512)
#                 img_tensor = torch.rot90(img_tensor, k, [1, 2])
#                 img_tensor = img_tensor.permute(1, 2, 0)  # 转回(512, 512, 3)
            
#             # 验证输出形状
#             assert img_tensor.shape == (512, 512, 3), f"Shape mismatch: {img_tensor.shape}"
            
#             augmented_images.append(img_tensor.numpy())
    
#     return augmented_images[:num_augmented]

def log_validation(args, accelerator, weight_dtype, step, epoch, model, noise_scheduler, valid_dataloader):
    """执行验证并记录结果
    
    Args:
        args: 训练参数
        accelerator: Accelerator对象
        weight_dtype: 权重数据类型
        step: 当前步数
        epoch: 当前epoch
        model: 模型
        noise_scheduler: 噪声调度器
        valid_dataloader: 验证集数据加载器
    """
    # 创建验证目录
    validation_dir = os.path.join(args.output_dir, f"validation_epoch_{epoch}_step_{step}")
    os.makedirs(validation_dir, exist_ok=True)

    # 设置模型为评估模式
    model.eval()
    model.to(accelerator.device)  # 确保模型在GPU上
    
    # 用于存储验证损失
    val_loss = 0
    val_image_loss = 0
    val_rna_loss = 0

    # valid_drug = args.valid_drug
    
    # # 只处理valid_drug中指定的药物
    # real_images = []
    # generated_images = []

    # 创建验证进度条
    progress_bar = tqdm(
        total=len(valid_dataloader),
        desc=f"Epoch {epoch} [Validation]",
        disable=not accelerator.is_local_main_process,
        leave=False
    )
    
    # # 用于存储生成的图像和指标
    # metrics = {
    #     'fid': [],
    #     'kid_mean': [],
    #     'kid_std': []
    # }

    # 创建损失函数
    gaussian_criterion = torch.nn.GaussianNLLLoss()
    mse_criterion = torch.nn.MSELoss(reduction='mean')
    kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')

    # # 获取原始模型组件
    # unet_ckpt = model.module.unet
    # vae_ckpt = model.module.vae
    # rna_encoder_ckpt = model.module.rna_encoder
    # rna_decoder_ckpt = model.module.rna_decoder
    # drug_proj_ckpt = model.module.drug_proj
    # rna_proj_ckpt = model.module.rna_proj
    # fusion_layer_ckpt = model.module.fusion_layer
    
    # feature_extractor = AutoFeatureExtractor.from_pretrained(
    #             args.pretrained_model_path+'/feature_extractor')
    
    # # 保存完整pipeline
    # pipeline = CustomStableDiffusionPipeline(
    #     vae=vae_ckpt,
    #     unet=unet_ckpt,
    #     scheduler=noise_scheduler,
    #     feature_extractor=feature_extractor,
    #     rna_encoder=rna_encoder_ckpt,
    #     rna_decoder=rna_decoder_ckpt,
    #     drug_proj=drug_proj_ckpt,
    #     rna_proj=rna_proj_ckpt,
    #     fusion_layer=fusion_layer_ckpt,
    #     requires_safety_checker=False
    # ).to(accelerator.device)

    # # 设置为评估模式
    # pipeline.vae.eval()
    # pipeline.unet.eval()
    # pipeline.rna_encoder.eval()
    # pipeline.rna_decoder.eval()
    # pipeline.drug_proj.eval()
    # pipeline.rna_proj.eval()
    # pipeline.fusion_layer.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dataloader):
            
            batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}

            # 1. 计算验证损失
            outputs = model(batch, noise_scheduler)
            
            # 计算图像损失(添加权重)
            image_loss = F.mse_loss(
                outputs["noise_pred"].float(),
                outputs["noise_target"].float(),
                reduction="mean"
            ) * args.image_loss_weight
            
            # 计算RNA损失(添加权重)
            gene_means = outputs["rna_mu_pred"]
            gene_vars = torch.exp(outputs["rna_logvar_pred"])
            target = outputs["rna_target"]
            
            rna_loss = torch.tensor(0.0, device=accelerator.device)
            if "GUSS" in args.rna_loss_types:
                guss_loss = gaussian_criterion(gene_means, target, gene_vars)
                rna_loss += guss_loss
                
            if "MSE" in args.rna_loss_types:
                mse_loss = mse_criterion(gene_means, target)
                rna_loss += mse_loss * args.mse_weight
                
            if "KL" in args.rna_loss_types:
                kl_loss = kl_criterion(
                    F.log_softmax(gene_means, dim=1),
                    F.softmax(target, dim=1)
                )
                rna_loss += kl_loss * args.kl_weight
            
            rna_loss = rna_loss * args.rna_loss_weight
            total_loss = image_loss + rna_loss
    
            # perturbations = batch['drug_id']
            # real_imgs = batch['source_image']
            
            # # 找到当前batch中属于该药物的图像
            # drug_indices = [i for i, pert in enumerate(perturbations) 
            #                 if pert == valid_drug]
            
            # if drug_indices:
            #     # 收集该药物的真实图像
            #     drug_real_imgs = [real_imgs[i].cpu().numpy() for i in drug_indices]
                
            #     # 对真实图像进行数据增强
            #     augmented_real_imgs = augment_images(drug_real_imgs, num_augmented=10*len(drug_real_imgs))
            #     real_images.extend(augmented_real_imgs)
                
            #     # 为该药物生成60张新图像
            #     with torch.autocast("cuda"):
            #         # 重复使用第一个找到的该药物的条件
            #         idx = drug_indices[0]
            #         drug_embedding = batch['drug_embedding'][idx:idx+1].repeat(10*len(drug_indices), 1)
            #         pre_rna = batch['pre_rna'][idx:idx+1].repeat(10*len(drug_indices), 1)
                    
            #         images = pipeline(
            #             drug_embedding=drug_embedding,
            #             pre_rna=pre_rna,
            #             device=accelerator.device,
            #             num_images_per_prompt=1,
            #             guidance_scale=7.5,
            #             num_inference_steps=50
            #         ).images
                    
            #         generated_images.extend([np.array(img) for img in images])

            # 累加损失
            val_loss += total_loss.item()
            val_image_loss += image_loss.item()
            val_rna_loss += rna_loss.item()

            progress_bar.update(1)

    # 计算平均损失
    num_batches = len(valid_dataloader)
    val_loss /= num_batches
    val_image_loss /= num_batches
    val_rna_loss /= num_batches

    # # 3. 计算FID和KID
    # gen_dir = os.path.join(validation_dir, f"generated_{valid_drug}")
    # real_dir = os.path.join(validation_dir, f"real_{valid_drug}")
    # os.makedirs(gen_dir, exist_ok=True)
    # os.makedirs(real_dir, exist_ok=True)
    
    # # 保存生成的图像
    # for idx, gen_img in enumerate(generated_images):
    #     gen_img = gen_img.clip(0, 255).astype(np.uint8)
    #     img = Image.fromarray(gen_img)
    #     img.save(os.path.join(gen_dir, f"gen_{idx}.png"))
        
    
    # # 保存增强后的真实图像
    # for idx, real_img in enumerate(real_images):
    #     real_img = real_img.clip(0, 255).astype(np.uint8)
    #     img = Image.fromarray(real_img)
    #     img.save(os.path.join(real_dir, f"real_{idx}.png"))
    
    # # 计算FID和KID
    # metrics_dict = torch_fidelity.calculate_metrics(
    #     input1=gen_dir,
    #     input2=real_dir,
    #     cuda=True if torch.cuda.is_available() else False,
    #     isc=False,  # 禁用 Inception Score
    #     fid=True,   # 只计算 FID
    #     kid=False,  # 禁用 KID
    #     batch_size=256,
    #     kid_subset_size=500,
    #     verbose=False
    # )
    
    # metrics['fid'].append(metrics_dict['frechet_inception_distance'])
    # metrics['kid_mean'].append(metrics_dict['kernel_inception_distance_mean'])
    # metrics['kid_std'].append(metrics_dict['kernel_inception_distance_std'])

    # 计算平均指标
    # avg_fid = sum(metrics['fid']) / len(metrics['fid'])
    # avg_kid_mean = sum(metrics['kid_mean']) / len(metrics['kid_mean'])
    # avg_kid_std = sum(metrics['kid_std']) / len(metrics['kid_std'])

    logs = {
                "validation/total_loss": val_loss,
                "validation/image_loss": val_image_loss,
                "validation/rna_loss": val_rna_loss,
                "epoch": epoch,
                "step": step
            }

    # 记录结果
    if accelerator.is_main_process:  
        # 使用 wandb 记录
        wandb.log(logs)
        
        # # 如果你还想保存一些生成的图像示例
        # if len(generated_images) > 0:
        #     # 选择一些示例图像
        #     example_images = generated_images[:4]  # 比如展示前4张图
        #     wandb.log({
        #         "validation/generated_samples": [wandb.Image(img) for img in example_images],
        #         "step": step + epoch * len(valid_dataloader)
        #     })
    
    # 恢复训练模式
    model.train()
    
    # 清理GPU内存
    torch.cuda.empty_cache()

    return {
        "valid_loss": val_loss,
        "image_loss": val_image_loss,
        "rna_loss": val_rna_loss
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for AIVCdiff model")
    
    # 删除原有的数据相关参数
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data h5ad file")
    parser.add_argument("--valid_data", type=str, required=True,
                       help="Path to validation data h5ad file")
    parser.add_argument("--ctrl_data", type=str, required=True,
                       help="Path to control RNA data h5ad file")
    parser.add_argument("--perturbation_embedding_path", type=str, required=True,
                       help="Path to perturbation embeddings CSV file")
    
    # 模型相关参数
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                       help="Path to pretrained stable diffusion model")
    parser.add_argument("--rna_input_dim", type=int, default=977,
                       help="Input dimension of RNA expression data")
    parser.add_argument("--drug_latent_dim", type=int, default=193,
                       help="Latent dimension of drug embeddings")
    parser.add_argument("--condition_dim", type=int, default=768,
                       help="Dimension of condition embeddings")

    # RNA损失相关参数
    parser.add_argument("--rna_loss_types", 
                       type=str, 
                       nargs="+", 
                       choices=["GUSS", "NB", "MSE", "KL"],
                       default=["GUSS", "MSE", "KL"],
                       help="Types of RNA loss to use: GUSS, NB, MSE, KL")
    parser.add_argument("--rna_loss_weight", type=float, default=0.01,
                       help="Weight for RNA loss")
    parser.add_argument("--image_loss_weight", type=float, default=10,
                       help="Weight for RNA loss")
    parser.add_argument("--mse_weight", type=float, default=10.0,
                       help="Weight for MSE loss in RNA prediction")
    parser.add_argument("--kl_weight", type=float, default=0.01,
                       help="Weight for KL loss in RNA prediction")
    
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
    parser.add_argument("--save_epoch",type=int,default=1)
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
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--valid_drug",
        type=str,
        default="BRD-K23984332",
        help=(
            "The drug id to use for validation."
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
        "--total_steps",
        type=int,
        default=200000,
        help=(
            "The total number of steps to train."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.report_to == 'wandb':
        args.tracker_project_name = args.output_dir.split('/')[-1] 

    return args

def encode_prompt(identifier):
    """Get gene embedding generated by scGPT based on input identifier.

    Args:
        identifier (str): perturbation identifier

    Returns:
        prompt_embeds (torch.Tensor): gene embedding"""
    global dataset_id
    global naive_conditional
    encoder = PerturbationEncoder(dataset_id, naive_conditional, 'SD')
    prompt_embeds = encoder.get_gene_embedding(identifier)
    # shape (bs, 77, 768)
    return prompt_embeds

def train_one_epoch(
    args,
    accelerator,
    epoch,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    ema_unet=None,  # 添加 ema_unet 参数
    weight_dtype=torch.float32,
):
    """训练一个epoch
    
    Args:
        args: 训练参数
        accelerator: Accelerator对象
        epoch: 当前epoch
        model: AIVCdiff模型
        noise_scheduler: 噪声调度器
        optimizer: 优化器
        train_dataloader: 数据加载器
        lr_scheduler: 学习率调度器
        ema_unet: EMA模型
        weight_dtype: 权重数据类型
    """
    model.train()
    
    # 创建进度条
    progress_bar = tqdm(
        total=len(train_dataloader), 
        desc=f"Epoch {epoch} [Training]",
        disable=not accelerator.is_local_main_process,
        leave=False
    )
    
    # 定义RNA预测的损失函数
    gaussian_criterion = torch.nn.GaussianNLLLoss()
    mse_criterion = torch.nn.MSELoss(reduction='mean')
    kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            # 前向传播
            outputs = model(batch, noise_scheduler)
            
            # 1. 计算图像重建损失
            if args.snr_gamma is None:
                image_loss = F.mse_loss(
                    outputs["noise_pred"].float(),
                    outputs["noise_target"].float(),
                    reduction="mean"
                )
            else:
                # 使用SNR加权的损失
                snr = compute_snr(noise_scheduler, outputs["timesteps"])
                mse_loss_weights = torch.stack(
                    [snr, args.snr_gamma * torch.ones_like(outputs["timesteps"])], dim=1
                ).min(dim=1)[0] / snr
                image_loss = F.mse_loss(
                    outputs["noise_pred"].float(),
                    outputs["noise_target"].float(),
                    reduction="none"
                ).mean(dim=[1, 2, 3])
                image_loss = (image_loss * mse_loss_weights).mean()
            
            # 2. 计算RNA预测损失
            gene_means = outputs["rna_mu_pred"]
            gene_vars = torch.exp(outputs["rna_logvar_pred"])  # 转换为方差
            target = outputs["rna_target"]
            
            rna_loss = torch.tensor(0.0, device=accelerator.device)  # 初始化为0，确保在正确的设备上
            # 根据选择的损失函数计算RNA损失
            if "GUSS" in args.rna_loss_types:
                # 高斯损失
                guss_loss = gaussian_criterion(
                    gene_means,
                    target,
                    gene_vars
                )
                rna_loss += guss_loss
                
            if "NB" in args.rna_loss_types:
                # 负二项分布损失
                counts, logits = model.pgm._convert_mean_disp_to_counts_logits(
                    torch.clamp(gene_means, min=1e-3, max=1e3),
                    torch.clamp(gene_vars, min=1e-3, max=1e3)
                )
                nb_dist = torch.distributions.negative_binomial.NegativeBinomial(
                    total_count=counts,
                    logits=logits
                )
                nb_loss = -nb_dist.log_prob(target).mean()
                rna_loss += nb_loss
                
            if "MSE" in args.rna_loss_types:
                mse_loss = mse_criterion(gene_means, target)
                rna_loss += mse_loss * args.mse_weight
                
            if "KL" in args.rna_loss_types:
                kl_loss = kl_criterion(
                    F.log_softmax(gene_means, dim=1),
                    F.softmax(target, dim=1)
                )
                rna_loss += kl_loss * args.kl_weight
            
            # 3. 总损失
            loss = args.image_loss_weight * image_loss + args.rna_loss_weight * rna_loss
            
            # 确保损失需要梯度
            if not loss.requires_grad:
                raise ValueError("Loss does not require gradients. Check if model parameters require gradients.")
                
            # 反向传播
            accelerator.backward(loss)
            # if accelerator.sync_gradients:
            #     accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # 更新 EMA 模型
            if args.use_ema and accelerator.sync_gradients and ema_unet is not None:
                ema_unet.step(model.module.unet.parameters())
            
            # 记录日志
            logs = {
                "loss": loss.detach().item(),
                "image_loss": args.image_loss_weight * image_loss.detach().item(),
                "rna_loss": args.rna_loss_weight * rna_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            
             # 记录到wandb
            if accelerator.is_main_process:
                # 使用 wandb 记录
                wandb.log(logs)
            
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
        
    progress_bar.close()
    
    return

def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
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

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.logging_dir
    )

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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    if accelerator.is_main_process:
        wandb.init(
            project="AIVCdiff",  # 项目名称
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # 运行名称
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.num_train_epochs,
                "batch_size": args.train_batch_size,
                "image_loss_weight": args.image_loss_weight,
                "rna_loss_weight": args.rna_loss_weight,
                # 其他你想记录的配置参数
            }
        )
    
    # 初始化模型
    model = AIVCdiff(
        args,
        pretrained_model_path=args.pretrained_model_path,
        rna_input_dim=args.rna_input_dim,
        drug_latent_dim=args.drug_latent_dim,
        condition_dim=args.condition_dim
    )
    
    # 创建训练集和验证集
    train_dataset = AIVCdiffDataset(
        adata_path=args.train_data,
        adata_ctrl_path=args.ctrl_data,
        perturbation_embedding_path=args.perturbation_embedding_path
    )
    
    valid_dataset = AIVCdiffDataset(
        adata_path=args.valid_data,
        adata_ctrl_path=args.ctrl_data,
        perturbation_embedding_path=args.perturbation_embedding_path
    )
    
    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=args.dataloader_num_workers
    )
    
    # 初始化噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_path,
        subfolder="scheduler"
    )
    
    # 初始化优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
    else:
        optimizer_cls = torch.optim.AdamW
        
    optimizer = optimizer_cls(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    # 如果启用了xformers
    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    # 如果使用梯度检查点
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    max_train_steps = args.num_train_epochs * (len(train_dataset) // args.train_batch_size)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # 准备模型，包括 EMA
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(),
                            model_cls=UNet2DConditionModel,
                            model_config=ema_unet.config)
    else:
        ema_unet = None

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    model.module.vae.to(accelerator.device, dtype=weight_dtype)
    ema_unet.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     tracker_config = dict(vars(args))
    #     accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num validation examples = {len(valid_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    # 初始化epoch和global_step
    global_step = 0
    first_epoch = 0

    # 如果从checkpoint恢复
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            first_epoch = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            first_epoch = int(path.split("-")[1])

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)

    # 在训练开始前进行初始验证
    # logger.info("Running initial validation...")
    # _ = log_validation(
    #     args=args,
    #     accelerator=accelerator,
    #     weight_dtype=weight_dtype,
    #     step=0,
    #     epoch=0,
    #     model=model,
    #     noise_scheduler=noise_scheduler,
    #     valid_dataloader=valid_dataloader
    # )

    # 初始化最佳验证损失和epoch
    best_loss = float('inf')
    best_epoch = -1
    
    # 主训练循环
    for epoch in range(first_epoch, args.num_train_epochs):
        train_one_epoch(
            args=args,
            accelerator=accelerator,
            epoch=epoch,
            model=model,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            lr_scheduler=lr_scheduler,
            ema_unet=ema_unet,
            weight_dtype=weight_dtype,
        )
        
        # # 在训练循环中的验证部分
        # if (epoch+1) % args.save_epoch == 0:
        #     logger.info(f"Running validation at epoch {epoch}...")
        #     val_metrics = log_validation(
        #         args=args,
        #         accelerator=accelerator,
        #         weight_dtype=weight_dtype,
        #         step=global_step,
        #         epoch=epoch,
        #         model=model,
        #         noise_scheduler=noise_scheduler,
        #         valid_dataloader=valid_dataloader
        #     )
            
        #     # 获取验证损失
        #     valid_loss = val_metrics['valid_loss']
            
        #     # 检查是否是最佳模型
        #     if valid_loss < best_loss:
        #         best_loss = valid_loss
        #         best_epoch = epoch
                
        #         best_model_path = os.path.join(args.output_dir, "best_model")
        #         os.makedirs(best_model_path, exist_ok=True)
                
        #         # 保存训练状态
        #         accelerator.save_state(best_model_path)

        #         # noise_scheduler.save_pretrained(best_model_path)

        #         # # save checkpoint
        #         # unet_ckpt = unwrap_model(unet)
        #         # if args.use_ema:
        #         #     ema_unet.copy_to(unet_ckpt.parameters())
                
        #         # feature_extractor = AutoFeatureExtractor.from_pretrained(
        #         #     args.pretrained_vae_path+'/feature_extractor')

        #         # pipeline = CustomStableDiffusionPipeline(
        #         #     vae=accelerator.unwrap_model(vae),
        #         #     text_encoder=None,
        #         #     tokenizer=None,
        #         #     unet=unet_ckpt,
        #         #     scheduler=noise_scheduler,
        #         #     feature_extractor=feature_extractor,
        #         #     safety_checker=None,
        #         # )

        #         # save checkpoint
        #         model_ckpt = unwrap_model(model)
        #         if args.use_ema:
        #             ema_unet.copy_to(model_ckpt.unet.parameters())
                
        #         feature_extractor = AutoFeatureExtractor.from_pretrained(
        #                     args.pretrained_model_path+'/feature_extractor')
                
        #         # 获取原始模型组件
        #         unet_ckpt =model_ckpt.unet
        #         vae_ckpt = model_ckpt.vae
        #         rna_encoder_ckpt = model_ckpt.rna_encoder
        #         rna_decoder_ckpt = model_ckpt.rna_decoder
        #         drug_proj_ckpt = model_ckpt.drug_proj
        #         rna_proj_ckpt = model_ckpt.rna_proj
        #         fusion_layer_ckpt = model_ckpt.fusion_layer
                
        #         # 保存完整pipeline
        #         pipeline = CustomStableDiffusionPipeline(
        #             vae=vae_ckpt,
        #             unet=unet_ckpt,
        #             scheduler=noise_scheduler,
        #             feature_extractor=feature_extractor,
        #             rna_encoder=rna_encoder_ckpt,
        #             rna_decoder=rna_decoder_ckpt,
        #             drug_proj=drug_proj_ckpt,
        #             rna_proj=rna_proj_ckpt,
        #             fusion_layer=fusion_layer_ckpt
        #         )
            
        #         # 保存所有组件
        #         pipeline.save_pretrained(best_model_path)
                
        #         # 保存最佳模型信息
        #         with open(os.path.join(best_model_path, 'training_config.json'), 'w') as f:
        #                     f.write(json.dumps(vars(args), indent=2))
                
        #         logger.info(f"保存最佳模型 (epoch {best_epoch}, validation loss: {best_loss:.4f})")

    best_model_path = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    
    # 保存训练状态
    accelerator.save_state(best_model_path)

    # save checkpoint
    model_ckpt = unwrap_model(model)
    if args.use_ema:
        ema_unet.copy_to(model_ckpt.unet.parameters())
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(
                args.pretrained_model_path+'/feature_extractor')
    
    # 获取原始模型组件
    unet_ckpt =model_ckpt.unet
    vae_ckpt = model_ckpt.vae
    rna_encoder_ckpt = model_ckpt.rna_encoder
    rna_decoder_ckpt = model_ckpt.rna_decoder
    drug_proj_ckpt = model_ckpt.drug_proj
    rna_proj_ckpt = model_ckpt.rna_proj
    fusion_layer_ckpt = model_ckpt.fusion_layer
    
    # 保存完整pipeline
    pipeline = CustomStableDiffusionPipeline(
        vae=vae_ckpt,
        unet=unet_ckpt,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        rna_encoder=rna_encoder_ckpt,
        rna_decoder=rna_decoder_ckpt,
        drug_proj=drug_proj_ckpt,
        rna_proj=rna_proj_ckpt,
        fusion_layer=fusion_layer_ckpt
    )

    # 保存所有组件
    pipeline.save_pretrained(best_model_path)
    
    accelerator.end_training()

if __name__ == "__main__":
    main()