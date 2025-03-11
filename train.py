#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

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
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator,InitProcessGroupKwargs
from datetime import timedelta
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
from models import AIVCdiff
from datasets import AIVCdiffDataset



if is_wandb_available():
    import wandb
    os.environ['WANDB_DIR'] = "tmp/"
    # os.environ["WANDB_MODE"] = "dryrun"


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


class CustomStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae,
                 unet,
                 scheduler,
                 feature_extractor,
                 text_encoder=None,
                 rna_encoder=None,
                 rna_decoder=None,
                 drug_proj=None,
                 rna_proj=None,
                 fusion_layer=None,
                 tokenizer=None,
                 safety_checker=None,
                 image_encoder=None,
                 requires_safety_checker=False):
        super().__init__(vae=vae,
                         text_encoder=text_encoder,
                         tokenizer=tokenizer,
                         unet=unet,
                         scheduler=scheduler,
                         safety_checker=safety_checker,
                         feature_extractor=feature_extractor,
                         image_encoder=image_encoder,
                         requires_safety_checker=requires_safety_checker,
                         rna_encoder=rna_encoder,
                         rna_decoder=rna_decoder,
                         drug_proj=drug_proj,
                         rna_proj=rna_proj,
                         fusion_layer=fusion_layer,
                         )
        # self.custom_text_encoder = text_encoder

        # super().__init__()
        
        # self.register_modules(
        #     vae=vae,
        #     unet=unet,
        #     scheduler=scheduler,
        #     feature_extractor=feature_extractor,
        #     rna_encoder=rna_encoder,
        #     rna_decoder=rna_decoder,
        #     drug_proj=drug_proj,
        #     rna_proj=rna_proj,
        #     fusion_layer=fusion_layer
        # )

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

    # def encode_prompt(
    #     self,
    #     prompt,
    #     device,
    #     num_images_per_prompt=1,
    #     do_classifier_free_guidance=False,
    #     negative_prompt=None,
    #     prompt_embeds: Optional[torch.FloatTensor] = None,
    #     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     lora_scale: Optional[float] = None,
    #     clip_skip: Optional[int] = None,
    # ):
    #     embeddings = self.custom_text_encoder(prompt)
    #     embeddings = embeddings.to(device)
    #     return embeddings, None


# def log_validation(args, accelerator, weight_dtype, step, ckpt_path):
#     """ Log validation images to tensorboard and wandb.
    
#     Args:
#         args (argparse.Namespace): The parsed arguments.
#         accelerator (Accelerator): The Accelerator object.
#         weight_dtype (str): The weight dtype used for training.
#         step (int): The current training step.
#         ckpt_path (str): The path to the checkpoint.
        
#     Returns:
#         List[torch.Tensor]: The validation images.
#     """
#     logger.info("Running validation... ")

#     if args.pretrained_vae_path != ckpt_path:
#         if not os.path.exists(ckpt_path+'/feature_extractor'):
#             os.makedirs(ckpt_path+'/feature_extractor')
#         shutil.copyfile(
#             args.pretrained_vae_path+'/feature_extractor/preprocessor_config.json',
#             ckpt_path+'/feature_extractor/preprocessor_config.json')
#         unet = UNet2DConditionModel.from_pretrained(
#             ckpt_path, subfolder="unet_ema", use_auth_token=True)
#     else:
#         unet = UNet2DConditionModel.from_pretrained(
#             ckpt_path, subfolder="unet", use_auth_token=True)

#     feature_extractor = AutoFeatureExtractor.from_pretrained(
#         ckpt_path+'/feature_extractor')

#     vae = AutoencoderKL.from_pretrained(
#         args.pretrained_vae_path,
#         subfolder="vae")

#     noise_scheduler = DDPMScheduler.from_pretrained(
#         ckpt_path, subfolder="scheduler")

#     custom_text_encoder = PerturbationEncoderInference(
#         args.dataset_id, args.naive_conditional, 'SD')

#     pipeline = CustomStableDiffusionPipeline(
#         vae=vae,
#         unet=unet,
#         text_encoder=custom_text_encoder,
#         feature_extractor=feature_extractor,
#         scheduler=noise_scheduler)

#     pipeline = pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=True)

#     if args.enable_xformers_memory_efficient_attention:
#         pipeline.enable_xformers_memory_efficient_attention()

#     if args.seed is None:
#         generator = None
#     else:
#         generator = torch.Generator(device=accelerator.device).manual_seed(
#             args.seed)

#     validation_path = args.output_dir+"/checkpoint-"+str(step) +\
#         "/validation/"
#     if not os.path.exists(validation_path):
#         os.makedirs(validation_path)

#     images = []
#     updated_validation_prompts = []
#     for i in range(len(args.validation_prompts)):
#         for j in range(4):
#             with torch.autocast("cuda"):
#                 image = pipeline(
#                     args.validation_prompts[i],
#                     generator=generator).images[0]
#             images.append(image)
#             updated_validation_prompts.append(args.validation_prompts[i]+'-'+str(j))

#     for tracker in accelerator.trackers:
#         if tracker.name == "tensorboard":
#             np_images = np.stack([np.asarray(img) for img in images])
#             tracker.writer.add_images(
#                 "validation", np_images, step, dataformats="NHWC")
#         elif tracker.name == "wandb":
#             tracker.log(
#                 {
#                     "validation": [
#                         wandb.Image(
#                             image,
#                             caption=f"{i}: {updated_validation_prompts[i]} - step {step}",)
#                         for i, image in enumerate(images)
#                     ]
#                 }
#             )

#         else:
#             logger.warn(f"image logging not implemented for {tracker.name}")

#     del pipeline
#     torch.cuda.empty_cache()

#     return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_data",
        type=str
    )
    parser.add_argument(
        "--valid_data",
        type=str
    )
    parser.add_argument(
        "--ctrl_data",
        type=str
    )
    parser.add_argument(
        "--perturbation_embedding_path",
        type=str
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="model/stable-diffusion-v1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--rna_input_dim", type=int, default=977,
                       help="Input dimension of RNA expression data")
    parser.add_argument("--drug_latent_dim", type=int, default=193,
                       help="Latent dimension of drug embeddings")
    parser.add_argument("--condition_dim", type=int, default=768,
                       help="Dimension of condition embeddings")
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


    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1."
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
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
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
        "--train_data_dir",
        type=str,
        default="datasets/BBBC021/",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="additional_feature",
        help="The column of the dataset containing a caption or a list of captions.",
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
        default="text2image-fine-tune",
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.report_to == 'wandb':
        args.tracker_project_name = args.output_dir.split('/')[-1] 
    args.validation_prompts = args.validation_prompts[0].split(',')

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
    prompt_embeds = encoder.get_perturbation_embedding(identifier)
    # shape (bs, 77, 768)
    return prompt_embeds


def main():
    args = parse_args()

    global dataset_id
    dataset_id = args.dataset_id

    global naive_conditional
    naive_conditional = args.naive_conditional

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

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir,
                                                      logging_dir=args.logging_dir)

    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[init_process_group_kwargs]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        wandb.init(
            project="AIVCdiff",  # 项目名称
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # 运行名称
        )

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_path,
        subfolder="scheduler")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    
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

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
       
        ema_unet = EMAModel(ema_unet.parameters(),
                            model_cls=UNet2DConditionModel,
                            model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()
       

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        # def save_model_hook(models, weights, output_dir):
        #     if accelerator.is_main_process:
        #         if args.use_ema:
        #             ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        #         for i, model in enumerate(models):
        #             model.save_pretrained(os.path.join(output_dir, "unet"))

        #             # make sure to pop weight so that corresponding model is not saved again
        #             weights.pop()

        # def load_model_hook(models, input_dir):
        #     if args.use_ema:
        #         load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
        #         ema_unet.load_state_dict(load_model.state_dict())
        #         ema_unet.to(accelerator.device)
        #         del load_model

        #     for i in range(len(models)):
        #         # pop models so that they are not loaded again
        #         model = models.pop()

        #         # load diffusers style into model
        #         load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
        #         model.register_to_config(**load_model.config)

        #         model.load_state_dict(load_model.state_dict())
        #         del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    # 保存UNet
                    model.unet.save_pretrained(os.path.join(output_dir, "unet"))
                    
                    # 保存RNA编码器和解码器
                    torch.save(model.rna_encoder.state_dict(), os.path.join(output_dir, "rna_encoder.pt"))
                    torch.save(model.rna_decoder.state_dict(), os.path.join(output_dir, "rna_decoder.pt"))
                    
                    # 保存投影层和融合层
                    torch.save({
                        'drug_proj': model.drug_proj.state_dict(),
                        'rna_proj': model.rna_proj.state_dict(),
                        'fusion_layer': model.fusion_layer.state_dict()
                    }, os.path.join(output_dir, "projection_layers.pt"))

                    # 确保权重不会被重复保存
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # 弹出模型以避免重复加载
                model = models.pop()

                # 加载UNet
                load_unet = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.unet.register_to_config(**load_unet.config)
                model.unet.load_state_dict(load_unet.state_dict())
                del load_unet

                # 加载RNA编码器和解码器
                model.rna_encoder.load_state_dict(
                    torch.load(os.path.join(input_dir, "rna_encoder.pt"))
                )
                model.rna_decoder.load_state_dict(
                    torch.load(os.path.join(input_dir, "rna_decoder.pt"))
                )

                # 加载投影层和融合层
                projection_states = torch.load(os.path.join(input_dir, "projection_layers.pt"))
                model.drug_proj.load_state_dict(projection_states['drug_proj'])
                model.rna_proj.load_state_dict(projection_states['rna_proj'])
                model.fusion_layer.load_state_dict(projection_states['fusion_layer'])

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

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

    optimizer = optimizer_cls(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
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

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     tracker_config = dict(vars(args))
    #     tracker_config.pop("validation_prompts")
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
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    first_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            trained_step = int(path.split("-")[1])

            initial_global_step = trained_step
            first_epoch = trained_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
        trained_step = -1

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    generate_img_step0_sign = True
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    model.train()

    with open('./model_grad_status.txt', 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"Layer: {name}, requires_grad: {param.requires_grad}\n")
    
    gaussian_criterion = torch.nn.GaussianNLLLoss()

    for epoch in range(first_epoch, args.num_train_epochs):
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
                
                # 高斯损失
                guss_loss = gaussian_criterion(
                    gene_means,
                    target,
                    gene_vars
                )
                rna_loss += guss_loss
                
                # 3. 总损失
                loss = args.image_loss_weight * image_loss + args.rna_loss_weight * rna_loss
                
                # Gather all losses across all processes
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_image_loss = accelerator.gather((args.image_loss_weight * image_loss).repeat(args.train_batch_size)).mean()
                avg_rna_loss = accelerator.gather((args.rna_loss_weight * rna_loss).repeat(args.train_batch_size)).mean()

                # 累积loss
                train_loss = avg_loss.item() / args.gradient_accumulation_steps
                train_image_loss = avg_image_loss.item() / args.gradient_accumulation_steps
                train_rna_loss = avg_rna_loss.item() / args.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # 记录日志
                logs = {
                    "loss": train_loss,
                    "image_loss": train_image_loss,
                    "rna_loss": train_rna_loss,
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                
                # 记录到wandb
                if accelerator.is_main_process:
                    # 使用 wandb 记录
                    wandb.log(logs)
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(model.module.unet.parameters())
                progress_bar.update(1)
                trained_step += 1
                accelerator.log({"train_loss": train_loss}, step=trained_step)

                if (trained_step % args.checkpointing_steps == 0) | (trained_step == args.max_train_steps-1):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
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
                            args.output_dir, f"checkpoint-{trained_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

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
                        pipeline.save_pretrained(save_path)

                        # # write in the args.checkpointing_log_file file
                        # with open(args.checkpointing_log_file, "a") as f:
                        #     f.write(args.dataset_id+','+args.logging_dir+','+args.pretrained_model_path+','+save_path+',' +
                        #             str(args.seed)+','+str(trained_step)+','+str(args.checkpoint_number)+"\n")
                    
                    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()