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
import anndata as ad
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
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


if is_wandb_available():
    import wandb
    os.environ['WANDB_DIR'] = "tmp/"
    # os.environ["WANDB_MODE"] = "dryrun"


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    # Removed default mapping as we are not using datasets hub anymore
    # "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


class CustomStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae,
                 text_encoder,
                 unet,
                 scheduler,
                 feature_extractor,
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
                         requires_safety_checker=requires_safety_checker)
        self.custom_text_encoder = text_encoder

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        embeddings = self.custom_text_encoder(prompt)
        embeddings = embeddings.to(device)
        return embeddings, None


def log_validation(args, accelerator, weight_dtype, step, ckpt_path):
    """ Log validation images to tensorboard and wandb.
    
    Args:
        args (argparse.Namespace): The parsed arguments.
        accelerator (Accelerator): The Accelerator object.
        weight_dtype (str): The weight dtype used for training.
        step (int): The current training step.
        ckpt_path (str): The path to the checkpoint.
        
    Returns:
        List[torch.Tensor]: The validation images.
    """
    logger.info("Running validation... ")

    if args.pretrained_vae_path != ckpt_path:
        if not os.path.exists(ckpt_path+'/feature_extractor'):
            os.makedirs(ckpt_path+'/feature_extractor')
        shutil.copyfile(
            args.pretrained_vae_path+'/feature_extractor/preprocessor_config.json',
            ckpt_path+'/feature_extractor/preprocessor_config.json')
        unet = UNet2DConditionModel.from_pretrained(
            ckpt_path, subfolder="unet_ema", use_auth_token=True)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            ckpt_path, subfolder="unet", use_auth_token=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        ckpt_path+'/feature_extractor')

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_path,
        subfolder="vae")

    noise_scheduler = DDPMScheduler.from_pretrained(
        ckpt_path, subfolder="scheduler")

    custom_text_encoder = PerturbationEncoderInference(
        args.dataset_id, args.naive_conditional, 'SD')

    pipeline = CustomStableDiffusionPipeline(
        vae=vae,
        unet=unet,
        text_encoder=custom_text_encoder,
        feature_extractor=feature_extractor,
        scheduler=noise_scheduler)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(
            args.seed)

    validation_path = args.output_dir+"/checkpoint-"+str(step) +\
        "/validation/"
    if not os.path.exists(validation_path):
        os.makedirs(validation_path,exist_ok=True)

    images = []
    updated_validation_prompts = []
    for i in range(len(args.validation_prompts)):
        for j in range(3):
            with torch.autocast("cuda"):
                image = pipeline(
                    args.validation_prompts[i],
                    generator=generator).images[0]
            images.append(image)
            updated_validation_prompts.append(args.validation_prompts[i]+'-'+str(j))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(
                            image,
                            caption=f"{i}: {updated_validation_prompts[i]} - step {step}",)
                        for i, image in enumerate(images)
                    ]
                }
            )

        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


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


# +++ Added H5adDataset +++
class H5adDataset(Dataset):
    def __init__(self, adata_path, image_path_column, perturb_id_column, transform=None, image_root_dir=""):
        """
        Args:
            adata_path (str): Path to the h5ad file.
            image_path_column (str): Column name in adata.obs for image paths.
            perturb_id_column (str): Column name in adata.obs for perturbation IDs.
            transform (callable, optional): Optional transform to be applied on a sample's image.
            image_root_dir (str, optional): Root directory to prepend to relative image paths. Defaults to "".
        """
        self.adata_path = adata_path
        try:
            self.adata = ad.read_h5ad(adata_path)
            logger.info(f"Successfully loaded AnnData from {adata_path} with {len(self.adata)} observations.")
        except Exception as e:
            logger.error(f"Failed to load AnnData file {adata_path}: {e}")
            raise # Reraise the exception to stop execution if file loading fails

        self.image_path_column = image_path_column
        self.perturb_id_column = perturb_id_column
        self.transform = transform
        self.image_root_dir = image_root_dir

        # Validate column existence
        if self.image_path_column not in self.adata.obs.columns:
            raise ValueError(f"Image path column '{self.image_path_column}' not found in AnnData obs.")
        if self.perturb_id_column not in self.adata.obs.columns:
             raise ValueError(f"Perturbation ID column '{self.perturb_id_column}' not found in AnnData obs.")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        if idx >= len(self.adata):
             raise IndexError("Index out of bounds")
             
        image_rel_path = self.adata.obs[self.image_path_column].iloc[idx]
        # Construct full path if image_root_dir is provided
        image_path = os.path.join(self.image_root_dir, image_rel_path) if self.image_root_dir else image_rel_path
        
        try:
            # Ensure path separators are correct for the OS
            image_path = os.path.normpath(image_path)
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.warning(f"Image file not found at {image_path}. Skipping sample index {idx}.")
            return None # Signal to collate_fn to skip this sample
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}. Skipping sample index {idx}.")
            return None # Signal to collate_fn to skip this sample

        perturb_id = str(self.adata.obs[self.perturb_id_column].iloc[idx]) # Ensure it's a string

        sample = {"image": image, "perturb_id": perturb_id}

        if self.transform:
            sample["image"] = self.transform(sample["image"]) # Now image is a tensor

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
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_path,
            subfolder="vae", revision=args.revision, variant=args.variant
        )

    if args.pretrained_model_name_or_path == args.pretrained_vae_path:
        
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet", revision=args.non_ema_revision
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet_ema", revision=args.non_ema_revision
        )

    # Freeze vae and set unet to trainable
    vae.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_model_path = args.pretrained_model_name_or_path
        ema_subfolder = "unet" if args.pretrained_model_name_or_path == args.pretrained_vae_path else "unet_ema"
        try:
            ema_unet = UNet2DConditionModel.from_pretrained(
                ema_model_path, subfolder=ema_subfolder, revision=args.revision, variant=args.variant
            )
            ema_unet = EMAModel(
                 # Pass the parameters of the *trainable* unet model to EMA
                 ema_unet.parameters(), 
                 model_cls=UNet2DConditionModel, 
                 model_config=ema_unet.config
                 )
            logger.info("EMA model created successfully.")
        except Exception as e:
             logger.error(f"Failed to create EMA model from {ema_model_path}/{ema_subfolder}: {e}")
             args.use_ema = False # Disable EMA if creation fails
             ema_unet = None
             logger.warning("EMA disabled due to error during initialization.")
    else:
         ema_unet = None # Explicitly set to None if not using EMA

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema and ema_unet is not None:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema and ema_unet is not None:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

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
        filter(lambda p: p.requires_grad, unet.parameters()), # Ensure optimizer only optimizes trainable parameters
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # --- Removed old dataset loading logic ---
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     dataset = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #         cache_dir=args.cache_dir,
    #         data_dir=args.train_data_dir,
    #     )
    # else:
    #     data_files = {}
    #     if args.train_data_dir is not None:
    #         data_files["train"] = os.path.join(args.train_data_dir, "**")
    #     dataset = load_dataset(
    #         "imagefolder",
    #         data_files=data_files,
    #         cache_dir=args.cache_dir,
    #     )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # column_names = dataset["train"].column_names # Removed

    # 6. Get the column names for input/target.
    # dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None) # Removed
    # if args.image_column is None:
    #     image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    # else:
    #     image_column = args.image_column
    #     if image_column not in column_names:
    #         raise ValueError(
    #             f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
    #         )
    # if args.caption_column is None:
    #     caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    # else:
    #     caption_column = args.caption_column
    #     if caption_column not in column_names:
    #         raise ValueError(
    #             f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
    #         ) # --- End Removed old dataset loading logic ---

    # Define image transformations
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]) # Assuming single channel normalization? If RGB, use [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # def preprocess_train(examples): # Removed
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     examples["pixel_values"] = [train_transforms(image) for image in images]
    #     return examples

    # Create the H5adDataset
    with accelerator.main_process_first():
        try:
            train_dataset = H5adDataset(
                adata_path=args.train_data_path,
                image_path_column=args.image_path_column,
                perturb_id_column=args.perturb_id_column,
                transform=train_transforms,
                image_root_dir=args.image_root_dir
            )
            logger.info(f"Successfully created H5adDataset with {len(train_dataset)} samples.")
        except (ValueError, FileNotFoundError) as e:
             logger.error(f"Error creating H5adDataset: {e}")
             # Exit if dataset creation fails crucially
             exit(1) 
        # Removed max_train_samples logic as it's easier to handle by slicing the h5ad file beforehand if needed.
        # if args.max_train_samples is not None:
        #     dataset["train"] = dataset["train"].shuffle(
        #         seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        # train_dataset = dataset["train"].with_transform(preprocess_train) # Removed


    # +++ Updated collate_fn +++
    def collate_fn(examples):
        # Filter out None samples caused by loading errors in __getitem__
        original_count = len(examples)
        examples = [e for e in examples if e is not None]
        filtered_count = len(examples)
        
        if original_count != filtered_count:
             logger.warning(f"Filtered out {original_count - filtered_count} samples due to loading errors in this batch.")

        if not examples:
            # Return an empty dictionary or None if the whole batch failed
            # Returning None might be simpler for the training loop to handle.
            return None 

        # 'image' is already a transformed tensor from H5adDataset
        pixel_values = torch.stack([example["image"] for example in examples]) 
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # Get perturbation IDs
        perturb_ids = [example["perturb_id"] for example in examples]
        
        # Encode prompts (perturb_ids) one by one
        # This assumes encode_prompt is designed to take a single string identifier
        # and returns the embedding (e.g., shape [1, 77, 768])
        prompt_embeds = []
        for pid in perturb_ids:
            try:
                # Note: encode_prompt uses global dataset_id and naive_conditional
                embedding = encode_prompt(pid) 
                prompt_embeds.append(embedding)
            except Exception as e:
                logger.error(f"Error encoding prompt for ID '{pid}': {e}")
                # Handle error: maybe add a default embedding or skip? For now, let's error out if critical.
                # If non-critical, could append a zero tensor of correct shape?
                # Example: prompt_embeds.append(torch.zeros(1, 77, 768)) 
                # Let's re-raise for now to ensure visibility
                raise RuntimeError(f"Failed to encode prompt for ID '{pid}'. See previous errors.") from e
        
        if len(prompt_embeds) != len(examples):
             # This case shouldn't happen with the current error handling, but good check
             raise RuntimeError("Mismatch between number of samples and successfully encoded prompts.")

        # Stack the embeddings along the batch dimension
        # Assumes each embedding is shape [1, 77, 768]
        input_ids = torch.cat(prompt_embeds, dim=0) # Use torch.cat, not torch.stack

        # Final check of shapes
        # Expected shape for pixel_values: [batch_size, channels, height, width]
        # Expected shape for input_ids: [batch_size, 77, 768]
        # print("pixel_values shape:", pixel_values.shape) # Debug
        # print("input_ids shape:", input_ids.shape) # Debug

        return {"pixel_values": pixel_values, "input_ids": input_ids}
    # --- End Updated collate_fn ---


    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
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
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
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

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # --- MOVED BLOCK: Set tracker project name from dataset_id if using wandb ---
        if args.dataset_id and (args.report_to == "wandb" or args.report_to == "all"):
            if args.tracker_project_name is None: # Only set if not explicitly provided
                 args.tracker_project_name = args.dataset_id
                 logger.info(f"Using dataset_id '{args.dataset_id}' as wandb project name.")
            else:
                 logger.info(f"Using explicitly provided wandb project name: '{args.tracker_project_name}'")
        elif (args.report_to == "wandb" or args.report_to == "all") and args.tracker_project_name is None:
            # Fallback if dataset_id is None but wandb is used and no project name given
            args.tracker_project_name = "morphodiff-default-project"
            logger.warning(f"dataset_id not provided and tracker_project_name not set. Using default wandb project: '{args.tracker_project_name}'")
        # --- End MOVED BLOCK ---

        tracker_config = dict(vars(args))
        # Remove sensitive or large items if necessary, e.g., validation_prompts if very long
        if "validation_prompts" in tracker_config:
            tracker_config.pop("validation_prompts") 
        try:
             accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except Exception as e:
             logger.error(f"Failed to initialize trackers: {e}")

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
    global_step = 0
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
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

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
    total_trained_steps = args.trained_steps

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train() # Make sure model is in train mode at the start of each epoch
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip batch if collate_fn returned None due to errors
            if batch is None: 
                logger.warning(f"Skipping training step {step} in epoch {epoch} due to data loading errors in the batch.")
                # We might need to adjust the progress bar if we skip steps frequently
                # Consider if args.max_train_steps should account for skipped steps
                continue 
                
            with accelerator.accumulate(unet):
                # if generate_img_step0_sign:
                #     if accelerator.is_main_process: # Log validation only on main process
                #         log_validation(
                #             args,
                #             accelerator,
                #             weight_dtype,
                #             args.trained_steps,
                #             args.pretrained_model_name_or_path
                #         )
                #         generate_img_step0_sign = False
                    
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(
                        latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(
                        latents, noise, timesteps)

                # Get the text embedding for conditioning
                # Should be shape [bs, 77, 768] from collate_fn
                encoder_hidden_states = batch["input_ids"].to(accelerator.device, dtype=weight_dtype) # Ensure dtype and device

                # --- Conditional Check Logic ---
                # The original check might need adjustment depending on how embeddings are generated.
                # If naive means a specific embedding (e.g., all zeros or ones), check for that pattern.
                # If conditional means *not* that specific pattern, check for that.
                # Assuming '1.0' was just a placeholder for a specific naive embedding:
                # Example check if the naive embedding is all zeros:
                # if args.naive_conditional == 'naive':
                #     assert torch.all(encoder_hidden_states == 0.), \
                #         "encoder_hidden_states should be all zeros for naive SD"
                # elif args.naive_conditional == 'conditional':
                #     # check that the encoder_hidden_states are not all zeros
                #     assert not torch.all(encoder_hidden_states == 0.), \
                #         "encoder_hidden_states should not be all zeros for MorphoDiff"
                # Let's keep the original logic for now, assuming 1.0 is the intended check value.
                if args.naive_conditional == 'naive':
                     # Check if the mean is close to 1.0, allowing for floating point inaccuracy
                     is_all_ones = torch.allclose(encoder_hidden_states, torch.ones_like(encoder_hidden_states))
                     assert is_all_ones, \
                         f"encoder_hidden_states should be all ones for naive SD, but got mean {encoder_hidden_states.mean().item()}"
                elif args.naive_conditional == 'conditional':
                     # check that the encoder_hidden_states are not all ones
                     is_all_ones = torch.allclose(encoder_hidden_states, torch.ones_like(encoder_hidden_states))
                     assert not is_all_ones, \
                         f"encoder_hidden_states should not be all ones for MorphoDiff, but they appear to be."
                # --- End Conditional Check ---

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states,
                    return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(),
                        target.float(),
                        reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema and ema_unet is not None: # Check if ema_unet exists
                     # Pass unwrapped unet parameters to EMA
                     # ema_unet.step(unet.parameters()) # Original
                     # Pass parameters of the *accelerator-prepared* model
                     # Need to unwrap first if using deepspeed/fsdp?
                     # Let's assume accelerator handles unwrapping internally if needed for EMA step.
                     # Check EMA documentation for use with Accelerate.
                     # Usually, you step EMA with the *original* model's parameters.
                     # Let's try passing the unwrapped model's parameters.
                     ema_unet.step(accelerator.unwrap_model(unet).parameters()) 

                progress_bar.update(1)
                global_step += 1
                # Log training loss
                accelerator.log({"train_loss": train_loss}, step=global_step) # Use global_step for logging
                
                # Log learning rate
                if lr_scheduler is not None:
                     lr = lr_scheduler.get_last_lr()[0]
                     accelerator.log({"lr": lr}, step=global_step)
                     
                train_loss = 0.0 # Reset accumulated loss
                total_trained_steps = global_step + args.trained_steps # Calculate total steps including previous runs
                # print('global_step:', global_step, 'total_trained_steps:', total_trained_steps) # Debug print

                # --- Checkpointing Logic ---
                # Check if checkpointing is due based on total_trained_steps and checkpointing_steps
                # Also checkpoint at the very last step
                should_checkpoint = (total_trained_steps % args.checkpointing_steps == 0) or \
                                    (global_step >= args.max_train_steps)
                
                if should_checkpoint:
                    if accelerator.is_main_process:
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
                        accelerator.save_state(save_path)
                        logger.info(f"Saved accelerator state to {save_path}")

                        # --- Save Model Components Manually for Pipeline ---
                        logger.info("Saving model components for pipeline...")
                        unet_ckpt = accelerator.unwrap_model(unet) # Unwrap the model

                        # Apply EMA weights to the unwrapped model if EMA is used
                        if args.use_ema and ema_unet is not None:
                            logger.info("Applying EMA weights to saved model.")
                            # Store current weights
                            # current_state_dict = {k: v.clone() for k, v in unet_ckpt.state_dict().items()}
                            ema_unet.copy_to(unet_ckpt.parameters()) 

                        # Save UNet
                        unet_save_path = os.path.join(save_path, "unet")
                        unet_ckpt.save_pretrained(unet_save_path)
                        logger.info(f"Saved UNet model to {unet_save_path}")
                        
                        # Restore original weights if EMA was applied temporarily for saving
                        # if args.use_ema and ema_unet is not None:
                        #    unet_ckpt.load_state_dict(current_state_dict)
                        #    logger.info("Restored original model weights after EMA save.")


                        # Save Scheduler
                        noise_scheduler.save_pretrained(os.path.join(save_path, "scheduler"))
                        logger.info(f"Saved Scheduler config to {os.path.join(save_path, 'scheduler')}")

                        # Save VAE config (VAE is frozen, so no need to save weights unless fine-tuned)
                        # We need the VAE config for the pipeline, but weights come from original path
                        vae_config_save_path = os.path.join(save_path, "vae")
                        os.makedirs(vae_config_save_path, exist_ok=True)
                        # Save VAE config - Assuming vae is the loaded AutoencoderKL instance
                        # Need to unwrap VAE as well if prepared by accelerator? Usually frozen models aren't.
                        # Let's assume vae is not wrapped or doesn't need unwrapping here.
                        try:
                             # Save the config file
                             vae.save_config(vae_config_save_path)
                             logger.info(f"Saved VAE config to {vae_config_save_path}")
                             # We might also need the feature extractor config if it's not standard
                             feature_extractor_save_path = os.path.join(save_path, "feature_extractor")
                             os.makedirs(feature_extractor_save_path, exist_ok=True)
                             # Load feature extractor from original path to save its config
                             # This assumes it's standard and loaded during validation - maybe load it once here?
                             try:
                                 # Need to define feature_extractor earlier or pass path via args
                                 # Let's load it from the VAE path as per validation logic
                                 feature_extractor_orig_path = os.path.join(args.pretrained_vae_path, 'feature_extractor')
                                 if os.path.isdir(feature_extractor_orig_path):
                                     feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_orig_path)
                                     feature_extractor.save_pretrained(feature_extractor_save_path)
                                     logger.info(f"Saved Feature Extractor config to {feature_extractor_save_path}")
                                 else:
                                     logger.warning(f"Feature extractor directory not found at {feature_extractor_orig_path}, cannot save config.")
                             except Exception as e:
                                 logger.error(f"Could not load or save feature extractor config: {e}")

                        except Exception as e:
                             logger.error(f"Error saving VAE/Feature Extractor config: {e}")


                        # Save training arguments
                        try:
                            args_dict = vars(args)
                            # Convert Path objects to strings for JSON serialization
                            for key, value in args_dict.items():
                                if isinstance(value, Path):
                                    args_dict[key] = str(value)
                            
                            with open(os.path.join(save_path, 'training_args.json'), 'w') as f:
                                json.dump(args_dict, f, indent=2)
                            logger.info(f"Saved training arguments to {os.path.join(save_path, 'training_args.json')}")
                        except Exception as e:
                            logger.error(f"Failed to save training arguments: {e}")


                        # --- Deprecated Pipeline Saving within checkpoint ---
                        # pipeline = StableDiffusionPipeline(
                        #     vae=accelerator.unwrap_model(vae), # Use unwrapped VAE
                        #     text_encoder=None, # No text encoder in this setup
                        #     tokenizer=None, # No tokenizer
                        #     unet=unet_ckpt, # Use the potentially EMA-applied UNet
                        #     scheduler=noise_scheduler, # Use the current noise scheduler
                        #     feature_extractor=feature_extractor, # Need feature extractor instance
                        #     safety_checker=None, # No safety checker
                        # )
                        # try:
                        #     pipeline.save_pretrained(save_path)
                        #     logger.info(f"Saved full pipeline to {save_path}")
                        # except Exception as e:
                        #     logger.error(f"Failed to save pipeline: {e}")
                        # --- End Deprecated Pipeline Saving ---

                        # # Run validation using the *saved checkpoint path*
                        # if (args.validation_prompts is not None):
                        #      # Check if validation should run at this step
                        #      is_validation_step = (total_trained_steps % args.validation_epochs == 0) or \
                        #                           (global_step >= args.max_train_steps)
                             
                        #      if is_validation_step:
                        #          logger.info(f"Running validation at total trained step {total_trained_steps}.")
                        #          try:
                        #              # Pass the save_path (checkpoint directory) to log_validation
                        #              log_validation(
                        #                  args,
                        #                  accelerator,
                        #                  weight_dtype,
                        #                  total_trained_steps, # Pass total steps for logging clarity
                        #                  save_path # Pass the path to the saved checkpoint
                        #              )
                        #          except Exception as e:
                        #              logger.error(f"Validation failed at step {total_trained_steps}: {e}")
                            
                        # write checkpoint info to the log file
                        try:
                            # Ensure file exists and write header if needed (idempotent check)
                            if not os.path.exists(args.checkpointing_log_file):
                                with open(args.checkpointing_log_file, "w") as f:
                                     # Define header based on expected columns
                                     header = "dataset_id,log_dir,pretrained_model_dir,checkpoint_dir,seed,trained_steps,checkpoint_number\n"
                                     f.write(header)
                                     logger.info(f"Created checkpoint log file: {args.checkpointing_log_file}")

                            with open(args.checkpointing_log_file, "a") as f:
                                f.write(f"{args.dataset_id or 'N/A'},"
                                        f"{args.logging_dir},"
                                        f"{args.pretrained_model_name_or_path}," # Log the initial base model path
                                        f"{save_path}," # Log the specific checkpoint path saved
                                        f"{args.seed},"
                                        f"{total_trained_steps}," # Log total steps reached at this checkpoint
                                        f"{args.checkpoint_number or 'N/A'}\n") # Log the provided checkpoint number if any
                                logger.info(f"Appended checkpoint info to {args.checkpointing_log_file}")
                        except Exception as e:
                             logger.error(f"Failed to write to checkpoint log file {args.checkpointing_log_file}: {e}")

            # Log step loss and learning rate (already logged inside sync_gradients block)
            # logs = {"step_loss": loss.detach().item(),
            #         "lr": lr_scheduler.get_last_lr()[0]}
            # progress_bar.set_postfix(**logs)

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
    
    # Final Checkpointing and Validation (optional, could be redundant if last step checkpointed)
    # Consider if a final save/validation outside the loop is needed
    
    # Clean up trackers
    accelerator.end_training()
    logger.info("Accelerator ended training.")

if __name__ == "__main__":
    main()