# 导入必要的库
# diffusers库中的核心组件，用于构建和运行Stable Diffusion模型
from diffusers import StableDiffusionPipeline, DDPMScheduler
import os
import torch
import datetime
import argparse
# transformers库中的图像处理器，用于处理输入图像
from transformers import AutoImageProcessor
# diffusers中的VAE和UNet模型组件
from diffusers import AutoencoderKL, UNet2DConditionModel
# 自定义的扰动编码器
from perturbation_encoder import PerturbationEncoderInference
from typing import Optional
import pandas as pd
import random
import numpy as np
import scanpy as sc
from tqdm import tqdm

def str2bool(v):
    """将字符串转换为布尔值的辅助函数
    
    Args:
        v (str): 需要转换的字符串
    
    Returns:
        bool: 转换后的布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'TRUE', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'FALSE', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class CustomStableDiffusionPipeline(StableDiffusionPipeline):
    """自定义的Stable Diffusion Pipeline类
    
    继承自StableDiffusionPipeline，主要修改了文本编码部分，使用自定义的编码器
    """
    def __init__(self,
                 vae,  # VAE模型，用于图像的编码和解码
                 text_encoder,  # 文本编码器，将文本转换为潜在空间的表示
                 unet,  # U-Net模型，核心的扩散模型网络
                 scheduler,  # 调度器，控制扩散过程
                 feature_extractor,  # 特征提取器，用于处理输入
                 tokenizer=None,  # 分词器（这里不使用）
                 safety_checker=None,  # 安全检查器（这里不使用）
                 image_encoder=None,  # 图像编码器（这里不使用）
                 requires_safety_checker=False):  # 是否需要安全检查
        super().__init__(vae=vae,
                         text_encoder=text_encoder,
                         tokenizer=tokenizer,
                         unet=unet,
                         scheduler=scheduler,
                         safety_checker=safety_checker,
                         feature_extractor=feature_extractor,
                         image_encoder=image_encoder,
                         requires_safety_checker=requires_safety_checker)
        # 保存自定义的文本编码器
        self.custom_text_encoder = text_encoder

    def encode_prompt(
        self,
        prompt,  # 输入的提示文本
        device,  # 运行设备（CPU/GPU）
        num_images_per_prompt=1,  # 每个提示生成的图像数量
        do_classifier_free_guidance=False,  # 是否使用分类器引导
        negative_prompt=None,  # 负面提示词
        prompt_embeds: Optional[torch.FloatTensor] = None,  # 预计算的提示词嵌入
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 预计算的负面提示词嵌入
        lora_scale: Optional[float] = None,  # LoRA缩放因子
        clip_skip: Optional[int] = None,  # CLIP跳过层数
    ):
        """重写的提示词编码方法
        
        使用自定义的文本编码器处理提示词，而不是原始的CLIP编码器
        """
        # 添加调试信息
        print(f"Encoding prompt: {prompt}")
        
        # 使用自定义编码器处理提示词
        embeddings = self.custom_text_encoder(prompt)
        
        # 检查embeddings是否为None
        if embeddings is None:
            raise ValueError("Text encoder returned None embeddings")
            
        # print(f"Embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")
        
        # 将嵌入移动到指定设备
        embeddings = embeddings.to(device)
        
        # 如果需要分类器引导，创建一个空的negative embeddings
        if do_classifier_free_guidance:
            # 创建一个与embeddings相同形状的零张量作为negative embeddings
            negative_embeddings = torch.zeros_like(embeddings)
            return embeddings, negative_embeddings
        
        return embeddings, None

def set_seed(seed):
    """Set seed for reproducibility.

    Args:
        seed (int): seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return

def load_model_and_generate_images_batch(pipeline, model_checkpoint, prompts_df,
                                       gen_img_path, num_imgs=100, batch_size=10):
    """批量加载模型并为给定的提示词生成图像
    
    Args:
        pipeline: Stable Diffusion pipeline实例
        model_checkpoint: 模型检查点路径
        prompts_df: 包含提示词的DataFrame
        gen_img_path: 生成图像的保存路径
        num_imgs: 每个提示词生成的图像数量
        batch_size: 每批处理的提示词数量
    """
    model_name = model_checkpoint.split('/')[-2]+'_'+model_checkpoint.split('/')[-1]
    
    batch_prompts = prompts_df.iloc[:batch_size]
    
    # 收集这一批次的提示词和对应的保存目录
    prompts = []
    save_dirs = []
    remaining_counts = []
    
    for _, row in batch_prompts.iterrows():
        prompt = row['compound']
        model_dir = os.path.join(gen_img_path, prompt, model_name)
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        prompts.append(prompt)
        save_dirs.append(model_dir)
        
    # 开始生成图像
    start = datetime.datetime.now()
    print(f"Generating images for prompts: {prompts}")
    
    imgs = 0

    while imgs < num_imgs:
        images = pipeline(prompt=prompts).images

        for idx,(prompt, image) in enumerate(zip(prompts, images)):
            image_name = f'generated-{imgs}.png'
            image_path = os.path.join(save_dirs[idx], image_name)
            image.save(image_path)
        
        imgs += 1
    
    print(f"Batch generation time: {datetime.datetime.now()-start}")


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加各种命令行参数
    parser.add_argument('--model_checkpoint', default='',
                        help="模型检查点路径")
    parser.add_argument('--test_data_path',
                        default='',
                        help="包含所有扰动的文件路径")
    parser.add_argument('--gen_img_path', default='',
                        help="生成图像的保存路径")
    parser.add_argument('--num_imgs', default=3,
                        help="每个提示词生成的图像数量")
    parser.add_argument('--vae_path', default='',
                        help="VAE模型路径")
    parser.add_argument('--experiment', default='HUVEC-01',
                        help="实验名称")
    parser.add_argument('--model_type', default='conditional',
                        help="模型类型")
    parser.add_argument('--cluster', default='-',
                        help="集群名称")
    parser.add_argument('--model_name', default='SD',
                        help="模型名称")
    
    # 解析命令行参数
    args = parser.parse_args()
    args.num_imgs = int(args.num_imgs)

    # 创建图像保存目录
    if not os.path.exists(args.gen_img_path):
        os.makedirs(args.gen_img_path)

    # 读取扰动提示词列表
    # adata = sc.read_h5ad(args.test_data_path)
    image_data = pd.read_csv(args.test_data_path)
    image_data = image_data[image_data['split'] == 'test']

    # prompt_df = adata.obs.drop_duplicates(subset=['cmap_name']).iloc[:20]
    prompt_df = image_data.drop_duplicates(subset=['compound']).iloc[:10]

    if args.model_name == 'SD':
        # 初始化Stable Diffusion模型的各个组件
        
        # 1. 加载特征提取器
        feature_extractor = AutoImageProcessor.from_pretrained(
            os.path.join(args.model_checkpoint, 'feature_extractor'))
        print('Loaded feature_extractor')

        # 2. 加载VAE模型
        vae = AutoencoderKL.from_pretrained(
            args.vae_path, subfolder="vae")
        print('Loaded vae model')

        # 3. 加载U-Net模型（使用EMA版本）
        unet = UNet2DConditionModel.from_pretrained(
            args.model_checkpoint, subfolder="unet_ema", use_auth_token=True)
        print('Loaded EMA unet model')

        # 4. 加载噪声调度器
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.model_checkpoint, subfolder="scheduler")
        print('Loaded noise_scheduler')

        # 5. 初始化自定义的基因编码器
        custom_gene_encoder = PerturbationEncoderInference(
            args.experiment,
            args.model_type,
            args.model_name)
        print('Loaded custom_text_encoder')

        # 6. 初始化自定义的Stable Diffusion Pipeline
        pipeline = CustomStableDiffusionPipeline(
            vae=vae,
            unet=unet,
            text_encoder=custom_gene_encoder,
            feature_extractor=feature_extractor,
            scheduler=noise_scheduler)
        print('Initialized pipeline')

        # 7. 将模型移动到GPU（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.to(device)

        # 8. 开始生成图像
        load_model_and_generate_images_batch(
            pipeline,
            args.model_checkpoint,
            prompt_df,
            args.gen_img_path,
            args.num_imgs)
