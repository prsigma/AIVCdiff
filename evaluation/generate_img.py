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
from diffusers.image_processor import VaeImageProcessor
from diffusers.configuration_utils import ConfigMixin
from typing import Optional
import pandas as pd
import random
import numpy as np
import scanpy as sc
from tqdm import tqdm
from typing import Dict, Any, Union
from torch import nn
from typing import List, Union
from diffusers import (
    AutoencoderKL,
    DDPMScheduler, 
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DiffusionPipeline
)
from packaging import version

# 创建一个包装Linear层的类，添加dtype支持
class LinearWithDtype(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.register_buffer('_dummy', torch.zeros(1))
    
    @property
    def dtype(self):
        return self._dummy.dtype

# 创建一个包装Sequential的类，添加dtype支持
class SequentialWithDtype(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.register_buffer('_dummy', torch.zeros(1))
    
    @property
    def dtype(self):
        return self._dummy.dtype

class RNAEncoder(nn.Module):
    """RNA表达编码器，参考PRnet的实现"""
    def __init__(self, input_dim: int, hidden_layer_sizes: List[int] = [1024, 768], 
                 latent_dim: int = 512, dropout_rate: float = 0.1):
        super().__init__()
        
        # 构建编码器架构
        layer_sizes = [input_dim] + hidden_layer_sizes
        self.FC = nn.Sequential()
        
        # 构建层
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if i == 0:
                # 第一层不使用bias和批归一化
                self.FC.add_module(f"L{i}", nn.Linear(in_size, out_size, bias=False))
            else:
                self.FC.add_module(f"L{i}", nn.Linear(in_size, out_size))
                self.FC.add_module(f"N{i}", nn.BatchNorm1d(out_size))
                self.FC.add_module(f"A{i}", nn.LeakyReLU(negative_slope=0.3))
                self.FC.add_module(f"D{i}", nn.Dropout(p=dropout_rate))
        
        # 均值编码器
        self.mean_encoder = nn.Linear(hidden_layer_sizes[-1], latent_dim)
        # 方差编码器
        self.var_encoder = nn.Linear(hidden_layer_sizes[-1], latent_dim)
        
        # 初始化一个参数来确保dtype属性可用
        self.register_buffer('_dummy', torch.zeros(1))
        
    @property
    def dtype(self):
        return self._dummy.dtype
        
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        x = self.FC(x)
        mu = self.mean_encoder(x)
        log_var = self.var_encoder(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class RNADecoder(nn.Module):
    """RNA表达解码器，参考PRnet的实现"""
    def __init__(self, output_dim: int, hidden_layer_sizes: List[int] = [768, 1024], 
                 latent_dim: int = 512, dropout_rate: float = 0.1):
        super().__init__()
        
        layer_sizes = [latent_dim] + hidden_layer_sizes
        
        # 第一层解码器
        print("Decoder Architecture:")
        self.FirstL = nn.Sequential()
        print(f"\tFirst Layer in/out: {layer_sizes[0]}, {layer_sizes[1]}")
        self.FirstL.add_module("L0", nn.Linear(layer_sizes[0], layer_sizes[1], bias=False))
        self.FirstL.add_module("N0", nn.BatchNorm1d(layer_sizes[1]))
        self.FirstL.add_module("A0", nn.LeakyReLU(negative_slope=0.3))
        self.FirstL.add_module("D0", nn.Dropout(p=dropout_rate))
        
        # 隐藏层
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                print(f"\tHidden Layer {i+1} in/out: {in_size}, {out_size}")
                self.HiddenL.add_module(f"L{i+1}", nn.Linear(in_size, out_size, bias=False))
                self.HiddenL.add_module(f"N{i+1}", nn.BatchNorm1d(out_size))
                self.HiddenL.add_module(f"A{i+1}", nn.LeakyReLU(negative_slope=0.3))
                self.HiddenL.add_module(f"D{i+1}", nn.Dropout(p=dropout_rate))
        else:
            self.HiddenL = None
            
        # 输出层
        print(f"\tOutput Layer in/out: {layer_sizes[-1]}, {output_dim*2}")
        self.recon_decoder = nn.Sequential(
            nn.Linear(layer_sizes[-1], output_dim * 2)  # 输出均值和方差
        )
        self.relu = nn.ReLU()
        
        # 添加dummy参数
        self.register_buffer('_dummy', torch.zeros(1))
        
    @property
    def dtype(self):
        return self._dummy.dtype
        
    def forward(self, z):
        x = self.FirstL(z)
        
        if self.HiddenL is not None:
            x = self.HiddenL(x)
            
        recon_x = self.recon_decoder(x)
        # 分离均值和方差，对均值使用ReLU激活
        dim = recon_x.size(1) // 2
        recon_x = torch.cat((self.relu(recon_x[:, :dim]), recon_x[:, dim:]), dim=1)
        return recon_x

class CustomStableDiffusionPipeline(StableDiffusionPipeline,DiffusionPipeline):
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
        super(DiffusionPipeline).__init__()

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
        
        if scheduler is not None and getattr(scheduler.config, "steps_offset", 1) != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if scheduler is not None and getattr(scheduler.config, "clip_sample", False) is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = (
            unet is not None
            and hasattr(unet.config, "_diffusers_version")
            and version.parse(version.parse(unet.config._diffusers_version).base_version) < version.parse("0.9.0.dev0")
        )
        self._is_unet_config_sample_size_int = unet is not None and isinstance(unet.config.sample_size, int)
        is_unet_sample_size_less_64 = (
            unet is not None
            and hasattr(unet.config, "sample_size")
            and self._is_unet_config_sample_size_int
            and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- stable-diffusion-v1-5/stable-diffusion-v1-5"
                " \n- stable-diffusion-v1-5/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def eval(self):
        """将所有模块设置为评估模式"""
        for name, module in self.components.items():
            if hasattr(module, 'eval') and callable(module.eval):
                module.eval()
        return self
    
    def train(self):
        """将所有模块设置为训练模式"""
        for name, module in self.components.items():
            if hasattr(module, 'train') and callable(module.train):
                module.train()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path,**kwargs):
        """从预训练目录加载模型"""
        try:
            # 1. 加载标准组件
            vae = AutoencoderKL.from_pretrained(
                pretrained_model_path,
                subfolder="vae"
            )
            print('Loaded vae model')
            
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_path,
                subfolder="unet"
            )
            print('Loaded unet model')
            
            scheduler = DDPMScheduler.from_pretrained(
                pretrained_model_path,
                subfolder="scheduler"
            )
            print('Loaded scheduler')
            
            feature_extractor = AutoImageProcessor.from_pretrained(
                pretrained_model_path,
                subfolder="feature_extractor"
            )
            print('Loaded feature_extractor')
            
            # 2. 首先加载state_dict分析结构
            rna_encoder_state = torch.load(os.path.join(pretrained_model_path, "rna_encoder.pth"))
            rna_decoder_state = torch.load(os.path.join(pretrained_model_path, "rna_decoder.pth"))
            drug_proj_state = torch.load(os.path.join(pretrained_model_path, "drug_proj.pth"))
            rna_proj_state = torch.load(os.path.join(pretrained_model_path, "rna_proj.pth"))
            fusion_layer_state = torch.load(os.path.join(pretrained_model_path, "fusion_layer.pth"))
            
            # 分析RNA编码器结构
            fc_layers = [k for k in rna_encoder_state.keys() if k.startswith('FC.L')]
            input_dim = rna_encoder_state['FC.L0.weight'].shape[1]
            hidden_sizes = []
            for layer in fc_layers[:-1]:
                hidden_sizes.append(rna_encoder_state[layer].shape[0])
            latent_dim = rna_encoder_state['mean_encoder.weight'].shape[0]
            
            # 初始化RNA编码器
            rna_encoder = RNAEncoder(
                input_dim=input_dim,
                hidden_layer_sizes=hidden_sizes,
                latent_dim=latent_dim
            )
            rna_encoder.load_state_dict(rna_encoder_state, strict=False)
            
            # 分析RNA解码器结构
            output_dim = rna_decoder_state['recon_decoder.0.weight'].shape[0] // 2

            # 分析FirstL层
            first_layer_size = rna_decoder_state['FirstL.L0.weight'].shape[0]
            print(f"First layer size: {first_layer_size}")

            # 分析HiddenL层
            decoder_hidden_sizes = [first_layer_size]  # 首先添加FirstL的输出维度
            hidden_layers = [k for k in rna_decoder_state.keys() if k.startswith('HiddenL.L')]
            if hidden_layers:  # 如果存在HiddenL层
                for layer in sorted(hidden_layers):  # 确保按顺序处理层
                    layer_size = rna_decoder_state[layer].shape[0]
                    decoder_hidden_sizes.append(layer_size)
                    print(f"Hidden layer {layer}: size {layer_size}")

            # 初始化RNA解码器
            rna_decoder = RNADecoder(
                output_dim=output_dim,
                hidden_layer_sizes=decoder_hidden_sizes,
                latent_dim=latent_dim
            )
            rna_decoder.load_state_dict(rna_decoder_state, strict=False)
            
            # 初始化投影层
            drug_proj = LinearWithDtype(drug_proj_state['weight'].shape[1], 
                                      drug_proj_state['weight'].shape[0])
            drug_proj.load_state_dict(drug_proj_state, strict=False)
            
            rna_proj = LinearWithDtype(rna_proj_state['weight'].shape[1], 
                                     rna_proj_state['weight'].shape[0])
            rna_proj.load_state_dict(rna_proj_state, strict=False)
            
            # 初始化融合层
            fusion_layer = SequentialWithDtype(
                LinearWithDtype(fusion_layer_state['0.weight'].shape[1], 
                              fusion_layer_state['0.weight'].shape[0]),
                nn.LayerNorm(fusion_layer_state['1.weight'].shape[0]),
                nn.ReLU(),
                LinearWithDtype(fusion_layer_state['3.weight'].shape[1], 
                              fusion_layer_state['3.weight'].shape[0])
            )
            fusion_layer.load_state_dict(fusion_layer_state, strict=False)
            
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

def set_seed(seed):
    """Set seed for reproducibility.

    Args:
        seed (int): seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return

def load_model_and_generate_images_batch(pipeline, model_checkpoint, prompts_df,
                                       gen_img_path, drug_embed_df, adata_ctrl, device, num_imgs=100, batch_size=10):
    """批量加载模型并为给定的提示词生成图像
    
    Args:
        pipeline: Stable Diffusion pipeline实例
        model_checkpoint: 模型检查点路径
        prompts_df: 包含提示词的DataFrame
        gen_img_path: 生成图像的保存路径
        drug_embed_df: 药物嵌入DataFrame
        adata_ctrl: 对照组RNA数据
        num_imgs: 每个提示词生成的图像数量
        batch_size: 每批处理的提示词数量
    """
    model_name = model_checkpoint.split('/')[-2]+'_'+model_checkpoint.split('/')[-1]

    pipeline.eval()
    
    batch_prompts = prompts_df.iloc[:batch_size]
    
    # 收集这一批次的提示词和对应的保存目录
    prompts = []
    save_dirs = []
    drug_embeddings = []
    pre_rnas = []
    
    for _, row in batch_prompts.iterrows():
        cmap_name = row['cmap_name']
        
        # 获取drug embedding
        drug_embedding = torch.FloatTensor(
            drug_embed_df.loc[cmap_name].values
        ).unsqueeze(0)  # 添加batch维度
        
        # 随机选择一个ctrl组样本
        ctrl_idx = np.random.randint(0, len(adata_ctrl))
        pre_rna = torch.FloatTensor(
            adata_ctrl.X[ctrl_idx]
        ).unsqueeze(0)
        
        # 创建保存目录
        model_dir = os.path.join(gen_img_path, cmap_name, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        prompts.append(cmap_name)
        save_dirs.append(model_dir)
        drug_embeddings.append(drug_embedding)
        pre_rnas.append(pre_rna)
    
    # 将列表转换为批次张量
    drug_embeddings = torch.cat(drug_embeddings, dim=0).to(device)
    pre_rnas = torch.cat(pre_rnas, dim=0).to(device)
    
    # 开始生成图像
    start = datetime.datetime.now()
    print(f"Generating images for prompts: {prompts}")
    
    imgs = 0
    while imgs < num_imgs:
        # 调用pipeline时传入drug_embedding和pre_rna
        images = pipeline(
            drug_embedding=drug_embeddings,
            pre_rna=pre_rnas,
            device=device
        ).images

        for idx, (prompt, image) in enumerate(zip(prompts, images)):
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
    parser.add_argument('--adata_ctrl_path',
                        default='',
                        help="包含所有扰动的文件路径")
    parser.add_argument('--drug_embed_path',
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
    adata = sc.read_h5ad(args.test_data_path)

    prompt_df = adata.obs.drop_duplicates(subset=['cmap_name'])

    # 读取drug embedding数据
    drug_embed_df = pd.read_csv(args.drug_embed_path, index_col=0)
    
    # 读取对照组RNA数据
    adata_ctrl = sc.read_h5ad(args.adata_ctrl_path)

    if args.model_name == 'SD':
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        # 加载pipeline
        pipeline = CustomStableDiffusionPipeline.from_pretrained(args.model_checkpoint)
    
        print('Initialized pipeline')

        # 7. 将模型移动到GPU（如果可用）
        pipeline.to(device)

        # 8. 开始生成图像
        load_model_and_generate_images_batch(
            pipeline,
            args.model_checkpoint,
            prompt_df,
            args.gen_img_path,
            drug_embed_df,
            adata_ctrl,
            device,
            args.num_imgs
        )
