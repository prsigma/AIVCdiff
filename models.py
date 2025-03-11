import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL
from typing import List
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import logger
from packaging import version
import os
from diffusers.training_utils import EMAModel, compute_snr

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
        
    def forward(self, z):
        x = self.FirstL(z)
        
        if self.HiddenL is not None:
            x = self.HiddenL(x)
            
        recon_x = self.recon_decoder(x)
        # 分离均值和方差，对均值使用ReLU激活
        dim = recon_x.size(1) // 2
        recon_x = torch.cat((self.relu(recon_x[:, :dim]), recon_x[:, dim:]), dim=1)
        return recon_x

class AIVCdiff(nn.Module):
    """AIVCdiff模型：结合图像扩散模型和RNA自编码器"""
    def __init__(
        self,
        args,
        pretrained_model_path: str,
        rna_input_dim: int,
        drug_latent_dim: int = 193,
        condition_dim: int = 768,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # 加载预训练的图像VAE
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_path,
            subfolder="vae"
        )
        self.vae.requires_grad_(False)  # 冻结VAE参数

        self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_path,
                subfolder="unet"
            )
        
        # 初始化RNA编码器和解码器
        self.rna_encoder = RNAEncoder(
            input_dim=rna_input_dim,
            hidden_layer_sizes=[1024, 768],
            latent_dim=512,
            dropout_rate=dropout_rate
        )
        
        self.rna_decoder = RNADecoder(
            output_dim=rna_input_dim,
            hidden_layer_sizes=[768, 1024],
            latent_dim=512,
            dropout_rate=dropout_rate
        )
        
        # 投影层，将不同的embedding对齐到相同的维度
        self.drug_proj = nn.Linear(drug_latent_dim, condition_dim)
        self.rna_proj = nn.Linear(512, condition_dim)
        
        # 融合层，只融合drug embedding和RNA embedding
        self.fusion_layer = nn.Sequential(
            nn.Linear(condition_dim * 2, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim)
        )

    def enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        self.unet.enable_gradient_checkpointing()
        
    def enable_xformers_memory_efficient_attention(self):
        """启用xformers内存优化"""
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems during training, please update xFormers to at least 0.0.17. "
                    "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
            
    def train(self, mode: bool = True):
        """重写train方法以确保VAE保持eval模式"""
        super().train(mode)
        self.vae.eval()
        return self
        
    def encode_vae(self, x):
        """使用VAE编码图像"""
        return self.vae.encode(x).latent_dist.sample()
        
    def decode_vae(self, z):
        """使用VAE解码latents"""
        return self.vae.decode(z).sample
        
    def encode_rna(self, rna):
        """使用RNA编码器编码RNA表达"""
        return self.rna_encoder(rna)  # 返回(z, mu, log_var)
        
    def decode_rna(self, z):
        """使用RNA解码器解码RNA潜在表示"""
        return self.rna_decoder(z)  # 返回重建的RNA表达
        
    def standardize_condition_dimension(self, condition):
        """将融合后的condition向量标准化为stable diffusion所需的维度 (B, 77, 768)
        
        Args:
            condition (tensor): 融合后的condition向量, shape为(B, 768)
            
        Returns:
            tensor: 标准化后的condition向量, shape为(B, 77, 768)
        """
        batch_size = condition.shape[0]
        # 扩展序列维度
        condition = condition.unsqueeze(1)  # [B, 1, 768]
        # 复制77次
        condition = condition.repeat(1, 77, 1)  # [B, 77, 768]
        return condition

    def forward(self, batch, noise_scheduler, noise=None):
        # 1. 处理图像部分
        latents = self.encode_vae(batch["image"])
        latents = latents * self.vae.config.scaling_factor
        
        # 添加噪声
        noise = torch.randn_like(latents) if noise is None else noise
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(
                latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        
        # 2. 获取两种embedding并对齐维度
        # 药物embedding投影
        drug_embed = self.drug_proj(batch["drug_embedding"])  # [B, condition_dim]
        
        # RNA embedding
        rna_z, rna_mu, rna_logvar = self.encode_rna(batch["pre_rna"])  # [B, 512]
        rna_embed = self.rna_proj(rna_z)  # [B, condition_dim]
        
        # 3. 融合两种embedding
        combined_embed = torch.cat([
            drug_embed,  # 药物信息
            rna_embed,   # RNA表达信息
        ], dim=1)
        
        # 融合得到条件向量
        condition = self.fusion_layer(combined_embed)  # [B, condition_dim]
        
        # 标准化维度为stable diffusion所需的格式
        condition = self.standardize_condition_dimension(condition)  # [B, 77, 768]
        
        # 4. UNet预测噪声
        noise_pred = self.unet(
            noisy_latents, 
            timesteps,
            encoder_hidden_states=condition,
            return_dict=False
        )[0]
        
        # 5. 预测RNA表达
        # 直接使用RNA编码器的隐变量进行预测
        rna_output = self.decode_rna(rna_z)
        
        # 分离均值和方差
        dim = rna_output.size(1) // 2
        rna_mu_pred = rna_output[:, :dim]
        rna_logvar_pred = rna_output[:, dim:]
        
        return {
            # 图像相关
            "noise_pred": noise_pred,
            "noise_target": noise,
            "latents": latents,
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
            
            # RNA相关
            "rna_mu_pred": rna_mu_pred,     # 预测的RNA表达均值
            "rna_logvar_pred": rna_logvar_pred,  # 预测的RNA表达方差的对数
            "rna_target": batch["post_rna"],  # 目标RNA表达
            "rna_mu": rna_mu,               # RNA编码器输出的均值
            "rna_logvar": rna_logvar,       # RNA编码器输出的方差的对数
            
            # Embedding相关
            "drug_embed": drug_embed,
            "rna_embed": rna_embed,
            "condition": condition,
            
            # 原始输入
            "pre_rna": batch["pre_rna"],
            "drug_embedding": batch["drug_embedding"]
        } 