from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
import os
import torch
import datetime
import argparse
from transformers import AutoImageProcessor
import pandas as pd
import numpy as np
from tqdm import tqdm
import scanpy as sc

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
        
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker
        )
        
        # 然后手动设置自定义组件
        self.rna_encoder = rna_encoder
        self.rna_decoder = rna_decoder
        self.drug_proj = drug_proj
        self.rna_proj = rna_proj
        self.fusion_layer = fusion_layer

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

def load_model_and_generate_images_batch(pipeline, model_checkpoint, prompts_df,
                                       gen_img_path, drug_embed_df, adata_ctrl, device, num_imgs=100, batch_size=10):
    
    model_name = model_checkpoint.split('/')[-2]+'_'+model_checkpoint.split('/')[-1]

    # pipeline.eval()
    
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

def main():
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

    # 初始化模型组件
    feature_extractor = AutoImageProcessor.from_pretrained(
        os.path.join(args.model_checkpoint, 'feature_extractor'))
    
    vae = AutoencoderKL.from_pretrained(
        args.model_checkpoint, subfolder="vae")
    
    unet = UNet2DConditionModel.from_pretrained(
        args.model_checkpoint, subfolder="unet_ema")
    
    scheduler = DDPMScheduler.from_pretrained(
        args.model_checkpoint, subfolder="scheduler")

    # 加载自定义组件
    rna_encoder = torch.load(os.path.join(args.model_checkpoint, "rna_encoder.pt"))
    rna_decoder = torch.load(os.path.join(args.model_checkpoint, "rna_decoder.pt"))
    
    projection_states = torch.load(os.path.join(args.model_checkpoint, "projection_layers.pt"))
    drug_proj = projection_states['drug_proj']
    rna_proj = projection_states['rna_proj']
    fusion_layer = projection_states['fusion_layer']

    # 初始化pipeline
    pipeline = CustomStableDiffusionPipeline(
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

    # 移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


if __name__ == '__main__':
    main()