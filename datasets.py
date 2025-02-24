import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import ast
from PIL import Image
import random
import torch.nn.functional as F
from torchvision import transforms
import os
import logging

class AIVCdiffDataset(Dataset):
    """Dataset for RNA expression and cell image data"""
    
    def __init__(self, 
                 adata_path,
                 adata_ctrl_path,
                 perturbation_embedding_path,
                 transform=None):
        """
        Args:
            adata_path (str): Path to perturbed RNA expression h5ad file
            adata_ctrl_path (str): Path to control RNA expression h5ad file
            perturbation_embedding_path (str): Path to perturbation embeddings CSV file
            transform (callable, optional): Optional transform to be applied on images
        """
        # 加载数据
        self.adata = sc.read_h5ad(adata_path)
        self.adata_ctrl = sc.read_h5ad(adata_ctrl_path)
        
        # 创建cmap_name到ctrl_idx的映射
        self.cmap_to_ctrl = {}
        unique_cmaps = self.adata.obs['cmap_name'].unique()
        for cmap in unique_cmaps:
            # 为每个唯一的cmap_name随机选择一个ctrl_idx
            self.cmap_to_ctrl[cmap] = random.randint(0, len(self.adata_ctrl)-1)
        
        # 加载perturbation embeddings
        self.perturbation_embeddings = pd.read_csv(perturbation_embedding_path)
        self.perturbation_embeddings = self.perturbation_embeddings.rename(columns={'Unnamed: 0':'cmap_name'})
        self.perturbation_embeddings.set_index('cmap_name', inplace=True)
        
        # Parse image paths from adata.obs
        self.image_paths = self.adata.obs['merged_image']
        self.transform = transforms.Compose(
        [
            transforms.Resize((128, 128), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
        
        # Validate data
        assert len(self.image_paths) == len(self.adata), \
            "Number of image samples must match RNA samples"
            
    # def standardize_embedding_dimension(self, embedding):
    #     """将drug embedding标准化为stable diffusion所需的维度 (1, 77, 768)
        
    #     Args:
    #         embedding (tensor): 原始drug embedding, shape为(512,)
            
    #     Returns:
    #         tensor: 标准化后的embedding, shape为(1, 77, 768)
    #     """
    #     # 1. 首先确保输入是2D tensor
    #     if embedding.dim() == 1:
    #         embedding = embedding.unsqueeze(0)  # (512,) -> (1, 512)
            
    def _load_image(self, path):
        """加载并处理图像"""
        try:
            # 打开图像
            img = Image.open(path)
            
            # 确保图像是RGB格式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            source_image = np.array(img)
                
            # 应用转换
            if self.transform is not None:
                img = self.transform(img)
            
            return img,source_image                                     
            
        except Exception as e:
            logging.error(f"加载图像失败 {path}: {str(e)}")
            return None

    def __len__(self):
        return len(self.adata)
        
    def __getitem__(self, idx):
        # 获取当前样本的cmap_name
        cmap_name = self.adata.obs['cmap_name'].iloc[idx]
        
        # 获取该cmap_name对应的ctrl_idx
        ctrl_idx = self.cmap_to_ctrl[cmap_name]
        
        # 获取图像
        img_path = self.adata.obs['merged_image'].iloc[idx]
        image,source_image = self._load_image(img_path)
        if image is None:
            raise ValueError(f"无法加载图像: {img_path}")
        
        # 获取RNA表达数据
        rna_expr = torch.FloatTensor(self.adata.X[idx])
        ctrl_rna = torch.FloatTensor(self.adata_ctrl.X[ctrl_idx])
        
        # 获取drug embedding
        drug_embedding = torch.FloatTensor(
            self.perturbation_embeddings.loc[cmap_name].values
        )


        return {
            'image': image,
            'source_image':source_image,
            'pre_rna': ctrl_rna, 
            'post_rna': rna_expr,
            'drug_id': cmap_name,
            'drug_embedding': drug_embedding
        }

if __name__ == "__main__":
    # 设置数据路径
    adata_path = "/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/adata_train_cleaned.h5ad"
    adata_ctrl_path = "/home/pengrui/work_space_pengrui/project/RNA图像合成/1_3_rna_ctrl_data.h5ad"
    perturbation_embedding_path = "/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/molecule_embeddings_重复site.csv"
    
    # 创建数据集实例
    dataset = AIVCdiffDataset(
        adata_path=adata_path,
        adata_ctrl_path=adata_ctrl_path,
        perturbation_embedding_path=perturbation_embedding_path
    )
    
    # 测试数据集大小
    print(f"Dataset size: {len(dataset)}")
    
    # 测试获取一个样本
    sample = dataset[0]
    
    # 打印每个返回值的形状和类型
    print("\nSample contents:")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Pre-RNA shape: {sample['pre_rna'].shape}")
    print(f"Post-RNA shape: {sample['post_rna'].shape}")
    print(f"Drug ID: {sample['drug_id']}")
    print(f"Drug embedding shape: {sample['drug_embedding'].shape}")
        
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 获取一个batch的数据
    print("\nTesting DataLoader:")
    for batch in dataloader:
        print(f"Batch sizes:")
        print(f"Image batch shape: {batch['image'].shape}")
        print(f"Pre-RNA batch shape: {batch['pre_rna'].shape}")
        print(f"Post-RNA batch shape: {batch['post_rna'].shape}")
        print(f"Drug embedding batch shape: {batch['drug_embedding'].shape}")
        print(f"Drug IDs: {batch['drug_id']}")
        break  # 只测试第一个batch 