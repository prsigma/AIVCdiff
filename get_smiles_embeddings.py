import torch
import anndata
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import sys
import os
from huggingface_hub import hf_hub_download
from cloome.model import CLOOME

# # 添加CLOOME源代码路径
# CLOOME_PATH = "/home/pengrui/work_space_pengrui/project/RNA图像合成/reference_code/cloome/src"
# sys.path.append(CLOOME_PATH)

# from clip.model import CLIP, CLIPGeneral

import torch
import anndata
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import sys
import os
from huggingface_hub import hf_hub_download
from cloome.model import CLOOME
import logging
import scanpy as sc

# # 添加CLOOME源代码路径
# CLOOME_PATH = "/home/pengrui/work_space_pengrui/project/RNA图像合成/reference_code/cloome/src"
# sys.path.append(CLOOME_PATH)

from clip.model import CLIP, CLIPGeneral

def morgan_from_smiles(smiles, radius=3, nbits=1024, chiral=True):
    """
    将SMILES转换为Morgan指纹
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nbits, dtype=np.int8)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits, useChirality=chiral)
        arr = np.zeros(nbits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(nbits, dtype=np.int8)

def load_cloome_model(model_path, config_path=None):
    """加载预训练的CLOOME模型"""
    # 如果是本地文件，直接使用
    if os.path.exists(model_path):
        ckpt = model_path
    else:
        # 从Hugging Face下载
        REPO_ID = "anasanchezf/cloome"
        ckpt = hf_hub_download(REPO_ID, model_path)
    
    # 加载模型
    if config_path is None:
        config_path = os.path.join(os.path.dirname(ckpt), "/home/pengrui/work_space_pengrui/project/RNA图像合成/reference_code/cloome/src/training/model_configs/RN50.json")
    
    model = CLOOME(ckpt, config_path)
    return model

def get_smiles_embeddings(pr_df, model_path, output_path=None):
    """
    从h5ad文件中获取SMILES字符串，转换为Morgan指纹，然后使用CLOOME生成embeddings
    
    参数:
    adata_path: h5ad文件路径
    model_path: CLOOME预训练模型路径
    output_path: 输出embeddings的CSV文件路径（可选）
    
    返回:
    embeddings: numpy数组，shape为(n_compounds, embedding_dim)
    """
    mol_info = pr_df
    mol_info = mol_info.drop_duplicates(subset='canonical_smiles',keep='first')
    smiles_list = mol_info['canonical_smiles'].tolist()
    compound_names = mol_info['cmap_name'].tolist()
    
    # 生成Morgan指纹
    print("Generating Morgan fingerprints...")
    fps = [morgan_from_smiles(smiles) for smiles in smiles_list]
    fps = np.stack(fps)
    
    # 加载模型
    print("Loading CLOOME model...")
    model = load_cloome_model(model_path)
    
    # 生成embeddings
    print("Generating embeddings...")
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(fps), batch_size):
            batch_fps = fps[i:i + batch_size]
            
            # 获取embeddings
            batch_embeddings = model.encode_molecules(batch_fps)
            embeddings.append(batch_embeddings.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    
    # 保存embeddings和化合物名称到CSV（如果指定了输出路径）
    if output_path:
        print(f"Saving embeddings to {output_path}")
        # 创建列名
        feature_columns = [f'mol_feature_{i}' for i in range(embeddings.shape[1])]
        
        # 创建DataFrame
        df = pd.DataFrame(embeddings, columns=feature_columns)
        df.insert(0, 'cmap_name', compound_names)  # 在最前面插入化合物名称列
        
        df = df.set_index('cmap_name')
        
        normalized_df=(df-df.mean())/df.std()
        
        # refined_index = list()
        # for index in normalized_df.index.tolist():
        #     refined_index.append(index.replace(' ', '-').replace('/', '-').replace('(', '-').replace(')', '-').replace('\'', ''))
        # normalized_df.index = refined_index
        normalized_df.fillna(0, inplace=True)
        
        # 保存为CSV
        normalized_df.to_csv(output_path)
        print(f"Saved embeddings with shape {embeddings.shape} to {output_path}")
    
    return embeddings

if __name__ == "__main__":
    base_dir = "/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff"

    # 数据集路径
    datasets = {
        "训练集": os.path.join(base_dir, "adata_train_cleaned.h5ad"),
        "验证集": os.path.join(base_dir, "adata_valid_updated_cleaned.h5ad"),
        "测试集": os.path.join(base_dir, "adata_test_updated_cleaned.h5ad")
    }

    # 存储所有数据集的obs
    all_obs = []

    # 加载每个数据集
    for name, path in datasets.items():
        try:
            logging.info(f"加载{name}: {path}")
            adata = sc.read_h5ad(path)
            
            all_obs.append(adata.obs)
            
        except Exception as e:
            logging.error(f"加载{name}时出错: {str(e)}")
            continue

    # 合并所有obs
    logging.info("合并所有数据集")
    merged_obs = pd.concat(all_obs, axis=0)

    # 去重（基于cmp_name和canonical_smiles）
    smiles_df = merged_obs.drop_duplicates(
        subset=['cmap_name', 'canonical_smiles'],
        keep='first'
    )[['cmap_name', 'canonical_smiles']]

    smiles_df = smiles_df.drop_duplicates(subset='cmap_name',keep='first')

    # remove rows with NaN values
    smiles_df = smiles_df.dropna()
    # reset index
    smiles_df = smiles_df.reset_index(drop=True)

    # 设置路径
    MODEL_PATH = "/home/pengrui/work_space_pengrui/project/RNA图像合成/cloome-bioactivity.pt"
    OUTPUT_PATH = "/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff/molecule_embeddings_重复site.csv"  # 改为CSV输出路径
    
    # 获取embeddings
    embeddings = get_smiles_embeddings(smiles_df, MODEL_PATH, OUTPUT_PATH)
    print(f"Generated embeddings shape: {embeddings.shape}") 