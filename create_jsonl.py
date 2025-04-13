import anndata as ad
import json
import logging
from tqdm import tqdm
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# def process_compound_name(name):
#     """
#     处理化合物名称
#     """
#     return (name.lower()
#             .replace(' ', '-')
#             .replace('/', '-')
#             .replace('(', '-')
#             .replace(')', '-')
#             .replace('\'', ''))

def create_jsonl():
    """
    创建jsonl文件
    """
    # 文件路径
    base_dir = "/home/pengrui/work_space_pengrui/project/RNA图像合成/AIVCdiff"
    train_path = os.path.join(base_dir, "adata_train_只有位点1_cleaned.h5ad")
    output_path = os.path.join('/home/pengrui/work_space_pengrui/project/RNA图像合成/data_processed_只有位点1', "metadata.jsonl")
    
    # 加载训练数据
    logging.info(f"加载训练数据: {train_path}")
    try:
        adata = ad.read_h5ad(train_path)
    except Exception as e:
        logging.error(f"加载数据失败: {str(e)}")
        return
    
    logging.info(f"总样本数: {len(adata)}")
    
    # 创建jsonl文件
    count = 0
    failed = 0
    
    with open(output_path, 'w') as f:
        for idx, row in tqdm(adata.obs.iterrows(), total=len(adata.obs), desc="处理数据"):
            try:
                # 获取图片路径和化合物名称
                full_image_path = row['merged_image']
                compound_name = row['cmap_name']

                image_basename = os.path.basename(full_image_path)
                
                # 检查图片是否存在
                if not os.path.exists(full_image_path):
                    logging.warning(f"图片不存在: {full_image_path}")
                    failed += 1
                    continue
                
                # # 处理化合物名称
                # processed_name = process_compound_name(compound_name)
                
                # 创建json对象
                data = {
                    "file_name": image_basename,
                    "perturb_id": compound_name
                }
                
                # 写入jsonl文件
                f.write(json.dumps(data) + '\n')
                count += 1
                
            except Exception as e:
                logging.error(f"处理样本失败 {idx}: {str(e)}")
                failed += 1
                continue
    
    # 输出统计信息
    logging.info("\n处理完成统计:")
    logging.info(f"成功处理样本数: {count}")
    logging.info(f"失败样本数: {failed}")
    logging.info(f"输出文件: {output_path}")

if __name__ == '__main__':
    create_jsonl()