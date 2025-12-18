#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-RCC 数据集加载器
用于肾癌亚型分类任务（MIL多示例学习）

任务A: TCGA-KIRC (label=1) vs TCGA-KICH (label=0)
任务B: TCGA-KIRP (label=1) vs TCGA-KICH (label=0)
"""

import numpy as np
import torch
import torch.utils.data as data_utils
import os
import glob
from tqdm import tqdm


class TCGA_RCC_Feat(torch.utils.data.Dataset):
    """
    TCGA-RCC 预提取特征数据集
    
    Args:
        task: 'A' 或 'B'
              'A': TCGA-KIRC (label=1) vs TCGA-KICH (label=0)
              'B': TCGA-KIRP (label=1) vs TCGA-KICH (label=0)
        train: True为训练集，False为测试集
        downsample: slide级别下采样比例，默认1.0（使用全部数据）
        return_bag: False返回单个patch，True返回整个bag（slide的所有patch）
        base_dir: 数据根目录
        train_ratio: 训练集比例，默认0.8
        seed: 随机种子，保证划分可复现
    """
    
    def __init__(self, 
                 task='A',
                 train=True, 
                 downsample=1.0, 
                 return_bag=False,
                 base_dir="/workspace/cpfs-data/WENO",
                 train_ratio=0.8,
                 seed=42):
        
        self.task = task
        self.train = train
        self.downsample = downsample
        self.return_bag = return_bag
        self.base_dir = base_dir
        self.train_ratio = train_ratio
        self.seed = seed
        
        # 根据任务定义正类和负类
        if task == 'A':
            self.pos_class = 'TCGA-KIRC'  # label=1
            self.neg_class = 'TCGA-KICH'  # label=0
        elif task == 'B':
            self.pos_class = 'TCGA-KIRP'  # label=1
            self.neg_class = 'TCGA-KICH'  # label=0
        else:
            raise ValueError(f"task必须是'A'或'B'，当前值: {task}")
        
        print("=" * 60)
        print(f"[TCGA-RCC Dataset] 任务{task}: {self.pos_class}(1) vs {self.neg_class}(0)")
        print(f"[TCGA-RCC Dataset] 模式: {'训练集' if train else '测试集'}")
        print("=" * 60)
        
        # 1. 获取所有slide文件路径
        pos_dir = os.path.join(base_dir, self.pos_class)
        neg_dir = os.path.join(base_dir, self.neg_class)
        
        pos_slides = sorted(glob.glob(os.path.join(pos_dir, "*.npy")))
        neg_slides = sorted(glob.glob(os.path.join(neg_dir, "*.npy")))
        
        print(f"[DATA INFO] {self.pos_class}: {len(pos_slides)} slides")
        print(f"[DATA INFO] {self.neg_class}: {len(neg_slides)} slides")
        
        # 2. 固定随机种子，划分训练/测试集
        np.random.seed(seed)
        
        # 对正类划分
        pos_indices = np.random.permutation(len(pos_slides))
        pos_train_size = int(len(pos_slides) * train_ratio)
        pos_train_slides = [pos_slides[i] for i in pos_indices[:pos_train_size]]
        pos_test_slides = [pos_slides[i] for i in pos_indices[pos_train_size:]]
        
        # 对负类划分
        neg_indices = np.random.permutation(len(neg_slides))
        neg_train_size = int(len(neg_slides) * train_ratio)
        neg_train_slides = [neg_slides[i] for i in neg_indices[:neg_train_size]]
        neg_test_slides = [neg_slides[i] for i in neg_indices[neg_train_size:]]
        
        # 3. 根据train参数选择数据
        if train:
            selected_pos_slides = pos_train_slides
            selected_neg_slides = neg_train_slides
        else:
            selected_pos_slides = pos_test_slides
            selected_neg_slides = neg_test_slides
        
        print(f"[DATA INFO] 当前集合 - {self.pos_class}: {len(selected_pos_slides)}, {self.neg_class}: {len(selected_neg_slides)}")
        
        # 4. 下采样（如果需要）
        if downsample < 1.0:
            np.random.seed(seed + 1)
            selected_pos_slides = list(np.random.choice(
                selected_pos_slides, 
                size=max(1, int(len(selected_pos_slides) * downsample)), 
                replace=False
            ))
            selected_neg_slides = list(np.random.choice(
                selected_neg_slides, 
                size=max(1, int(len(selected_neg_slides) * downsample)), 
                replace=False
            ))
            print(f"[DATA INFO] 下采样后 - {self.pos_class}: {len(selected_pos_slides)}, {self.neg_class}: {len(selected_neg_slides)}")
        
        # 5. 合并所有slides并打乱
        all_slides = [(s, 1) for s in selected_pos_slides] + [(s, 0) for s in selected_neg_slides]
        np.random.seed(seed + 2)
        np.random.shuffle(all_slides)
        
        self.num_slides = len(all_slides)
        
        # 6. 加载所有特征数据
        self.patch_feat_all = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        
        print(f"\n加载特征数据...")
        for slide_idx, (slide_path, slide_label) in enumerate(tqdm(all_slides, desc='Loading data')):
            # 加载.npy文件
            data = np.load(slide_path, allow_pickle=True).item()
            feats = data['feature']  # torch.Tensor, shape=[N_patches, 512]
            
            # 转换为numpy（如果是tensor）
            if isinstance(feats, torch.Tensor):
                feats = feats.numpy()
            
            num_patches = feats.shape[0]
            slide_name = os.path.basename(slide_path)  # 文件名作为slide_name
            
            self.patch_feat_all.append(feats)
            self.patch_corresponding_slide_label.append(np.ones(num_patches, dtype=np.int64) * slide_label)
            self.patch_corresponding_slide_index.append(np.ones(num_patches, dtype=np.int64) * slide_idx)
            # 使用普通str而不是np.str_
            self.patch_corresponding_slide_name.extend([str(slide_name)] * num_patches)
        
        # 7. 合并所有数据
        self.patch_feat_all = np.concatenate(self.patch_feat_all, axis=0).astype(np.float32)
        self.patch_corresponding_slide_label = np.concatenate(self.patch_corresponding_slide_label).astype(np.int64)
        self.patch_corresponding_slide_index = np.concatenate(self.patch_corresponding_slide_index).astype(np.int64)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)
        
        # patch_label 不可用，全部设为0
        self.num_patches = self.patch_feat_all.shape[0]
        self.patch_label_all = np.zeros(self.num_patches, dtype=np.int64)
        
        # 8. 统计信息
        print(f"\n[DATA INFO] num_slides: {self.num_slides}")
        print(f"[DATA INFO] num_patches: {self.num_patches}")
        print(f"[DATA INFO] feature_dim: {self.patch_feat_all.shape[1]}")
        print(f"[DATA INFO] 正类slides: {len(selected_pos_slides)}, 负类slides: {len(selected_neg_slides)}")
        print(f"[DATA INFO] 正类patches: {(self.patch_corresponding_slide_label == 1).sum()}")
        print(f"[DATA INFO] 负类patches: {(self.patch_corresponding_slide_label == 0).sum()}")
        print("=" * 60 + "\n")
    
    def __getitem__(self, index):
        if self.return_bag:
            # 返回整个bag（slide的所有patch）
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index == index)[0]
            
            bag = self.patch_feat_all[idx_patch_from_slide_i, :]
            patch_labels = self.patch_label_all[idx_patch_from_slide_i]  # 全为0
            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i][0]
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = str(self.patch_corresponding_slide_name[idx_patch_from_slide_i][0])
            
            # 数据完整性检查
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != \
               self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise ValueError("同一个bag内slide_label不一致！")
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != \
               self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise ValueError("同一个bag内slide_index不一致！")
            
            # 转换为tensor
            bag = torch.from_numpy(bag)
            patch_labels = torch.from_numpy(patch_labels)
            slide_label = torch.tensor(slide_label)
            slide_index = torch.tensor(slide_index)
            
            return bag, [patch_labels, slide_label, slide_index, slide_name], index
        
        else:
            # 返回单个patch
            patch_feat = self.patch_feat_all[index]
            patch_label = self.patch_label_all[index]  # 全为0
            slide_label = self.patch_corresponding_slide_label[index]
            slide_index = self.patch_corresponding_slide_index[index]
            slide_name = str(self.patch_corresponding_slide_name[index])
            
            # 转换为tensor
            patch_feat = torch.from_numpy(patch_feat)
            patch_label = torch.tensor(patch_label)
            slide_label = torch.tensor(slide_label)
            slide_index = torch.tensor(slide_index)
            
            return patch_feat, [patch_label, slide_label, slide_index, slide_name], index
    
    def __len__(self):
        if self.return_bag:
            return self.num_slides
        else:
            return self.num_patches


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    import copy
    
    print("\n" + "=" * 80)
    print("测试任务A: TCGA-KIRC vs TCGA-KICH")
    print("=" * 80)
    
    # 任务A - 训练集
    train_ds_A = TCGA_RCC_Feat(task='A', train=True, return_bag=False)
    train_ds_A_bag = copy.deepcopy(train_ds_A)
    train_ds_A_bag.return_bag = True
    
    # 任务A - 测试集
    test_ds_A = TCGA_RCC_Feat(task='A', train=False, return_bag=False)
    test_ds_A_bag = TCGA_RCC_Feat(task='A', train=False, return_bag=True)
    
    # 测试DataLoader
    train_loader_instance = torch.utils.data.DataLoader(
        train_ds_A, batch_size=512, shuffle=True, num_workers=0, drop_last=False
    )
    train_loader_bag = torch.utils.data.DataLoader(
        train_ds_A_bag, batch_size=1, shuffle=True, num_workers=0, drop_last=False
    )
    
    print("\n--- 测试Instance Loader ---")
    for i, (data, label, idx) in enumerate(train_loader_instance):
        print(f"Batch {i}:")
        print(f"  data shape: {data.shape}")
        print(f"  patch_label shape: {label[0].shape}")
        print(f"  slide_label shape: {label[1].shape}")
        print(f"  slide_index shape: {label[2].shape}")
        print(f"  slide_label unique: {label[1].unique().tolist()}")
        if i >= 1:
            break
    
    print("\n--- 测试Bag Loader ---")
    for i, (data, label, idx) in enumerate(train_loader_bag):
        print(f"Bag {i}:")
        print(f"  data shape: {data.shape}")  # [1, N_patches, 512]
        print(f"  patch_labels shape: {label[0].shape}")
        print(f"  slide_label: {label[1].item()}")
        print(f"  slide_index: {label[2].item()}")
        print(f"  slide_name: {label[3]}")  # 现在应该是干净的字符串
        if i >= 2:
            break
    
    print("\n" + "=" * 80)
    print("测试任务B: TCGA-KIRP vs TCGA-KICH")
    print("=" * 80)
    
    # 任务B
    train_ds_B = TCGA_RCC_Feat(task='B', train=True, return_bag=False)
    test_ds_B = TCGA_RCC_Feat(task='B', train=False, return_bag=False)
    
    print("\n所有测试通过！")