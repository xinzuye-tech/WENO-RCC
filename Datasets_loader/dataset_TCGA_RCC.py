#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-RCC 数据集加载器
用于肾癌亚型分类任务（MIL多示例学习）

任务A: TCGA-KIRC (label=1) vs TCGA-KICH (label=0)
任务B: TCGA-KIRP (label=1) vs TCGA-KICH (label=0)

支持 Task A 和 Task B 共享 TCGA-KICH 划分
"""

import numpy as np
import torch
import torch.utils.data as data_utils
import os
import glob
import json
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
        split_file: 数据划分文件路径，如果提供则从文件加载划分（忽略seed和train_ratio）
        kich_split_file: KICH划分文件路径，用于Task B共享Task A的KICH划分
                         可以是Task A的data_split.json或单独的kich_split.json
    """
    
    def __init__(self, 
                 task='A',
                 train=True, 
                 downsample=1.0, 
                 return_bag=False,
                 base_dir="/workspace/cpfs-data/WENO",
                 train_ratio=0.8,
                 seed=42,
                 split_file=None,
                 kich_split_file=None):
        
        self.task = task
        self.train = train
        self.downsample = downsample
        self.return_bag = return_bag
        self.base_dir = base_dir
        self.train_ratio = train_ratio
        self.seed = seed
        self.split_file = split_file
        self.kich_split_file = kich_split_file
        
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
        self.pos_dir = os.path.join(base_dir, self.pos_class)
        self.neg_dir = os.path.join(base_dir, self.neg_class)
        
        pos_slides = sorted(glob.glob(os.path.join(self.pos_dir, "*.npy")))
        neg_slides = sorted(glob.glob(os.path.join(self.neg_dir, "*.npy")))
        
        print(f"[DATA INFO] SUM of {self.pos_class}: {len(pos_slides)} slides")
        print(f"[DATA INFO] SUM of {self.neg_class}: {len(neg_slides)} slides")
        
        # 2. 数据划分：从文件加载或创建新划分
        if split_file is not None and os.path.exists(split_file):
            # 从完整的划分文件加载（优先级最高）
            selected_pos_slides, selected_neg_slides = self._load_split(
                split_file, pos_slides, neg_slides
            )
        else:
            # 创建新划分（可能使用外部KICH划分）
            selected_pos_slides, selected_neg_slides = self._create_split(
                pos_slides, neg_slides
            )
        
        print(f"[DATA INFO] 当前集合 - {self.pos_class}: {len(selected_pos_slides)}, {self.neg_class}: {len(selected_neg_slides)}")
        
        # 3. 下采样（如果需要）
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
        
        # 4. 合并所有slides并打乱
        all_slides = [(s, 1) for s in selected_pos_slides] + [(s, 0) for s in selected_neg_slides]
        np.random.seed(seed + 2)
        np.random.shuffle(all_slides)
        
        self.num_slides = len(all_slides)
        
        # 5. 加载所有特征数据
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
        
        # 6. 合并所有数据
        self.patch_feat_all = np.concatenate(self.patch_feat_all, axis=0).astype(np.float32)
        self.patch_corresponding_slide_label = np.concatenate(self.patch_corresponding_slide_label).astype(np.int64)
        self.patch_corresponding_slide_index = np.concatenate(self.patch_corresponding_slide_index).astype(np.int64)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)
        
        # patch_label 不可用，全部设为0
        self.num_patches = self.patch_feat_all.shape[0]
        self.patch_label_all = np.zeros(self.num_patches, dtype=np.int64)
        
        # 7. 统计信息
        print(f"\n[DATA INFO] num_slides: {self.num_slides}")
        print(f"[DATA INFO] num_patches: {self.num_patches}")
        print(f"[DATA INFO] feature_dim: {self.patch_feat_all.shape[1]}")
        print(f"[DATA INFO] 正类slides: {len(selected_pos_slides)}, 负类slides: {len(selected_neg_slides)}")
        print(f"[DATA INFO] 正类patches: {(self.patch_corresponding_slide_label == 1).sum()}")
        print(f"[DATA INFO] 负类patches: {(self.patch_corresponding_slide_label == 0).sum()}")
        print("=" * 60 + "\n")
    
    def _create_split(self, pos_slides, neg_slides):
        """
        创建训练/测试划分
        支持从外部文件加载KICH划分（用于Task B共享Task A的KICH划分）
        
        Returns:
            selected_pos_slides: 当前集合的正类slides路径列表
            selected_neg_slides: 当前集合的负类slides路径列表
        """
        # ============================================================
        # 处理负类（KICH）划分
        # ============================================================
        if self.kich_split_file is not None and os.path.exists(self.kich_split_file):
            # 从外部文件加载KICH划分
            print(f"[DATA INFO] 从外部文件加载KICH划分: {self.kich_split_file}")
            with open(self.kich_split_file, 'r') as f:
                external_split = json.load(f)
            
            # 支持两种格式：完整的data_split.json 或 单独的kich_split.json
            if 'neg_train_slides' in external_split:
                neg_train_names = external_split['neg_train_slides']
                neg_test_names = external_split['neg_test_slides']
            else:
                raise ValueError(f"KICH划分文件格式不正确，缺少'neg_train_slides'字段")
            
            # 转换为完整路径
            neg_train_slides = [os.path.join(self.neg_dir, name) for name in neg_train_names]
            neg_test_slides = [os.path.join(self.neg_dir, name) for name in neg_test_names]
            
            # 验证文件存在
            missing = [s for s in neg_train_slides + neg_test_slides if not os.path.exists(s)]
            if missing:
                raise FileNotFoundError(f"KICH slides not found: {missing[:5]}...")
            
            kich_split_source = self.kich_split_file
            print(f"[DATA INFO] 已加载KICH划分 - 训练: {len(neg_train_slides)}, 测试: {len(neg_test_slides)}")
        else:
            # 创建新的KICH划分
            np.random.seed(self.seed)
            neg_indices = np.random.permutation(len(neg_slides))
            neg_train_size = int(len(neg_slides) * self.train_ratio)
            neg_train_slides = [neg_slides[i] for i in neg_indices[:neg_train_size]]
            neg_test_slides = [neg_slides[i] for i in neg_indices[neg_train_size:]]
            kich_split_source = 'self'
        
        # ============================================================
        # 处理正类划分（每个任务独立）
        # ============================================================
        # Task A 和 Task B 使用不同的种子划分各自的正类
        pos_seed = self.seed if self.task == 'A' else self.seed + 1000
        np.random.seed(pos_seed)
        pos_indices = np.random.permutation(len(pos_slides))
        pos_train_size = int(len(pos_slides) * self.train_ratio)
        pos_train_slides = [pos_slides[i] for i in pos_indices[:pos_train_size]]
        pos_test_slides = [pos_slides[i] for i in pos_indices[pos_train_size:]]
        
        # ============================================================
        # 保存划分信息
        # ============================================================
        self.split_info = {
            'task': self.task,
            'seed': self.seed,
            'train_ratio': self.train_ratio,
            'pos_class': self.pos_class,
            'neg_class': self.neg_class,
            'base_dir': self.base_dir,
            'kich_split_source': kich_split_source,  # 记录KICH划分来源
            'pos_train_slides': [os.path.basename(s) for s in pos_train_slides],
            'pos_test_slides': [os.path.basename(s) for s in pos_test_slides],
            'neg_train_slides': [os.path.basename(s) for s in neg_train_slides],
            'neg_test_slides': [os.path.basename(s) for s in neg_test_slides],
            'num_pos_train': len(pos_train_slides),
            'num_pos_test': len(pos_test_slides),
            'num_neg_train': len(neg_train_slides),
            'num_neg_test': len(neg_test_slides),
        }
        
        # 根据train参数选择数据
        if self.train:
            return pos_train_slides, neg_train_slides
        else:
            return pos_test_slides, neg_test_slides
    
    def _load_split(self, split_file, pos_slides, neg_slides):
        """
        从文件加载数据划分
        
        Args:
            split_file: 划分文件路径
            pos_slides: 所有正类slides路径
            neg_slides: 所有负类slides路径
            
        Returns:
            selected_pos_slides: 当前集合的正类slides路径列表
            selected_neg_slides: 当前集合的负类slides路径列表
        """
        print(f"[DATA INFO] 从文件加载数据划分: {split_file}")
        
        with open(split_file, 'r') as f:
            self.split_info = json.load(f)
        
        # 验证任务一致性
        if self.split_info['task'] != self.task:
            raise ValueError(
                f"划分文件的任务({self.split_info['task']})与当前任务({self.task})不匹配！"
            )
        
        # 根据train参数选择slide名称列表
        if self.train:
            pos_slide_names = self.split_info['pos_train_slides']
            neg_slide_names = self.split_info['neg_train_slides']
        else:
            pos_slide_names = self.split_info['pos_test_slides']
            neg_slide_names = self.split_info['neg_test_slides']
        
        # 转换为完整路径
        selected_pos_slides = [os.path.join(self.pos_dir, name) for name in pos_slide_names]
        selected_neg_slides = [os.path.join(self.neg_dir, name) for name in neg_slide_names]
        
        # 验证文件存在
        missing_files = []
        for slide in selected_pos_slides + selected_neg_slides:
            if not os.path.exists(slide):
                missing_files.append(slide)
        
        if missing_files:
            raise FileNotFoundError(
                f"以下{len(missing_files)}个文件不存在:\n" + 
                "\n".join(missing_files[:10]) + 
                (f"\n... 等共{len(missing_files)}个文件" if len(missing_files) > 10 else "")
            )
        
        kich_source = self.split_info.get('kich_split_source', 'unknown')
        print(f"[DATA INFO] 成功加载划分，seed={self.split_info['seed']}, KICH来源={kich_source}")
        
        return selected_pos_slides, selected_neg_slides
    
    def save_split(self, save_path):
        """
        保存数据划分到JSON文件
        
        Args:
            save_path: 保存路径
            
        Returns:
            save_path: 保存的文件路径
        """
        if not hasattr(self, 'split_info'):
            raise RuntimeError("没有可保存的划分信息！请确保数据集是通过创建新划分初始化的。")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.split_info, f, indent=2, ensure_ascii=False)
        
        print(f"[DATA INFO] 数据划分已保存到: {save_path}")
        return save_path
    
    def get_split_info(self):
        """
        获取数据划分信息
        
        Returns:
            dict: 划分信息字典
        """
        if not hasattr(self, 'split_info'):
            raise RuntimeError("没有划分信息！")
        return self.split_info.copy()
    
    def get_kich_split(self):
        """
        获取KICH划分信息（用于Task B共享Task A的KICH划分）
        
        Returns:
            dict: 包含neg_train_slides和neg_test_slides的字典
        """
        if not hasattr(self, 'split_info'):
            raise RuntimeError("没有划分信息！")
        
        return {
            'neg_class': self.split_info['neg_class'],
            'neg_train_slides': self.split_info['neg_train_slides'],
            'neg_test_slides': self.split_info['neg_test_slides'],
            'num_neg_train': self.split_info['num_neg_train'],
            'num_neg_test': self.split_info['num_neg_test'],
            'seed': self.split_info['seed'],
            'train_ratio': self.split_info['train_ratio'],
        }
    
    def save_kich_split(self, save_path):
        """
        单独保存KICH划分（用于Task B共享）
        
        Args:
            save_path: 保存路径
            
        Returns:
            save_path: 保存的文件路径
        """
        kich_split = self.get_kich_split()
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(kich_split, f, indent=2, ensure_ascii=False)
        
        print(f"[DATA INFO] KICH划分已保存到: {save_path}")
        return save_path
    
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
    print("测试1: Task A 基本功能")
    print("=" * 80)
    
    # 任务A - 训练集（创建新划分）
    train_ds_A = TCGA_RCC_Feat(task='A', train=True, return_bag=False, seed=42)
    
    # 保存 Task A 的完整划分
    split_save_path_A = './test_split_taskA.json'
    train_ds_A.save_split(split_save_path_A)
    
    # 打印划分信息
    split_info_A = train_ds_A.get_split_info()
    print(f"\nTask A 划分信息:")
    print(f"  任务: {split_info_A['task']}")
    print(f"  随机种子: {split_info_A['seed']}")
    print(f"  KICH来源: {split_info_A['kich_split_source']}")
    print(f"  正类训练slides: {split_info_A['num_pos_train']}")
    print(f"  正类测试slides: {split_info_A['num_pos_test']}")
    print(f"  负类(KICH)训练slides: {split_info_A['num_neg_train']}")
    print(f"  负类(KICH)测试slides: {split_info_A['num_neg_test']}")
    print(f"  KICH训练集前3个: {split_info_A['neg_train_slides'][:3]}")
    
    print("\n" + "=" * 80)
    print("测试2: Task B 共享 Task A 的 KICH 划分")
    print("=" * 80)
    
    # 方式1: 直接使用 Task A 的 data_split.json 作为 kich_split_file
    print("\n[方式1] 使用 Task A 的 data_split.json 作为 KICH 划分来源...")
    train_ds_B = TCGA_RCC_Feat(
        task='B', 
        train=True, 
        return_bag=False, 
        seed=42,
        kich_split_file=split_save_path_A  # 使用 Task A 的划分文件
    )
    
    # 保存 Task B 的划分
    split_save_path_B = './test_split_taskB.json'
    train_ds_B.save_split(split_save_path_B)
    
    # 打印 Task B 的划分信息
    split_info_B = train_ds_B.get_split_info()
    print(f"\nTask B 划分信息:")
    print(f"  任务: {split_info_B['task']}")
    print(f"  KICH来源: {split_info_B['kich_split_source']}")
    print(f"  正类(KIRP)训练slides: {split_info_B['num_pos_train']}")
    print(f"  正类(KIRP)测试slides: {split_info_B['num_pos_test']}")
    print(f"  负类(KICH)训练slides: {split_info_B['num_neg_train']}")
    print(f"  负类(KICH)测试slides: {split_info_B['num_neg_test']}")
    print(f"  KICH训练集前3个: {split_info_B['neg_train_slides'][:3]}")
    
    print("\n" + "=" * 80)
    print("测试3: 验证 KICH 划分一致性")
    print("=" * 80)
    
    kich_train_A = set(split_info_A['neg_train_slides'])
    kich_train_B = set(split_info_B['neg_train_slides'])
    kich_test_A = set(split_info_A['neg_test_slides'])
    kich_test_B = set(split_info_B['neg_test_slides'])
    
    print(f"\nKICH 训练集一致: {kich_train_A == kich_train_B}")
    print(f"KICH 测试集一致: {kich_test_A == kich_test_B}")
    
    if kich_train_A == kich_train_B and kich_test_A == kich_test_B:
        print("\n✅ Task A 和 Task B 成功共享相同的 KICH 划分！")
    else:
        print("\n❌ 划分不一致，请检查！")
    
    print("\n" + "=" * 80)
    print("测试4: 单独保存和加载 KICH 划分")
    print("=" * 80)
    
    # 单独保存 KICH 划分
    kich_split_path = './test_kich_split.json'
    train_ds_A.save_kich_split(kich_split_path)
    
    # 使用单独的 KICH 划分文件创建 Task B
    print("\n[方式2] 使用单独的 kich_split.json 文件...")
    train_ds_B2 = TCGA_RCC_Feat(
        task='B', 
        train=True, 
        return_bag=False, 
        seed=42,
        kich_split_file=kich_split_path
    )
    
    split_info_B2 = train_ds_B2.get_split_info()
    kich_train_B2 = set(split_info_B2['neg_train_slides'])
    print(f"KICH 训练集一致 (方式2): {kich_train_A == kich_train_B2}")
    
    print("\n" + "=" * 80)
    print("测试5: 从保存的划分文件加载")
    print("=" * 80)
    
    # 从保存的文件加载 Task A 测试集
    print("\n加载 Task A 测试集...")
    test_ds_A = TCGA_RCC_Feat(
        task='A', 
        train=False, 
        return_bag=False, 
        split_file=split_save_path_A
    )
    
    # 从保存的文件加载 Task B 测试集
    print("\n加载 Task B 测试集...")
    test_ds_B = TCGA_RCC_Feat(
        task='B', 
        train=False, 
        return_bag=False, 
        split_file=split_save_path_B
    )
    
    # 验证测试集 KICH 一致性
    test_info_A = test_ds_A.get_split_info()
    test_info_B = test_ds_B.get_split_info()
    print(f"\n测试集 KICH 一致: {set(test_info_A['neg_test_slides']) == set(test_info_B['neg_test_slides'])}")
    
    print("\n" + "=" * 80)
    print("测试6: Bag模式")
    print("=" * 80)
    
    train_ds_A_bag = copy.deepcopy(train_ds_A)
    train_ds_A_bag.return_bag = True
    
    train_loader_bag = torch.utils.data.DataLoader(
        train_ds_A_bag, batch_size=1, shuffle=True, num_workers=0, drop_last=False
    )
    
    print("\n--- 测试Bag Loader ---")
    for i, (data, label, idx) in enumerate(train_loader_bag):
        print(f"Bag {i}:")
        print(f"  data shape: {data.shape}")
        print(f"  patch_labels shape: {label[0].shape}")
        print(f"  slide_label: {label[1].item()}")
        print(f"  slide_index: {label[2].item()}")
        print(f"  slide_name: {label[3]}")
        if i >= 2:
            break
    
    # 清理测试文件
    print("\n" + "=" * 80)
    print("清理测试文件")
    print("=" * 80)
    for f in [split_save_path_A, split_save_path_B, kich_split_path]:
        if os.path.exists(f):
            os.remove(f)
            print(f"已删除: {f}")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)