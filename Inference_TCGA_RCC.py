#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-RCC 三分类联合推理脚本 (Student 推理版)

使用 Task A (KIRC vs KICH) 和 Task B (KIRP vs KICH) 的 Student 模型
对测试集进行三分类推理：KIRC / KIRP / KICH

原论文推理方式：
"In inference, we use the instance classifier in the student branch to make 
predictions for all instances in a bag, and then use a simple max-pooling 
to aggregate the predictions of instances to accomplish bag prediction."


# ================================================================
# Student 推理（原论文方式）- 使用 best_student checkpoint
# ================================================================
python Inference_TCGA_RCC.py \
    --checkpoint_A ./checkpoints/TaskA/xxx/checkpoint_best_student.pth \
    --checkpoint_B ./checkpoints/TaskB/yyy/checkpoint_best_student.pth \
    --split_file_A ./checkpoints/TaskA/xxx/data_split.json \
    --split_file_B ./checkpoints/TaskB/yyy/data_split.json \
    --data_dir /workspace/cpfs-data/WENO \
    --output_dir ./inference_results \
    --mode student

"""

import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_recall_fscore_support, cohen_kappa_score,
    roc_auc_score, roc_curve
)
import pandas as pd
from datetime import datetime

from models.alexnet import camelyon_feat_projecter, teacher_DSMIL_head, student_head


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型，用于JSON序列化"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class ThreeClassInferenceStudent:
    """三分类联合推理器 - 使用 Student 推理（原论文方式）"""
    
    def __init__(self,
                 checkpoint_A,
                 checkpoint_B,
                 split_file_A,
                 split_file_B,
                 data_dir,
                 output_dir,
                 threshold=0.5,
                 device='cuda',
                 inference_mode='student'):  # 'student', 'teacher', 'both'
        
        self.checkpoint_A = checkpoint_A
        self.checkpoint_B = checkpoint_B
        self.split_file_A = split_file_A
        self.split_file_B = split_file_B
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.device = device
        self.inference_mode = inference_mode
        
        # 类别定义 (按字母顺序)
        self.class_names = ['KICH', 'KIRC', 'KIRP']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        os.makedirs(output_dir, exist_ok=True)
        
        self._load_models()
        self._load_data()
    
    def _load_models(self):
        """加载模型"""
        print("=" * 70)
        print("加载模型")
        print("=" * 70)
        
        # ==================== Task A: KIRC vs KICH ====================
        print(f"\n[Task A] KIRC vs KICH")
        print(f"  Checkpoint: {self.checkpoint_A}")
        
        # Encoder (共享)
        self.encoder_A = camelyon_feat_projecter(input_dim=512, output_dim=512).to(self.device)
        
        # Teacher Head
        self.teacher_A = teacher_DSMIL_head(input_feat_dim=512).to(self.device)
        
        # Student Head
        self.student_A = student_head(input_feat_dim=512).to(self.device)
        
        # 加载权重
        ckpt_A = torch.load(self.checkpoint_A, map_location=self.device, weights_only=False)
        self.encoder_A.load_state_dict(ckpt_A['model_encoder_state_dict'])
        self.teacher_A.load_state_dict(ckpt_A['model_teacherHead_state_dict'])
        self.student_A.load_state_dict(ckpt_A['model_studentHead_state_dict'])
        
        # 设置为评估模式
        self.encoder_A.eval()
        self.teacher_A.eval()
        self.student_A.eval()
        
        # 记录训练时的 AUC
        self.train_teacher_auc_A = ckpt_A.get('best_teacher_auc', 'N/A')
        self.train_student_auc_A = ckpt_A.get('best_student_auc', 'N/A')
        print(f"  训练时 Teacher AUC: {self.train_teacher_auc_A}")
        print(f"  训练时 Student AUC: {self.train_student_auc_A}")
        
        # ==================== Task B: KIRP vs KICH ====================
        print(f"\n[Task B] KIRP vs KICH")
        print(f"  Checkpoint: {self.checkpoint_B}")
        
        # Encoder (共享)
        self.encoder_B = camelyon_feat_projecter(input_dim=512, output_dim=512).to(self.device)
        
        # Teacher Head
        self.teacher_B = teacher_DSMIL_head(input_feat_dim=512).to(self.device)
        
        # Student Head
        self.student_B = student_head(input_feat_dim=512).to(self.device)
        
        # 加载权重
        ckpt_B = torch.load(self.checkpoint_B, map_location=self.device, weights_only=False)
        self.encoder_B.load_state_dict(ckpt_B['model_encoder_state_dict'])
        self.teacher_B.load_state_dict(ckpt_B['model_teacherHead_state_dict'])
        self.student_B.load_state_dict(ckpt_B['model_studentHead_state_dict'])
        
        # 设置为评估模式
        self.encoder_B.eval()
        self.teacher_B.eval()
        self.student_B.eval()
        
        # 记录训练时的 AUC
        self.train_teacher_auc_B = ckpt_B.get('best_teacher_auc', 'N/A')
        self.train_student_auc_B = ckpt_B.get('best_student_auc', 'N/A')
        print(f"  训练时 Teacher AUC: {self.train_teacher_auc_B}")
        print(f"  训练时 Student AUC: {self.train_student_auc_B}")
        
        print(f"\n推理模式: {self.inference_mode}")
    
    def _load_data(self):
        """加载测试数据"""
        print("\n" + "=" * 70)
        print("加载数据")
        print("=" * 70)
        
        with open(self.split_file_A, 'r') as f:
            split_A = json.load(f)
        with open(self.split_file_B, 'r') as f:
            split_B = json.load(f)
        
        # 验证 KICH 划分一致性
        kich_test_A = set(split_A['neg_test_slides'])
        kich_test_B = set(split_B['neg_test_slides'])
        
        if kich_test_A == kich_test_B:
            print(f"\n[✓] KICH 测试集一致: {len(kich_test_A)} slides")
            kich_test = list(kich_test_A)
        else:
            print(f"\n[!] KICH 测试集不一致，使用交集")
            print(f"    Task A: {len(kich_test_A)}, Task B: {len(kich_test_B)}")
            kich_test = list(kich_test_A & kich_test_B)
            print(f"    交集: {len(kich_test)}")
        
        kirc_test = split_A['pos_test_slides']
        kirp_test = split_B['pos_test_slides']
        
        print(f"\n测试集构成:")
        print(f"  KIRC: {len(kirc_test)} slides")
        print(f"  KIRP: {len(kirp_test)} slides")
        print(f"  KICH: {len(kich_test)} slides")
        print(f"  总计: {len(kirc_test) + len(kirp_test) + len(kich_test)} slides")
        
        # 构建数据列表
        self.test_data = []
        
        # KIRC
        kirc_dir = os.path.join(self.data_dir, 'TCGA-KIRC')
        for name in kirc_test:
            path = os.path.join(kirc_dir, name)
            if os.path.exists(path):
                self.test_data.append({
                    'path': path,
                    'name': name,
                    'true_class': 'KIRC',
                    'true_idx': self.class_to_idx['KIRC']
                })
        
        # KIRP
        kirp_dir = os.path.join(self.data_dir, 'TCGA-KIRP')
        for name in kirp_test:
            path = os.path.join(kirp_dir, name)
            if os.path.exists(path):
                self.test_data.append({
                    'path': path,
                    'name': name,
                    'true_class': 'KIRP',
                    'true_idx': self.class_to_idx['KIRP']
                })
        
        # KICH
        kich_dir = os.path.join(self.data_dir, 'TCGA-KICH')
        for name in kich_test:
            path = os.path.join(kich_dir, name)
            if os.path.exists(path):
                self.test_data.append({
                    'path': path,
                    'name': name,
                    'true_class': 'KICH',
                    'true_idx': self.class_to_idx['KICH']
                })
        
        print(f"\n成功加载: {len(self.test_data)} slides")
    
    def _load_features(self, path):
        """加载特征"""
        data = np.load(path, allow_pickle=True).item()
        feats = data['feature']
        if isinstance(feats, torch.Tensor):
            feats = feats.numpy()
        return torch.from_numpy(feats).float().to(self.device)
    
    def _inference_student(self, features):
        """
        Student 推理方式（原论文推荐）
        
        原论文: "we use the instance classifier in the student branch to make 
        predictions for all instances in a bag, and then use a simple max-pooling 
        to aggregate the predictions of instances to accomplish bag prediction."
        
        Returns:
            score_A: Task A 的 bag 级别预测分数 (max-pooling of instance predictions)
            score_B: Task B 的 bag 级别预测分数
        """
        with torch.no_grad():
            # ==================== Task A ====================
            # Step 1: Encoder 提取特征
            feat_A = self.encoder_A(features)  # [n_instances, feat_dim]
            
            # Step 2: Student 对每个 instance 预测
            instance_pred_A = self.student_A(feat_A)  # [n_instances, 2]
            
            # Step 3: Softmax 得到每个 instance 的正类概率
            instance_prob_A = torch.softmax(instance_pred_A, dim=1)[:, 1]  # [n_instances]
            
            # Step 4: Max-pooling 聚合为 bag 级别预测
            score_A = instance_prob_A.max().item()
            
            # 同时记录 instance 级别的统计信息（用于分析）
            mean_prob_A = instance_prob_A.mean().item()
            std_prob_A = instance_prob_A.std().item()
            num_pos_A = (instance_prob_A > 0.5).sum().item()
            
            # ==================== Task B ====================
            feat_B = self.encoder_B(features)
            instance_pred_B = self.student_B(feat_B)
            instance_prob_B = torch.softmax(instance_pred_B, dim=1)[:, 1]
            score_B = instance_prob_B.max().item()
            
            mean_prob_B = instance_prob_B.mean().item()
            std_prob_B = instance_prob_B.std().item()
            num_pos_B = (instance_prob_B > 0.5).sum().item()
        
        # 返回额外的统计信息
        stats = {
            'A_mean': mean_prob_A,
            'A_std': std_prob_A,
            'A_num_pos': num_pos_A,
            'A_num_instances': len(instance_prob_A),
            'B_mean': mean_prob_B,
            'B_std': std_prob_B,
            'B_num_pos': num_pos_B,
            'B_num_instances': len(instance_prob_B),
        }
        
        return score_A, score_B, stats
    
    def _inference_teacher(self, features):
        """
        Teacher 推理方式（用于对比）
        
        Returns:
            score_A: Task A 的 bag 级别预测分数 (Teacher 的 bag 输出)
            score_B: Task B 的 bag 级别预测分数
        """
        with torch.no_grad():
            # Task A
            feat_A = self.encoder_A(features)
            _, bag_pred_A, _, _ = self.teacher_A(feat_A)
            score_A = torch.softmax(bag_pred_A, dim=1)[0, 1].item()
            
            # Task B
            feat_B = self.encoder_B(features)
            _, bag_pred_B, _, _ = self.teacher_B(feat_B)
            score_B = torch.softmax(bag_pred_B, dim=1)[0, 1].item()
        
        return score_A, score_B
    
    def _compute_three_class_probs(self, score_A, score_B):
        """
        根据两个二分类分数计算三分类概率
        
        score_A: P(KIRC) - 模型A认为是KIRC的概率
        score_B: P(KIRP) - 模型B认为是KIRP的概率
        
        Returns:
            probs: [P(KICH), P(KIRC), P(KIRP)] - 保证和为1
        """
        # 综合得分计算
        # KIRC: A高，B低 → score_A * (1 - score_B)
        # KIRP: A低，B高 → score_B * (1 - score_A)
        # KICH: A低，B低 → (1 - score_A) * (1 - score_B)
        
        raw_KICH = (1 - score_A) * (1 - score_B)
        raw_KIRC = score_A * (1 - score_B)
        raw_KIRP = score_B * (1 - score_A)
        
        # 归一化确保和为1
        total = raw_KICH + raw_KIRC + raw_KIRP
        if total < 1e-10:
            return np.array([1/3, 1/3, 1/3])
        
        probs = np.array([raw_KICH / total, raw_KIRC / total, raw_KIRP / total])
        probs = probs / probs.sum()  # 确保和为1
        
        return probs
    
    def _make_prediction(self, score_A, score_B):
        """根据分数做出预测"""
        probs = self._compute_three_class_probs(score_A, score_B)
        pred_idx = int(np.argmax(probs))
        pred_class = self.idx_to_class[pred_idx]
        
        # 决策情况分析
        pred_A = 1 if score_A >= self.threshold else 0
        pred_B = 1 if score_B >= self.threshold else 0
        
        if pred_A == 1 and pred_B == 0:
            decision_case = "A=1,B=0"
        elif pred_A == 0 and pred_B == 1:
            decision_case = "A=0,B=1"
        elif pred_A == 0 and pred_B == 0:
            decision_case = "A=0,B=0"
        else:
            decision_case = "A=1,B=1"
        
        return pred_class, pred_idx, probs, decision_case
    
    def run_inference(self):
        """运行推理"""
        print("\n" + "=" * 70)
        print(f"开始推理 (模式: {self.inference_mode})")
        print("=" * 70)
        
        self.results = []
        
        for item in tqdm(self.test_data, desc="Inference"):
            features = self._load_features(item['path'])
            
            result = {
                'slide_name': item['name'],
                'true_class': item['true_class'],
                'true_idx': int(item['true_idx']),
            }
            
            # Student 推理
            if self.inference_mode in ['student', 'both']:
                score_A_stu, score_B_stu, stats = self._inference_student(features)
                pred_class_stu, pred_idx_stu, probs_stu, case_stu = self._make_prediction(score_A_stu, score_B_stu)
                
                result.update({
                    'score_A': float(score_A_stu),
                    'score_B': float(score_B_stu),
                    'pred_class': pred_class_stu,
                    'pred_idx': int(pred_idx_stu),
                    'correct': pred_class_stu == item['true_class'],
                    'prob_KICH': float(probs_stu[0]),
                    'prob_KIRC': float(probs_stu[1]),
                    'prob_KIRP': float(probs_stu[2]),
                    'decision_case': case_stu,
                    # Instance 级别统计
                    'A_instance_mean': float(stats['A_mean']),
                    'A_instance_std': float(stats['A_std']),
                    'A_num_pos_instances': int(stats['A_num_pos']),
                    'A_num_instances': int(stats['A_num_instances']),
                    'B_instance_mean': float(stats['B_mean']),
                    'B_instance_std': float(stats['B_std']),
                    'B_num_pos_instances': int(stats['B_num_pos']),
                    'B_num_instances': int(stats['B_num_instances']),
                })
            
            # Teacher 推理（用于对比）
            if self.inference_mode in ['teacher', 'both']:
                score_A_tea, score_B_tea = self._inference_teacher(features)
                pred_class_tea, pred_idx_tea, probs_tea, case_tea = self._make_prediction(score_A_tea, score_B_tea)
                
                if self.inference_mode == 'both':
                    result.update({
                        'teacher_score_A': float(score_A_tea),
                        'teacher_score_B': float(score_B_tea),
                        'teacher_pred_class': pred_class_tea,
                        'teacher_pred_idx': int(pred_idx_tea),
                        'teacher_correct': pred_class_tea == item['true_class'],
                        'teacher_prob_KICH': float(probs_tea[0]),
                        'teacher_prob_KIRC': float(probs_tea[1]),
                        'teacher_prob_KIRP': float(probs_tea[2]),
                        'teacher_decision_case': case_tea,
                    })
                else:
                    result.update({
                        'score_A': float(score_A_tea),
                        'score_B': float(score_B_tea),
                        'pred_class': pred_class_tea,
                        'pred_idx': int(pred_idx_tea),
                        'correct': pred_class_tea == item['true_class'],
                        'prob_KICH': float(probs_tea[0]),
                        'prob_KIRC': float(probs_tea[1]),
                        'prob_KIRP': float(probs_tea[2]),
                        'decision_case': case_tea,
                    })
            
            self.results.append(result)
        
        print(f"\n推理完成: {len(self.results)} slides")
        return self.results
    
    def compute_metrics(self):
        """计算评估指标"""
        print("\n" + "=" * 70)
        print("计算评估指标")
        print("=" * 70)
        
        self.metrics = {}
        
        # 主要结果（Student 或 Teacher）
        y_true = np.array([r['true_idx'] for r in self.results])
        y_pred = np.array([r['pred_idx'] for r in self.results])
        y_prob = np.array([[r['prob_KICH'], r['prob_KIRC'], r['prob_KIRP']] for r in self.results])
        
        # 归一化概率
        prob_sums = y_prob.sum(axis=1)
        if not np.allclose(prob_sums, 1.0, atol=1e-5):
            y_prob = y_prob / prob_sums[:, np.newaxis]
        
        # 基础指标
        self.metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        for avg in ['macro', 'weighted']:
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
            self.metrics[f'precision_{avg}'] = float(p)
            self.metrics[f'recall_{avg}'] = float(r)
            self.metrics[f'f1_{avg}'] = float(f1)
        
        self.metrics['kappa'] = float(cohen_kappa_score(y_true, y_pred))
        
        # 多类别 AUC
        try:
            self.metrics['auc_ovr_macro'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
            self.metrics['auc_ovr_weighted'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'))
            self.metrics['auc_ovo_macro'] = float(roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro'))
            self.metrics['auc_ovo_weighted'] = float(roc_auc_score(y_true, y_prob, multi_class='ovo', average='weighted'))
        except Exception as e:
            print(f"[Warning] 计算多类别AUC失败: {e}")
            self.metrics['auc_ovr_macro'] = None
            self.metrics['auc_ovr_weighted'] = None
            self.metrics['auc_ovo_macro'] = None
            self.metrics['auc_ovo_weighted'] = None
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        # 各类别指标
        p_class, r_class, f1_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        self.metrics['per_class'] = {}
        for i, name in enumerate(self.class_names):
            self.metrics['per_class'][name] = {
                'precision': float(p_class[i]),
                'recall': float(r_class[i]),
                'f1': float(f1_class[i]),
                'support': int(support[i])
            }
            try:
                y_true_binary = (y_true == i).astype(int)
                auc_class = roc_auc_score(y_true_binary, y_prob[:, i])
                self.metrics['per_class'][name]['auc'] = float(auc_class)
            except:
                self.metrics['per_class'][name]['auc'] = None
        
        # 决策情况统计
        case_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for r in self.results:
            case = r['decision_case']
            case_stats[case]['total'] += 1
            if r['correct']:
                case_stats[case]['correct'] += 1
        
        self.metrics['decision_cases'] = {}
        for case, stats in case_stats.items():
            self.metrics['decision_cases'][case] = {
                'total': int(stats['total']),
                'correct': int(stats['correct']),
                'accuracy': float(stats['correct'] / stats['total']) if stats['total'] > 0 else 0.0
            }
        
        # 如果是 both 模式，也计算 Teacher 的指标
        if self.inference_mode == 'both':
            self._compute_teacher_metrics()
        
        self._print_metrics()
        
        return self.metrics
    
    def _compute_teacher_metrics(self):
        """计算 Teacher 的指标（用于对比）"""
        y_true = np.array([r['true_idx'] for r in self.results])
        y_pred_tea = np.array([r['teacher_pred_idx'] for r in self.results])
        y_prob_tea = np.array([[r['teacher_prob_KICH'], r['teacher_prob_KIRC'], r['teacher_prob_KIRP']] for r in self.results])
        
        self.metrics['teacher'] = {}
        self.metrics['teacher']['accuracy'] = float(accuracy_score(y_true, y_pred_tea))
        
        for avg in ['macro', 'weighted']:
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_tea, average=avg, zero_division=0)
            self.metrics['teacher'][f'f1_{avg}'] = float(f1)
        
        self.metrics['teacher']['kappa'] = float(cohen_kappa_score(y_true, y_pred_tea))
        
        try:
            self.metrics['teacher']['auc_ovr_macro'] = float(roc_auc_score(y_true, y_prob_tea, multi_class='ovr', average='macro'))
        except:
            self.metrics['teacher']['auc_ovr_macro'] = None
    
    def _print_metrics(self):
        """打印评估指标"""
        print("\n" + "=" * 70)
        print(f"【{self.inference_mode.upper()} 推理结果】")
        print("=" * 70)
        
        print(f"\n  Accuracy:              {self.metrics['accuracy']:.4f}")
        print(f"  Macro Precision:       {self.metrics['precision_macro']:.4f}")
        print(f"  Macro Recall:          {self.metrics['recall_macro']:.4f}")
        print(f"  Macro F1:              {self.metrics['f1_macro']:.4f}")
        print(f"  Weighted F1:           {self.metrics['f1_weighted']:.4f}")
        print(f"  Cohen's Kappa:         {self.metrics['kappa']:.4f}")
        
        if self.metrics.get('auc_ovr_macro') is not None:
            print(f"\n  AUC (OvR, macro):      {self.metrics['auc_ovr_macro']:.4f}")
            print(f"  AUC (OvR, weighted):   {self.metrics['auc_ovr_weighted']:.4f}")
            print(f"  AUC (OvO, macro):      {self.metrics['auc_ovo_macro']:.4f}")
            print(f"  AUC (OvO, weighted):   {self.metrics['auc_ovo_weighted']:.4f}")
        
        # 如果是 both 模式，打印对比
        if self.inference_mode == 'both' and 'teacher' in self.metrics:
            print("\n" + "-" * 70)
            print("【Student vs Teacher 对比】")
            print("-" * 70)
            print(f"  {'指标':<20} {'Student':<15} {'Teacher':<15}")
            print("  " + "-" * 50)
            print(f"  {'Accuracy':<20} {self.metrics['accuracy']:<15.4f} {self.metrics['teacher']['accuracy']:<15.4f}")
            print(f"  {'Macro F1':<20} {self.metrics['f1_macro']:<15.4f} {self.metrics['teacher']['f1_macro']:<15.4f}")
            if self.metrics.get('auc_ovr_macro') and self.metrics['teacher'].get('auc_ovr_macro'):
                print(f"  {'AUC (OvR)':<20} {self.metrics['auc_ovr_macro']:<15.4f} {self.metrics['teacher']['auc_ovr_macro']:<15.4f}")
        
        print("\n" + "=" * 70)
        print("【各类别指标】")
        print("=" * 70)
        print(f"\n  {'类别':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12} {'Support':<10}")
        print("  " + "-" * 66)
        
        for name in self.class_names:
            m = self.metrics['per_class'][name]
            auc_str = f"{m['auc']:.4f}" if m['auc'] is not None else "N/A"
            print(f"  {name:<8} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {auc_str:<12} {m['support']:<10}")
        
        print("\n" + "=" * 70)
        print("【混淆矩阵】")
        print("=" * 70)
        cm = np.array(self.metrics['confusion_matrix'])
        print(f"\n  {'':>10}", end='')
        for name in self.class_names:
            print(f"{name:>10}", end='')
        print()
        for i, name in enumerate(self.class_names):
            print(f"  {name:>10}", end='')
            for j in range(3):
                print(f"{cm[i,j]:>10}", end='')
            print()
        
        print("\n" + "=" * 70)
        print("【决策情况分析】")
        print("=" * 70)
        print(f"\n  {'情况':<12} {'样本数':<10} {'正确数':<10} {'准确率':<10}")
        print("  " + "-" * 42)
        
        for case in sorted(self.metrics['decision_cases'].keys()):
            stats = self.metrics['decision_cases'][case]
            print(f"  {case:<12} {stats['total']:<10} {stats['correct']:<10} {stats['accuracy']:<10.4f}")
    
    def visualize(self):
        """生成可视化"""
        print("\n" + "=" * 70)
        print("生成可视化")
        print("=" * 70)
        
        plt.rcParams['font.size'] = 10
        
        self._plot_confusion_matrix()
        self._plot_roc_curves()
        self._plot_score_scatter()
        self._plot_score_distribution()
        self._plot_metrics_bar()
        self._plot_decision_analysis()
        
        # Student 特有的可视化
        if self.inference_mode in ['student', 'both']:
            self._plot_instance_statistics()
        
        print(f"\n可视化已保存到: {self.output_dir}")
    
    def _plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        cm = np.array(self.metrics['confusion_matrix'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Confusion Matrix (Counts)')
        
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title('Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self):
        """绘制 ROC 曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        y_true = np.array([r['true_idx'] for r in self.results])
        y_prob = np.array([[r['prob_KICH'], r['prob_KIRC'], r['prob_KIRP']] for r in self.results])
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for i, (name, color) in enumerate(zip(self.class_names, colors)):
            y_true_binary = (y_true == i).astype(int)
            y_score = y_prob[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            auc = self.metrics['per_class'][name]['auc']
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            
            axes[i].plot(fpr, tpr, color=color, linewidth=2, label=f'AUC = {auc_str}')
            axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1)
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{name} vs Rest')
            axes[i].legend(loc='lower right')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_score_scatter(self):
        """绘制分数散点图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {'KICH': '#2ecc71', 'KIRC': '#3498db', 'KIRP': '#e74c3c'}
        markers = {'KICH': 'o', 'KIRC': 's', 'KIRP': '^'}
        
        for name in self.class_names:
            data = [r for r in self.results if r['true_class'] == name]
            x = [r['score_A'] for r in data]
            y = [r['score_B'] for r in data]
            ax.scatter(x, y, c=colors[name], marker=markers[name], label=name, alpha=0.7, s=60)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax.text(0.75, 0.25, 'KIRC\n(A=1,B=0)', ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.text(0.25, 0.75, 'KIRP\n(A=0,B=1)', ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        ax.text(0.25, 0.25, 'KICH\n(A=0,B=0)', ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(0.75, 0.75, 'Conflict\n(A=1,B=1)', ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Score A: P(KIRC) [Student Max-Pooling]', fontsize=12)
        ax.set_ylabel('Score B: P(KIRP) [Student Max-Pooling]', fontsize=12)
        ax.set_title('Score Distribution by True Class (Student Inference)', fontsize=14)
        ax.legend(loc='upper left')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_scatter.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distribution(self):
        """绘制分数分布"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, name in enumerate(self.class_names):
            data = [r for r in self.results if r['true_class'] == name]
            scores_A = [r['score_A'] for r in data]
            scores_B = [r['score_B'] for r in data]
            
            axes[0, idx].hist(scores_A, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
            axes[0, idx].axvline(x=0.5, color='red', linestyle='--')
            axes[0, idx].set_xlabel('Score A: P(KIRC)')
            axes[0, idx].set_title(f'{name} (n={len(data)})')
            
            axes[1, idx].hist(scores_B, bins=20, alpha=0.7, color='#e74c3c', edgecolor='black')
            axes[1, idx].axvline(x=0.5, color='red', linestyle='--')
            axes[1, idx].set_xlabel('Score B: P(KIRP)')
        
        axes[0, 0].set_ylabel('Count')
        axes[1, 0].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_bar(self):
        """绘制指标条形图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics_names = ['Accuracy', 'Macro P', 'Macro R', 'Macro F1', 'Kappa']
        metrics_values = [
            self.metrics['accuracy'],
            self.metrics['precision_macro'],
            self.metrics['recall_macro'],
            self.metrics['f1_macro'],
            self.metrics['kappa']
        ]
        
        if self.metrics.get('auc_ovr_macro') is not None:
            metrics_names.append('AUC(OvR)')
            metrics_values.append(self.metrics['auc_ovr_macro'])
        
        bars = axes[0].bar(metrics_names, metrics_values, color='#3498db', edgecolor='black')
        axes[0].set_ylim(0, 1.1)
        axes[0].set_title(f'Overall Metrics ({self.inference_mode.upper()} Inference)')
        axes[0].set_ylabel('Score')
        
        for bar, val in zip(bars, metrics_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontsize=9)
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        p_vals = [self.metrics['per_class'][n]['precision'] for n in self.class_names]
        r_vals = [self.metrics['per_class'][n]['recall'] for n in self.class_names]
        f1_vals = [self.metrics['per_class'][n]['f1'] for n in self.class_names]
        
        axes[1].bar(x - width, p_vals, width, label='Precision', color='#3498db')
        axes[1].bar(x, r_vals, width, label='Recall', color='#2ecc71')
        axes[1].bar(x + width, f1_vals, width, label='F1', color='#e74c3c')
        
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.class_names)
        axes[1].set_ylim(0, 1.1)
        axes[1].set_title('Per-Class Metrics')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_bar.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_decision_analysis(self):
        """绘制决策情况分析"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        cases = sorted(self.metrics['decision_cases'].keys())
        totals = [self.metrics['decision_cases'][c]['total'] for c in cases]
        accuracies = [self.metrics['decision_cases'][c]['accuracy'] for c in cases]
        
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        axes[0].pie(totals, labels=cases, autopct='%1.1f%%', colors=colors[:len(cases)])
        axes[0].set_title('Decision Case Distribution')
        
        bars = axes[1].bar(cases, accuracies, color=colors[:len(cases)], edgecolor='black')
        axes[1].set_ylim(0, 1.1)
        axes[1].set_title('Accuracy by Decision Case')
        axes[1].set_ylabel('Accuracy')
        
        for bar, total, acc in zip(bars, totals, accuracies):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{acc:.2%}\n(n={total})', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'decision_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_instance_statistics(self):
        """绘制 Instance 级别统计（Student 特有）"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, name in enumerate(self.class_names):
            data = [r for r in self.results if r['true_class'] == name]
            
            # Task A: Instance 平均概率
            mean_probs_A = [r['A_instance_mean'] for r in data]
            axes[0, idx].hist(mean_probs_A, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
            axes[0, idx].axvline(x=0.5, color='red', linestyle='--')
            axes[0, idx].set_xlabel('Mean Instance Prob (Task A)')
            axes[0, idx].set_title(f'{name}: Instance Mean Prob A')
            
            # Task B: Instance 平均概率
            mean_probs_B = [r['B_instance_mean'] for r in data]
            axes[1, idx].hist(mean_probs_B, bins=20, alpha=0.7, color='#e74c3c', edgecolor='black')
            axes[1, idx].axvline(x=0.5, color='red', linestyle='--')
            axes[1, idx].set_xlabel('Mean Instance Prob (Task B)')
            axes[1, idx].set_title(f'{name}: Instance Mean Prob B')
        
        axes[0, 0].set_ylabel('Count')
        axes[1, 0].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'instance_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """保存结果"""
        print("\n" + "=" * 70)
        print("保存结果")
        print("=" * 70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        output = {
            'meta': {
                'timestamp': timestamp,
                'inference_mode': self.inference_mode,
                'checkpoint_A': self.checkpoint_A,
                'checkpoint_B': self.checkpoint_B,
                'train_teacher_auc_A': self.train_teacher_auc_A if isinstance(self.train_teacher_auc_A, str) else float(self.train_teacher_auc_A),
                'train_student_auc_A': self.train_student_auc_A if isinstance(self.train_student_auc_A, str) else float(self.train_student_auc_A),
                'train_teacher_auc_B': self.train_teacher_auc_B if isinstance(self.train_teacher_auc_B, str) else float(self.train_teacher_auc_B),
                'train_student_auc_B': self.train_student_auc_B if isinstance(self.train_student_auc_B, str) else float(self.train_student_auc_B),
                'threshold': float(self.threshold),
                'n_samples': len(self.results)
            },
            'metrics': convert_to_serializable(self.metrics),
            'predictions': convert_to_serializable(self.results)
        }
        
        json_path = os.path.join(self.output_dir, 'results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  [✓] JSON: {json_path}")
        
        # CSV
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"  [✓] CSV: {csv_path}")
        
        # Summary
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"TCGA-RCC 三分类推理结果 ({self.inference_mode.upper()} 推理)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"推理模式: {self.inference_mode}\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"样本数: {len(self.results)}\n\n")
            
            f.write(f"Task A - Teacher AUC: {self.train_teacher_auc_A}, Student AUC: {self.train_student_auc_A}\n")
            f.write(f"Task B - Teacher AUC: {self.train_teacher_auc_B}, Student AUC: {self.train_student_auc_B}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("总体指标\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Accuracy:        {self.metrics['accuracy']:.4f}\n")
            f.write(f"  Macro F1:        {self.metrics['f1_macro']:.4f}\n")
            f.write(f"  Weighted F1:     {self.metrics['f1_weighted']:.4f}\n")
            f.write(f"  Cohen's Kappa:   {self.metrics['kappa']:.4f}\n")
            
            if self.metrics.get('auc_ovr_macro') is not None:
                f.write(f"  AUC (OvR):       {self.metrics['auc_ovr_macro']:.4f}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("各类别指标\n")
            f.write("-" * 70 + "\n")
            for name in self.class_names:
                m = self.metrics['per_class'][name]
                f.write(f"  {name}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}\n")
        
        print(f"  [✓] Summary: {summary_path}")
        
        # Errors
        errors = [r for r in self.results if not r['correct']]
        if errors:
            error_df = pd.DataFrame(errors)
            error_path = os.path.join(self.output_dir, 'errors.csv')
            error_df.to_csv(error_path, index=False)
            print(f"  [✓] Errors: {error_path} ({len(errors)} samples)")
    
    def run(self):
        """运行完整流程"""
        self.run_inference()
        self.compute_metrics()
        self.visualize()
        self.save_results()
        
        n_correct = sum(r['correct'] for r in self.results)
        auc_str = f"{self.metrics['auc_ovr_macro']:.4f}" if self.metrics.get('auc_ovr_macro') else "N/A"
        
        print("\n" + "=" * 70)
        print("推理完成!")
        print("=" * 70)
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  三分类结果汇总 ({self.inference_mode.upper()} 推理)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  样本数:     {len(self.results):<10}                                │
│  正确数:     {n_correct:<10}                                        │
│                                                                     │
│  【主要指标】                                                       │
│  Accuracy:   {self.metrics['accuracy']:.4f}                         │
│  Macro F1:   {self.metrics['f1_macro']:.4f}                         │
│  AUC (OvR):  {auc_str:<10}                                          │
│  Kappa:      {self.metrics['kappa']:.4f}                            │
│                                                                     │
│  【各类别 F1】                                                      │
│  KICH:       {self.metrics['per_class']['KICH']['f1']:.4f}          │
│  KIRC:       {self.metrics['per_class']['KIRC']['f1']:.4f}          │
│  KIRP:       {self.metrics['per_class']['KIRP']['f1']:.4f}          │
│                                                                     │
│  结果保存在: {self.output_dir:<40}│
└─────────────────────────────────────────────────────────────────────┘
""")


def get_parser():
    parser = argparse.ArgumentParser(description='TCGA-RCC Three-Class Inference (Student)')
    
    parser.add_argument('--checkpoint_A', type=str, required=True,
                        help='Task A checkpoint path')
    parser.add_argument('--checkpoint_B', type=str, required=True,
                        help='Task B checkpoint path')
    parser.add_argument('--split_file_A', type=str, required=True,
                        help='Task A data split JSON')
    parser.add_argument('--split_file_B', type=str, required=True,
                        help='Task B data split JSON')
    parser.add_argument('--data_dir', type=str, default='/workspace/cpfs-data/WENO',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--mode', type=str, default='student',
                        choices=['student', 'teacher', 'both'],
                        help='Inference mode: student (原论文方式), teacher, both')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'three_class_{args.mode}_{timestamp}')
    
    inferencer = ThreeClassInferenceStudent(
        checkpoint_A=args.checkpoint_A,
        checkpoint_B=args.checkpoint_B,
        split_file_A=args.split_file_A,
        split_file_B=args.split_file_B,
        data_dir=args.data_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        device=args.device,
        inference_mode=args.mode
    )
    
    inferencer.run()