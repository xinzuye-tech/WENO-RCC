'''
# ============================================================
# 方式1：先训练 Task A，再训练 Task B（共享 KICH）
# ============================================================

# Step 1: 训练 Task A（自动创建并保存划分）
python train_TCGA_RCC.py --task A --epochs 200 --lr 0.005

# Step 2: 训练 Task B，使用 Task A 的 KICH 划分
python train_TCGA_RCC.py --task B --epochs 200 --lr 0.005 \
    --kich_split_file ./checkpoints/TaskA/xxx/data_split.json

# ============================================================
# 方式2：使用已保存的完整划分文件
# ============================================================

# 使用 Task A 的完整划分重新训练
python train_TCGA_RCC.py --task A --epochs 200 \
    --split_file ./checkpoints/TaskA/xxx/data_split.json

# 使用 Task B 的完整划分重新训练
python train_TCGA_RCC.py --task B --epochs 200 \
    --split_file ./checkpoints/TaskB/yyy/data_split.json

# 实验组2: WENO 有 HPM (新实验)
# ================================================================
# --smoothE 150 表示从第 150 epoch 启用 HPM

python train_TCGA_RCC.py --task A --epochs 200 --lr 0.005 --seed 42 \
    --smoothE 50 \
    --StuFilterType FilterNegInstance__ThreProb50 \
    --save_dir ./checkpoints_with_hpm \
    --comment "WENO_with_HPM"

python train_TCGA_RCC.py --task B --epochs 200 --lr 0.005 --seed 42 \
    --smoothE 50 \
    --StuFilterType FilterNegInstance__ThreProb50 \
    --kich_split_file ./checkpoints_with_hpm/TaskA/xxx/data_split.json \
    --save_dir ./checkpoints_with_hpm \
    --comment "WENO_with_HPM"
'''

import argparse
import warnings
import os
import time
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
from models.alexnet import camelyon_feat_projecter, teacher_DSMIL_head, student_head
from Datasets_loader.dataset_TCGA_RCC import TCGA_RCC_Feat
import datetime
import utliz
import util
import random
from tqdm import tqdm
import copy


class Optimizer:
    def __init__(self, model_encoder, model_teacherHead, model_studentHead,
                 optimizer_encoder, optimizer_teacherHead, optimizer_studentHead,
                 train_bagloader, train_instanceloader, test_bagloader, test_instanceloader,
                 writer=None, num_epoch=100,
                 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 PLPostProcessMethod='NegGuide', StuFilterType='ReplaceAS', smoothE=100,
                 stu_loss_weight_neg=0.1, stuOptPeriod=1,
                 save_dir='./checkpoints', save_interval=50):
        self.model_encoder = model_encoder
        self.model_teacherHead = model_teacherHead
        self.model_studentHead = model_studentHead
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_teacherHead = optimizer_teacherHead
        self.optimizer_studentHead = optimizer_studentHead
        self.train_bagloader = train_bagloader
        self.train_instanceloader = train_instanceloader
        self.test_bagloader = test_bagloader
        self.test_instanceloader = test_instanceloader
        self.writer = writer
        self.num_epoch = num_epoch
        self.dev = dev
        self.log_period = 10
        self.PLPostProcessMethod = PLPostProcessMethod
        self.StuFilterType = StuFilterType
        self.smoothE = smoothE
        self.stu_loss_weight_neg = stu_loss_weight_neg
        self.stuOptPeriod = stuOptPeriod
        
        # 保存相关
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.best_teacher_auc = 0.0
        self.best_student_auc = 0.0
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, epoch, suffix=''):
        """保存模型checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_encoder_state_dict': self.model_encoder.state_dict(),
            'model_teacherHead_state_dict': self.model_teacherHead.state_dict(),
            'model_studentHead_state_dict': self.model_studentHead.state_dict(),
            'optimizer_encoder_state_dict': self.optimizer_encoder.state_dict(),
            'optimizer_teacherHead_state_dict': self.optimizer_teacherHead.state_dict(),
            'optimizer_studentHead_state_dict': self.optimizer_studentHead.state_dict(),
            'best_teacher_auc': self.best_teacher_auc,
            'best_student_auc': self.best_student_auc,
        }
        
        if suffix:
            filename = os.path.join(self.save_dir, f'checkpoint_{suffix}.pth')
        else:
            filename = os.path.join(self.save_dir, f'checkpoint_epoch{epoch}.pth')
        
        torch.save(checkpoint, filename)
        print(f"[SAVE] Checkpoint saved to {filename}")
        return filename

    def load_checkpoint(self, checkpoint_path):
        """加载模型checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
        self.model_teacherHead.load_state_dict(checkpoint['model_teacherHead_state_dict'])
        self.model_studentHead.load_state_dict(checkpoint['model_studentHead_state_dict'])
        self.optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
        self.optimizer_teacherHead.load_state_dict(checkpoint['optimizer_teacherHead_state_dict'])
        self.optimizer_studentHead.load_state_dict(checkpoint['optimizer_studentHead_state_dict'])
        self.best_teacher_auc = checkpoint.get('best_teacher_auc', 0.0)
        self.best_student_auc = checkpoint.get('best_student_auc', 0.0)
        
        print(f"[LOAD] Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']

    def optimize(self):
        self.Bank_all_Bags_label = None
        self.Bank_all_instances_pred_byTeacher = None
        self.Bank_all_instances_feat_byTeacher = None
        self.Bank_all_instances_pred_processed = None
        self.Bank_all_instances_pred_byStudent = None

        for epoch in range(self.num_epoch):
            self.optimize_teacher(epoch)
            teacher_auc = self.evaluate_teacher(epoch)
            
            if epoch % self.stuOptPeriod == 0:
                self.optimize_student(epoch)
                student_auc = self.evaluate_student(epoch)
            else:
                student_auc = self.best_student_auc
            
            # 保存最佳Teacher模型
            if teacher_auc > self.best_teacher_auc:
                self.best_teacher_auc = teacher_auc
                self.save_checkpoint(epoch, suffix='best_teacher')
                print(f"[BEST] New best Teacher AUC: {teacher_auc:.4f}")
            
            # 保存最佳Student模型
            if student_auc > self.best_student_auc:
                self.best_student_auc = student_auc
                self.save_checkpoint(epoch, suffix='best_student')
                print(f"[BEST] New best Student AUC: {student_auc:.4f}")
            
            # 定期保存checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)
        
        # 保存最终模型
        self.save_checkpoint(self.num_epoch - 1, suffix='final')
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Teacher AUC: {self.best_teacher_auc:.4f}")
        print(f"Best Student AUC: {self.best_student_auc:.4f}")
        print(f"{'='*60}")
        
        return 0

    def optimize_teacher(self, epoch):
        self.model_encoder.train()
        self.model_teacherHead.train()
        self.model_studentHead.eval()
        criterion = torch.nn.CrossEntropyLoss()
        loader = self.train_bagloader
        patch_label_gt = []
        patch_label_pred = []
        bag_label_gt = []
        bag_label_pred = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc=f'Epoch {epoch} Teacher training')):
            # 适配新的label格式: [patch_labels, slide_label, slide_index, slide_name]
            # 只处理tensor类型的label
            for i in range(len(label)):
                if torch.is_tensor(label[i]):
                    label[i] = label[i].to(self.dev)
            
            selected = selected.squeeze(0) if torch.is_tensor(selected) else selected
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            feat = self.model_encoder(data.squeeze(0))
            
            # label[1] 是 slide_label
            if epoch > self.smoothE:
                if "FilterNegInstance" in self.StuFilterType:
                    if label[1] == 1:
                        with torch.no_grad():
                            pred_byStudent = self.model_studentHead(feat)
                            pred_byStudent = torch.softmax(pred_byStudent, dim=1)[:, 1]
                        if '_Top' in self.StuFilterType:
                            idx_to_keep = torch.topk(-pred_byStudent, k=int(self.StuFilterType.split('_Top')[-1]))[1]
                        elif '_ThreProb' in self.StuFilterType:
                            idx_to_keep = torch.where(pred_byStudent <= int(self.StuFilterType.split('_ThreProb')[-1])/100.0)[0]
                            if idx_to_keep.shape[0] == 0:
                                idx_to_keep = torch.topk(pred_byStudent, k=1)[1]
                        feat_removedNeg = feat[idx_to_keep]
                        instance_attn_score, bag_prediction, _, _  = self.model_teacherHead(feat_removedNeg)
                        instance_attn_score = torch.cat([instance_attn_score, instance_attn_score[:, 1].min()*torch.ones(feat.shape[0]-instance_attn_score.shape[0], 2).to(instance_attn_score.device)], dim=0)
                    else:
                        instance_attn_score, bag_prediction, _, _  = self.model_teacherHead(feat)
                else:
                    instance_attn_score, bag_prediction, _, _  = self.model_teacherHead(feat)
            else:
                instance_attn_score, bag_prediction, _, _  = self.model_teacherHead(feat)

            max_id = torch.argmax(instance_attn_score[:, 1])
            bag_pred_byMax = instance_attn_score[max_id, :].squeeze(0)
            bag_loss = criterion(bag_prediction, label[1])
            bag_loss_byMax = criterion(bag_pred_byMax.unsqueeze(0), label[1])
            loss_teacher = 0.5 * bag_loss + 0.5 * bag_loss_byMax

            self.optimizer_encoder.zero_grad()
            self.optimizer_teacherHead.zero_grad()
            loss_teacher.backward()
            self.optimizer_encoder.step()
            self.optimizer_teacherHead.step()

            bag_prediction = 1.0 * torch.softmax(bag_prediction, dim=1) + \
                       0.0 * torch.softmax(bag_pred_byMax.unsqueeze(0), dim=1)

            patch_label_pred.append(instance_attn_score[:, 1].detach().squeeze(0))
            patch_label_gt.append(label[0].squeeze(0))  # patch_labels
            bag_label_pred.append(bag_prediction.detach()[0, 1])
            bag_label_gt.append(label[1])  # slide_label
            
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Teacher', loss_teacher.item(), niter)

        patch_label_pred = torch.cat(patch_label_pred)
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_pred = torch.tensor(bag_label_pred)
        bag_label_gt = torch.cat(bag_label_gt)

        self.estimated_AttnScore_norm_para_min = patch_label_pred.min()
        self.estimated_AttnScore_norm_para_max = patch_label_pred.max()
        patch_label_pred_normed = self.norm_AttnScore2Prob(patch_label_pred)
        instance_auc_ByTeacher = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))

        bag_auc_ByTeacher = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_pred.reshape(-1))
        self.writer.add_scalar('train_instance_AUC_byTeacher', instance_auc_ByTeacher, epoch)
        self.writer.add_scalar('train_bag_AUC_byTeacher', bag_auc_ByTeacher, epoch)
        return 0

    def norm_AttnScore2Prob(self, attn_score):
        prob = (attn_score - self.estimated_AttnScore_norm_para_min) / (self.estimated_AttnScore_norm_para_max - self.estimated_AttnScore_norm_para_min)
        return prob

    def optimize_student(self, epoch):
        self.model_teacherHead.train()
        self.model_encoder.train()
        self.model_studentHead.train()
        loader = self.train_instanceloader
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc=f'Epoch {epoch} Student training')):
            # 适配新的label格式: [patch_label, slide_label, slide_index, slide_name]
            for i in range(len(label)):
                if torch.is_tensor(label[i]):
                    label[i] = label[i].to(self.dev)
            
            selected = selected.squeeze(0) if torch.is_tensor(selected) else selected
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            feat = self.model_encoder(data)
            with torch.no_grad():
                instance_attn_score, _, _, _ = self.model_teacherHead(feat)
                pseudo_instance_label = self.norm_AttnScore2Prob(instance_attn_score[:, 1]).clamp(min=1e-5, max=1-1e-5).squeeze(0)
                pseudo_instance_label[label[1] == 0] = 0  # label[1] 是 slide_label

            patch_prediction = self.model_studentHead(feat)
            patch_prediction = torch.softmax(patch_prediction, dim=1)

            loss_student = -1. * torch.mean(self.stu_loss_weight_neg * (1-pseudo_instance_label) * torch.log(patch_prediction[:, 0] + 1e-5) +
                                            (1-self.stu_loss_weight_neg) * pseudo_instance_label * torch.log(patch_prediction[:, 1] + 1e-5))
            self.optimizer_encoder.zero_grad()
            self.optimizer_studentHead.zero_grad()
            loss_student.backward()
            self.optimizer_encoder.step()
            self.optimizer_studentHead.step()

            patch_corresponding_slide_idx[selected, 0] = label[2]  # label[2] 是 slide_index
            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]  # label[0] 是 patch_label
            bag_label_gt[selected, 0] = label[1]  # label[1] 是 slide_label
            
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Student', loss_student.item(), niter)

        instance_auc_ByStudent = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred.reshape(-1))
        self.writer.add_scalar('train_instance_AUC_byStudent', instance_auc_ByStudent, epoch)

        bag_label_gt_coarse = []
        bag_label_prediction = []
        available_bag_idx = patch_corresponding_slide_idx.unique()
        for bag_idx_i in available_bag_idx:
            idx_same_bag_i = torch.where(patch_corresponding_slide_idx == bag_idx_i)
            if bag_label_gt[idx_same_bag_i].max() != bag_label_gt[idx_same_bag_i].max():
                raise
            bag_label_gt_coarse.append(bag_label_gt[idx_same_bag_i].max())
            bag_label_prediction.append(patch_label_pred[idx_same_bag_i].max())
        bag_label_gt_coarse = torch.tensor(bag_label_gt_coarse)
        bag_label_prediction = torch.tensor(bag_label_prediction)
        bag_auc_ByStudent = utliz.cal_auc(bag_label_gt_coarse.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('train_bag_AUC_byStudent', bag_auc_ByStudent, epoch)
        return 0

    def evaluate_teacher(self, epoch):
        self.model_encoder.eval()
        self.model_teacherHead.eval()
        loader = self.test_bagloader
        patch_label_gt = []
        patch_label_pred = []
        bag_label_gt = []
        bag_label_prediction_withAttnScore = []
        
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc=f'Epoch {epoch} Teacher evaluating')):
            # 适配新的label格式
            for i in range(len(label)):
                if torch.is_tensor(label[i]):
                    label[i] = label[i].to(self.dev)
            
            selected = selected.squeeze(0) if torch.is_tensor(selected) else selected
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            with torch.no_grad():
                feat = self.model_encoder(data.squeeze(0))
                instance_attn_score, bag_prediction_withAttnScore, _, _ = self.model_teacherHead(feat)
                bag_prediction_withAttnScore = torch.softmax(bag_prediction_withAttnScore, 1)

            patch_label_pred.append(instance_attn_score[:, 1].detach().squeeze(0))
            patch_label_gt.append(label[0].squeeze(0))  # patch_labels
            bag_label_prediction_withAttnScore.append(bag_prediction_withAttnScore.detach()[0, 1])
            bag_label_gt.append(label[1])  # slide_label

        patch_label_pred = torch.cat(patch_label_pred)
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_prediction_withAttnScore = torch.tensor(bag_label_prediction_withAttnScore)
        bag_label_gt = torch.cat(bag_label_gt)

        patch_label_pred_normed = (patch_label_pred - patch_label_pred.min()) / (patch_label_pred.max() - patch_label_pred.min())
        instance_auc_ByTeacher = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))
        bag_auc_ByTeacher_withAttnScore = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_prediction_withAttnScore.reshape(-1))
        self.writer.add_scalar('test_instance_AUC_byTeacher', instance_auc_ByTeacher, epoch)
        self.writer.add_scalar('test_bag_AUC_byTeacher', bag_auc_ByTeacher_withAttnScore, epoch)
        
        return bag_auc_ByTeacher_withAttnScore

    def evaluate_student(self, epoch):
        self.model_encoder.eval()
        self.model_studentHead.eval()
        loader = self.test_instanceloader
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc=f'Epoch {epoch} Student evaluating')):
            # 适配新的label格式
            for i in range(len(label)):
                if torch.is_tensor(label[i]):
                    label[i] = label[i].to(self.dev)
            
            selected = selected.squeeze(0) if torch.is_tensor(selected) else selected
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            with torch.no_grad():
                feat = self.model_encoder(data)
                patch_prediction = self.model_studentHead(feat)
                patch_prediction = torch.softmax(patch_prediction, dim=1)

            patch_corresponding_slide_idx[selected, 0] = label[2]  # slide_index
            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]  # patch_label
            bag_label_gt[selected, 0] = label[1]  # slide_label

        instance_auc_ByStudent = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred.reshape(-1))
        self.writer.add_scalar('test_instance_AUC_byStudent', instance_auc_ByStudent, epoch)

        bag_label_gt_coarse = []
        bag_label_prediction = []
        available_bag_idx = patch_corresponding_slide_idx.unique()
        for bag_idx_i in available_bag_idx:
            idx_same_bag_i = torch.where(patch_corresponding_slide_idx == bag_idx_i)
            if bag_label_gt[idx_same_bag_i].max() != bag_label_gt[idx_same_bag_i].max():
                raise
            bag_label_gt_coarse.append(bag_label_gt[idx_same_bag_i].max())
            bag_label_prediction.append(patch_label_pred[idx_same_bag_i].max())
        bag_label_gt_coarse = torch.tensor(bag_label_gt_coarse)
        bag_label_prediction = torch.tensor(bag_label_prediction)
        bag_auc_ByStudent = utliz.cal_auc(bag_label_gt_coarse.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('test_bag_AUC_byStudent', bag_auc_ByStudent, epoch)
        
        return bag_auc_ByStudent


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='WENO Training on TCGA-RCC')
    # 任务选择参数
    parser.add_argument('--task', default='A', type=str, choices=['A', 'B'],
                        help='Task selection: A (KIRC vs KICH) or B (KIRP vs KICH)')
    parser.add_argument('--data_dir', default='/workspace/cpfs-data/WENO', type=str,
                        help='Base directory for TCGA-RCC data')
    
    # optimizer
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size (default: 512)')
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate (default: 0.005)')
    parser.add_argument('--lrdrop', default=1500, type=int, help='multiply LR by 0.5 every (default: 1500 epochs)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--dtype', default='f64', choices=['f64', 'f32'], type=str, help='SK-algo dtype (default: f64)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=0, type=int, help='number workers (default: 0)')
    parser.add_argument('--comment', default='TCGA_RCC_WENO', type=str, help='name for tensorboardX')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    parser.add_argument('--PLPostProcessMethod', default='NegGuide', type=str,
                        help='Post-processing method of Attention Scores to build Pseudo Labels',
                        choices=['NegGuide', 'NegGuide_TopK', 'NegGuide_Similarity'])
    parser.add_argument('--StuFilterType', default='FilterNegInstance__ThreProb50', type=str,
                        help='Type of using Student Prediction to improve Teacher')
    parser.add_argument('--smoothE', default=9999, type=int, help='num of epoch to apply StuFilter')
    parser.add_argument('--stu_loss_weight_neg', default=0.1, type=float, help='weight of neg instances in stu training')
    parser.add_argument('--stuOptPeriod', default=1, type=int, help='period of stu optimization')
    
    # checkpoint相关参数
    parser.add_argument('--save_dir', default='./checkpoints', type=str, help='directory to save checkpoints')
    parser.add_argument('--save_interval', default=50, type=int, help='save checkpoint every N epochs')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint to resume from')
    
    # 数据划分相关参数
    parser.add_argument('--split_file', default='', type=str, 
                        help='path to data split JSON file (if provided, will use this split instead of creating new one)')
    parser.add_argument('--kich_split_file', default='', type=str,
                        help='path to KICH split file for Task B to share Task A KICH split (can be Task A data_split.json)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    # 任务描述
    task_desc = {
        'A': 'TCGA-KIRC(1) vs TCGA-KICH(0)',
        'B': 'TCGA-KIRP(1) vs TCGA-KICH(0)'
    }
    
    print("=" * 70)
    print(f"WENO Training on TCGA-RCC")
    print(f"Task {args.task}: {task_desc[args.task]}")
    print("=" * 70)

    # 实验名称包含任务信息
    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + \
           f"_Task{args.task}_{args.comment}" + \
           f"_Seed{args.seed}_Bs{args.batch_size}_lr{args.lr}" + \
           f"_PLPostProcessBy{args.PLPostProcessMethod}" + \
           f"_StuFilterType{args.StuFilterType}" + \
           f"_smoothE{args.smoothE}_weightN{args.stu_loss_weight_neg}_StuOptP{args.stuOptPeriod}"
    
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device
    util.setup_runtime(seed=args.seed, cuda_dev_id=list(np.unique(args.modeldevice + args.device)))

    print(f"Experiment name: {name}", flush=True)

    # 创建保存目录（包含实验名称）
    save_dir = os.path.join(args.save_dir, f'Task{args.task}', name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")

    # TensorBoard writer
    writer = SummaryWriter(f'./runs_TCGA_RCC/Task{args.task}/{name}')
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    # Setup model
    model_encoder = camelyon_feat_projecter(input_dim=512, output_dim=512).to('cuda:0')
    model_teacherHead = teacher_DSMIL_head(input_feat_dim=512).to('cuda:0')
    model_studentHead = student_head(input_feat_dim=512).to('cuda:0')

    optimizer_encoder = torch.optim.SGD(model_encoder.parameters(), lr=args.lr)
    optimizer_teacherHead = torch.optim.SGD(model_teacherHead.parameters(), lr=args.lr)
    optimizer_studentHead = torch.optim.SGD(model_studentHead.parameters(), lr=args.lr)

    # ============================================================
    # 数据集加载（支持从文件加载划分和KICH共享）
    # ============================================================
    split_file = args.split_file if args.split_file else None
    kich_split_file = args.kich_split_file if args.kich_split_file else None
    
    # 如果指定了split_file，则忽略kich_split_file（完整划分优先）
    if split_file:
        kich_split_file = None
        print(f"[DATA] Using complete split file: {split_file}")
        print(f"[DATA] kich_split_file will be ignored since split_file is provided")
    elif kich_split_file:
        print(f"[DATA] Using KICH split from: {kich_split_file}")
    
    # 训练集 - instance模式
    train_ds_return_instance = TCGA_RCC_Feat(
        task=args.task,
        train=True,
        return_bag=False,
        base_dir=args.data_dir,
        seed=args.seed,
        split_file=split_file,
        kich_split_file=kich_split_file
    )
    
    # 保存数据划分（仅在未指定split_file时保存）
    if not args.split_file:
        split_save_path = os.path.join(save_dir, 'data_split.json')
        train_ds_return_instance.save_split(split_save_path)
        print(f"[DATA] Data split saved to: {split_save_path}")
        
        # 显示KICH划分来源
        split_info = train_ds_return_instance.get_split_info()
        print(f"[DATA] KICH split source: {split_info.get('kich_split_source', 'self')}")
        
        # 后续数据集使用相同的划分文件
        split_file = split_save_path
        kich_split_file = None  # 已经保存在split_file中了
    else:
        print(f"[DATA] Using existing split file: {args.split_file}")

    # 训练集 - bag模式（使用相同的划分）
    train_ds_return_bag = TCGA_RCC_Feat(
        task=args.task,
        train=True,
        return_bag=True,
        base_dir=args.data_dir,
        seed=args.seed,
        split_file=split_file,
        kich_split_file=kich_split_file
    )
    
    # 验证集 - instance模式（使用相同的划分）
    val_ds_return_instance = TCGA_RCC_Feat(
        task=args.task,
        train=False,
        return_bag=False,
        base_dir=args.data_dir,
        seed=args.seed,
        split_file=split_file,
        kich_split_file=kich_split_file
    )
    
    # 验证集 - bag模式（使用相同的划分）
    val_ds_return_bag = TCGA_RCC_Feat(
        task=args.task,
        train=False,
        return_bag=True,
        base_dir=args.data_dir,
        seed=args.seed,
        split_file=split_file,
        kich_split_file=kich_split_file
    )

    # DataLoader
    train_loader_instance = torch.utils.data.DataLoader(
        train_ds_return_instance, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        drop_last=False
    )
    train_loader_bag = torch.utils.data.DataLoader(
        train_ds_return_bag, 
        batch_size=1, 
        shuffle=True, 
        num_workers=args.workers, 
        drop_last=False
    )
    val_loader_instance = torch.utils.data.DataLoader(
        val_ds_return_instance, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        drop_last=False
    )
    val_loader_bag = torch.utils.data.DataLoader(
        val_ds_return_bag, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.workers, 
        drop_last=False
    )

    # 打印数据摘要
    print(f"\n[Data Summary]")
    print(f"  Task: {args.task} ({task_desc[args.task]})")
    print(f"  Training patches: {len(train_loader_instance.dataset)}")
    print(f"  Training slides: {train_ds_return_bag.num_slides}")
    print(f"  Validation patches: {len(val_loader_instance.dataset)}")
    print(f"  Validation slides: {val_ds_return_bag.num_slides}")
    
    # 打印KICH划分信息
    split_info = train_ds_return_instance.get_split_info()
    print(f"  KICH train slides: {split_info['num_neg_train']}")
    print(f"  KICH test slides: {split_info['num_neg_test']}")
    print(f"  KICH split source: {split_info.get('kich_split_source', 'self')}")

    if torch.cuda.device_count() > 1:
        print("Let's use", len(args.modeldevice), "GPUs for the model")
        if len(args.modeldevice) == 1:
            print('single GPU model', flush=True)
        else:
            model_encoder = nn.DataParallel(model_encoder, device_ids=list(range(len(args.modeldevice))))
            model_teacherHead = nn.DataParallel(model_teacherHead, device_ids=list(range(len(args.modeldevice))))

    # Setup optimizer
    o = Optimizer(
        model_encoder=model_encoder, 
        model_teacherHead=model_teacherHead, 
        model_studentHead=model_studentHead,
        optimizer_encoder=optimizer_encoder, 
        optimizer_teacherHead=optimizer_teacherHead, 
        optimizer_studentHead=optimizer_studentHead,
        train_bagloader=train_loader_bag, 
        train_instanceloader=train_loader_instance,
        test_bagloader=val_loader_bag, 
        test_instanceloader=val_loader_instance,
        writer=writer, 
        num_epoch=args.epochs,
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        PLPostProcessMethod=args.PLPostProcessMethod, 
        StuFilterType=args.StuFilterType, 
        smoothE=args.smoothE,
        stu_loss_weight_neg=args.stu_loss_weight_neg, 
        stuOptPeriod=args.stuOptPeriod,
        save_dir=save_dir,
        save_interval=args.save_interval
    )
    
    # 如果指定了resume，加载checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = o.load_checkpoint(args.resume)
        print(f"Resuming from epoch {start_epoch}")
    
    # Optimize
    o.optimize()
    
    # 打印最终信息
    print("=" * 70)
    print("Training completed!")
    print(f"Task: {args.task} ({task_desc[args.task]})")
    print(f"Best Teacher AUC: {o.best_teacher_auc:.4f}")
    print(f"Best Student AUC: {o.best_student_auc:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"Data split saved to: {os.path.join(save_dir, 'data_split.json')}")
    print("=" * 70)