"""
训练脚本
"""
import os
import time
import logging
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from .config import ModelConfig, TrainConfig
from .data import FamilyTourBatch, FamilyTourDataset, collate_fn
from .model import FamilyTourGenerator, create_model
from .losses import FamilyTourLoss, MetricsCalculator


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: FamilyTourGenerator,
        model_config: ModelConfig,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: str = './checkpoints'
    ):
        self.model = model
        self.model_config = model_config
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备
        self.device = torch.device(train_config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = FamilyTourLoss(model_config, train_config)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # 混合精度训练
        self.scaler = GradScaler()
        self.use_amp = torch.cuda.is_available()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 指标计算
        self.metrics_calc = MetricsCalculator()
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {k: 0.0 for k in self.model_config.loss_weights.keys()}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_data in pbar:
            # 构建batch
            batch = batch_data.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(batch, teacher_forcing=True)
                    loss, losses = self.criterion(
                        predictions, batch.activities,
                        batch.member_mask, batch.activity_mask
                    )
                
                # 反向传播 (混合精度)
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_config.grad_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(batch, teacher_forcing=True)
                loss, losses = self.criterion(
                    predictions, batch.activities,
                    batch.member_mask, batch.activity_mask
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_clip
                )
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            for k, v in losses.items():
                loss_components[k] += v.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {'total': avg_loss, **avg_components}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {k: 0.0 for k in self.model_config.loss_weights.keys()}
        accuracies = {k: 0.0 for k in ['purpose', 'mode', 'driver', 'joint']}
        time_maes = {'start_time': 0.0, 'end_time': 0.0}
        num_batches = 0
        
        for batch_data in tqdm(self.val_loader, desc='Validating'):
            batch = batch_data.to(self.device)
            
            predictions = self.model(batch, teacher_forcing=True)
            loss, losses = self.criterion(
                predictions, batch.activities,
                batch.member_mask, batch.activity_mask
            )
            
            # 计算指标
            acc = self.metrics_calc.compute_accuracy(
                predictions, batch.activities, batch.activity_mask
            )
            mae = self.metrics_calc.compute_time_mae(
                predictions, batch.activities, batch.activity_mask
            )
            
            total_loss += loss.item()
            for k, v in losses.items():
                loss_components[k] += v.item()
            for k, v in acc.items():
                accuracies[k] += v
            for k, v in mae.items():
                time_maes[k] += v
            num_batches += 1
        
        # 计算平均
        avg_loss = total_loss / num_batches
        avg_components = {f'val_{k}': v / num_batches for k, v in loss_components.items()}
        avg_acc = {f'acc_{k}': v / num_batches for k, v in accuracies.items()}
        avg_mae = {f'mae_{k}': v / num_batches for k, v in time_maes.items()}
        
        return {'val_total': avg_loss, **avg_components, **avg_acc, **avg_mae}
    
    def train(self):
        """完整训练流程"""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        
        for epoch in range(self.train_config.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['total']:.4f}")
            # for k, v in train_metrics.items():
            #     if k != 'total':
            #         logger.info(f"  {k}: {v:.4f}")
            
            # 验证
            val_metrics = self.validate()
            if val_metrics:
                logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['val_total']:.4f}")
                # for k, v in val_metrics.items():
                #     if k != 'val_total':
                #         logger.info(f"  {k}: {v:.4f}")
                
                # 保存最佳模型
                if val_metrics['val_total'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_total']
                    self.save_checkpoint('best_model.pt')
                    logger.info(f"  New best model saved!")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config,
            'train_config': self.train_config
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_dummy_data(model_config: ModelConfig, num_samples: int = 1000):
    """
    创建示例数据 (用于测试)
    实际使用时需要替换为真实数据加载逻辑
    """
    # 随机生成训练数据
    family_data = np.random.randn(num_samples, model_config.family_dim).astype(np.float32)
    member_data = np.random.randn(num_samples, model_config.max_members, model_config.member_dim).astype(np.float32)
    activity_data = np.zeros(
        (num_samples, model_config.max_members, model_config.max_activities, model_config.activity_dim), 
        dtype=np.float32
    )
    
    # 填充activity_data
    # 连续属性 (开始时间, 结束时间的z-score)
    activity_data[..., :2] = np.random.randn(
        num_samples, model_config.max_members, model_config.max_activities, 2
    )
    
    # 目的 one-hot (10类)
    purpose_idx = np.random.randint(
        0, model_config.num_purposes, 
        (num_samples, model_config.max_members, model_config.max_activities)
    )
    for i in range(num_samples):
        for j in range(model_config.max_members):
            for k in range(model_config.max_activities):
                activity_data[i, j, k, 2 + purpose_idx[i, j, k]] = 1
    
    # 方式 one-hot (11类)
    mode_idx = np.random.randint(
        0, model_config.num_modes, 
        (num_samples, model_config.max_members, model_config.max_activities)
    )
    for i in range(num_samples):
        for j in range(model_config.max_members):
            for k in range(model_config.max_activities):
                activity_data[i, j, k, 12 + mode_idx[i, j, k]] = 1
    
    # 驾驶状态 one-hot (2类)
    driver_idx = np.random.randint(
        0, 2, 
        (num_samples, model_config.max_members, model_config.max_activities)
    )
    for i in range(num_samples):
        for j in range(model_config.max_members):
            for k in range(model_config.max_activities):
                activity_data[i, j, k, 23 + driver_idx[i, j, k]] = 1
    
    # 联合出行 one-hot (2类)
    joint_idx = np.random.randint(
        0, 2, 
        (num_samples, model_config.max_members, model_config.max_activities)
    )
    for i in range(num_samples):
        for j in range(model_config.max_members):
            for k in range(model_config.max_activities):
                activity_data[i, j, k, 25 + joint_idx[i, j, k]] = 1
    
    # 成员掩码 (每个家庭3-8个成员)
    member_mask = np.zeros((num_samples, model_config.max_members), dtype=bool)
    for i in range(num_samples):
        num_members = np.random.randint(3, model_config.max_members + 1)
        member_mask[i, :num_members] = True
    
    # 活动掩码 (每个成员2-6个活动)
    activity_mask = np.zeros(
        (num_samples, model_config.max_members, model_config.max_activities), 
        dtype=bool
    )
    for i in range(num_samples):
        for j in range(model_config.max_members):
            if member_mask[i, j]:
                num_activities = np.random.randint(2, model_config.max_activities + 1)
                activity_mask[i, j, :num_activities] = True
    
    return family_data, member_data, activity_data, member_mask, activity_mask


def main():
    """主函数"""
    # 配置
    model_config = ModelConfig(
        family_dim=10,        # 根据实际数据调整
        member_dim=51,        # 根据实际数据调整
        activity_dim=27,
        max_members=8,
        max_activities=6,
        d_model=256,
        num_heads=8,
        num_decoder_layers=20,
        num_inducing_points=16
    )
    
    train_config = TrainConfig(
        batch_size=900,
        learning_rate=1e-4,
        num_epochs=500
    )
    
    # 创建模型
    model = create_model(model_config)
    
    # 创建示例数据 (实际使用时替换为真实数据)
    # family_data, member_data, activity_data, member_mask, activity_mask = create_dummy_data(
    #     model_config, num_samples=1000
    # )
    data_dir = "../数据"
    family_data_train = np.load(f'{data_dir}/family_sample_improved_cluster_train.npy')[:,:10]
    member_data_train = np.load(f'{data_dir}/family_member_sample_improved_cluster_train.npy')
    activity_data_train = np.load(f'{data_dir}/family_activity_train.npy')
    member_mask_train = member_data_train[:,:,-1]
    activity_mask_train = activity_data_train.sum(axis=-1) != 0

    family_data_test = np.load(f'{data_dir}/family_sample_improved_cluster_test.npy')[:,:10]
    member_data_test = np.load(f'{data_dir}/family_member_sample_improved_cluster_test.npy')
    activity_data_test = np.load(f'{data_dir}/family_activity_test.npy')
    member_mask_test = member_data_test[:,:,-1] != 0
    activity_mask_test = activity_data_test.sum(axis=-1) != 0
    
    train_dataset = FamilyTourDataset(
        family_data_train,
        member_data_train,
        activity_data_train,
        member_mask_train,
        activity_mask_train
    )
    
    val_dataset = FamilyTourDataset(
        family_data_test,
        member_data_test,
        activity_data_test,
        member_mask_test,
        activity_mask_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir='../checkpoints'
    )
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
