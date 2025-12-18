"""
使用 Scheduled Sampling 处理 Exposure Bias 的训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging
from pathlib import Path
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from data import FamilyTourBatch, FamilyTourDataset, collate_fn
from model import create_model
from losses import FamilyTourLoss, MetricsCalculator
from exposure_bias import (
    ExposureBiasTrainer,
    ScheduledSamplingScheduler,
    create_exposure_bias_handler
)
import sys
sys.path.append('family_tour_generation')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScheduledSamplingTrainer:
    """
    使用 Scheduled Sampling 的训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        save_dir: str = './checkpoints',
        eb_strategy: str = 'scheduled_sampling'
    ):
        self.model = model
        self.model_config = model_config
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备
        self.device = torch.device(
            train_config.device if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        # Exposure Bias 处理器
        self.eb_trainer = create_exposure_bias_handler(
            model, model_config, strategy=eb_strategy
        )
        
        # 损失函数
        self.criterion = FamilyTourLoss(model_config, train_config)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # 混合精度
        self.scaler = GradScaler()
        self.use_amp = torch.cuda.is_available()
        
        # 状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 指标
        self.metrics_calc = MetricsCalculator()
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {k: 0.0 for k in self.model_config.loss_weights.keys()}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_data in pbar:

            batch = batch_data.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    loss, losses = self.eb_trainer.train_step(
                        batch, self.criterion, self.optimizer
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.train_config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, losses = self.eb_trainer.train_step(
                    batch, self.criterion, self.optimizer
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.train_config.grad_clip
                )
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            for k, v in losses.items():
                loss_components[k] += v.item()
            num_batches += 1
            self.global_step += 1
            
            # 获取当前 TF 概率
            tf_prob = self.eb_trainer.get_current_tf_prob()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'tf_prob': f'{tf_prob:.3f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {'total': avg_loss, **avg_components}
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {k: 0.0 for k in self.model_config.loss_weights.keys()}
        accuracies = {k: 0.0 for k in ['purpose', 'mode', 'driver', 'joint']}
        num_batches = 0
        
        for batch_data in tqdm(self.val_loader, desc='Validating'):
            batch = batch_data.to(self.device)
            
            # 使用纯自回归模式验证 (模拟真实推理)
            predictions = self.model.generate(
                batch.family_attr, batch.member_attr, batch.member_mask
            )
            
            # 调整预测格式以计算损失
            # generate 返回的 purpose, mode 等是索引，需要转换为 logits 格式
            # 这里简化处理，用 teacher forcing 计算损失
            # predictions_tf = self.model(batch, teacher_forcing=True)
            
            loss, losses = self.criterion(
                predictions, batch.activities,
                batch.member_mask, batch.activity_mask
            )
            
            # 计算指标 (使用自回归生成的结果)
            # 需要将生成结果转换为与 target 兼容的格式
            acc = self._compute_generation_accuracy(predictions, batch)
            
            total_loss += loss.item()
            for k, v in losses.items():
                loss_components[k] += v.item()
            for k, v in acc.items():
                accuracies[k] += v
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_components = {f'val_{k}': v / num_batches for k, v in loss_components.items()}
        avg_acc = {f'gen_acc_{k}': v / num_batches for k, v in accuracies.items()}
        
        return {'val_total': avg_loss, **avg_components, **avg_acc}
    
    def _compute_generation_accuracy(self, predictions, batch):
        """计算生成模式下的准确率"""
        # 解析目标
        target_purpose = batch.activities[..., 2:12].argmax(dim=-1)
        target_mode = batch.activities[..., 12:23].argmax(dim=-1)
        target_driver = batch.activities[..., 23:25].argmax(dim=-1)
        target_joint = batch.activities[..., 25:27].argmax(dim=-1)
        
        # 生成的预测已经是索引
        # pred_purpose = predictions['purpose']
        # pred_mode = predictions['mode']
        # pred_driver = predictions['driver']
        # pred_joint = predictions['joint']
        pred_purpose = predictions['purpose'].argmax(dim=-1)
        pred_mode = predictions['mode'].argmax(dim=-1)
        pred_driver = predictions['driver'].argmax(dim=-1)
        pred_joint = predictions['joint'].argmax(dim=-1)
        
        # 截断到相同长度
        max_len = min(pred_purpose.size(2), target_purpose.size(2))
        
        valid_mask = batch.activity_mask[..., :max_len]
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return {k: 0.0 for k in ['purpose', 'mode', 'driver', 'joint']}
        
        acc = {}
        acc['purpose'] = (
            (pred_purpose[..., :max_len] == target_purpose[..., :max_len]) & valid_mask
        ).sum().item() / num_valid
        acc['mode'] = (
            (pred_mode[..., :max_len] == target_mode[..., :max_len]) & valid_mask
        ).sum().item() / num_valid
        acc['driver'] = (
            (pred_driver[..., :max_len] == target_driver[..., :max_len]) & valid_mask
        ).sum().item() / num_valid
        acc['joint'] = (
            (pred_joint[..., :max_len] == target_joint[..., :max_len]) & valid_mask
        ).sum().item() / num_valid
        
        return acc
    
    def train(self):
        """完整训练"""
        logger.info(f"Starting training with Scheduled Sampling on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        
        for epoch in range(self.train_config.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            tf_prob = self.eb_trainer.get_current_tf_prob()
            
            logger.info(
                f"Epoch {epoch} - Loss: {train_metrics['total']:.4f}, "
                f"TF Prob: {tf_prob:.3f}"
            )
            
            # 验证
            if (epoch + 1) % 5 == 0:
                val_metrics = self.validate()
                if val_metrics:
                    logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['val_total']:.4f}")

                    # 打印生成准确率
                    # for k, v in val_metrics.items():
                    #     if 'gen_acc' in k:
                    #         logger.info(f"  {k}: {v:.4f}")

                    # 保存最佳
                    if val_metrics['val_total'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_total']
                        self.save_checkpoint('best_model.pt')
                        logger.info("  New best model saved!")
            
            # 定期保存
            if (epoch + 1) % 50 == 0:
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
            'train_config': self.train_config,
            'eb_trainer_state': self.eb_trainer.state_dict()
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
        
        if 'eb_trainer_state' in checkpoint:
            self.eb_trainer.load_state_dict(checkpoint['eb_trainer_state'])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def main():
    """主函数示例"""
    import numpy as np
    
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
        batch_size=200,
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
    
    # 创建训练器 (使用 Scheduled Sampling)
    trainer = ScheduledSamplingTrainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir='../checkpoints_ss',
        eb_strategy='scheduled_sampling'  # 可选: 'aggressive', 'conservative'
    )
    
    # 训练
    trainer.train()


if __name__ == '__main__':
    main()
