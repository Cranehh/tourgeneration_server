"""
损失函数模块
- 连续属性: Smooth L1 Loss
- 离散属性: Focal Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from config import ModelConfig, TrainConfig


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (*, num_classes) 未归一化的logits
            targets: (*,) 类别索引
        
        Returns:
            loss: (*,) 或 scalar
        """
        # 计算交叉熵 (不reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 计算p_t
        p = F.softmax(logits, dim=-1)
        p_t = p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        # Focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Focal loss
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FamilyTourLoss(nn.Module):
    """
    家庭活动链生成的损失函数
    
    包含:
    - 连续属性 (时间): Smooth L1 Loss
    - 离散属性 (目的、方式、驾驶状态、联合出行): Focal Loss
    """
    
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        
        # Smooth L1 Loss for continuous
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        
        # Focal Loss for discrete
        self.focal_loss = FocalLoss(
            alpha=train_config.focal_alpha,
            gamma=train_config.focal_gamma,
            reduction='none'
        )
        
        # 损失权重
        self.weights = model_config.loss_weights

        # 新增: 模式预测损失
        self.pattern_loss = None
        self.pattern_loss_weights = self.weights.get('pattern', 0.5)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor,
        pattern_outputs: Dict[str, torch.Tensor] = None  # 新增参数
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: 模型预测
                - 'continuous': (batch, max_members, max_activities, 2)
                - 'purpose': (batch, max_members, max_activities, num_purposes)
                - 'mode': (batch, max_members, max_activities, num_modes)
                - 'driver': (batch, max_members, max_activities, num_driver)
                - 'joint': (batch, max_members, max_activities, num_joint)
            targets: (batch, max_members, max_activities, 27) 目标活动
            member_mask: (batch, max_members) 有效成员
            activity_mask: (batch, max_members, max_activities) 有效活动
        
        Returns:
            total_loss: scalar
            loss_dict: 各部分损失，用于监控
        """
        # 有效位置掩码
        valid_mask = member_mask.unsqueeze(-1) & activity_mask  # (batch, max_members, max_activities)
        num_valid = valid_mask.sum().clamp(min=1)
        
        # 解析目标
        target_continuous = targets[..., :2]                    # (batch, max_members, max_activities, 2)
        target_purpose = targets[..., 2:12].argmax(dim=-1)      # (batch, max_members, max_activities)
        target_mode = targets[..., 12:23].argmax(dim=-1)
        target_driver = targets[..., 23:25].argmax(dim=-1)
        target_joint = targets[..., 25:27].argmax(dim=-1)
        
        losses = {}
        
        # 连续属性损失 (Smooth L1)
        continuous_loss = self.smooth_l1(
            predictions['continuous'], target_continuous
        ).sum(dim=-1)  # (batch, max_members, max_activities)
        losses['continuous'] = (continuous_loss * valid_mask.float()).sum() / num_valid
        
        # 目的损失 (Focal)
        purpose_logits = predictions['purpose'].view(-1, self.model_config.num_purposes)
        purpose_targets = target_purpose.view(-1)
        purpose_loss = self.focal_loss(purpose_logits, purpose_targets)
        purpose_loss = purpose_loss.view_as(target_purpose)
        losses['purpose'] = (purpose_loss * valid_mask.float()).sum() / num_valid
        
        # 方式损失 (Focal)
        mode_logits = predictions['mode'].view(-1, self.model_config.num_modes)
        mode_targets = target_mode.view(-1)
        mode_loss = self.focal_loss(mode_logits, mode_targets)
        mode_loss = mode_loss.view_as(target_mode)
        losses['mode'] = (mode_loss * valid_mask.float()).sum() / num_valid
        
        # 驾驶状态损失 (Focal)
        driver_logits = predictions['driver'].view(-1, self.model_config.num_driver_status)
        driver_targets = target_driver.view(-1)
        driver_loss = self.focal_loss(driver_logits, driver_targets)
        driver_loss = driver_loss.view_as(target_driver)
        losses['driver'] = (driver_loss * valid_mask.float()).sum() / num_valid
        
        # 联合出行损失 (Focal)
        joint_logits = predictions['joint'].view(-1, self.model_config.num_joint_status)
        joint_targets = target_joint.view(-1)
        joint_loss = self.focal_loss(joint_logits, joint_targets)
        joint_loss = joint_loss.view_as(target_joint)
        losses['joint'] = (joint_loss * valid_mask.float()).sum() / num_valid
        
        # 加权总损失
        total_loss = sum(self.weights[k] * v for k, v in losses.items())

        # 新增: 模式预测损失
        if pattern_outputs is not None:
            if self.pattern_loss is None:
                self.pattern_loss = PatternPredictionLoss()

            # 从 pattern_outputs 中获取目标分布
            family_target = pattern_outputs.get('family_pattern_target')
            individual_target = pattern_outputs.get('individual_pattern_target')

            if family_target is not None and individual_target is not None:
                pattern_loss, pattern_losses = self.pattern_loss(
                    pattern_outputs, family_target, individual_target, member_mask
                )
                # losses.update(pattern_losses)
                losses['pattern'] = pattern_loss
                total_loss = total_loss + self.pattern_loss_weights * pattern_loss

        return total_loss, losses
    
    def compute_member_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        计算每个成员的损失 (用于多任务学习)
        
        Returns:
            member_losses: (batch, max_members)
        """
        batch_size = targets.size(0)
        max_members = self.model_config.max_members
        
        # 解析目标
        target_continuous = targets[..., :2]
        target_purpose = targets[..., 2:12].argmax(dim=-1)
        target_mode = targets[..., 12:23].argmax(dim=-1)
        target_driver = targets[..., 23:25].argmax(dim=-1)
        target_joint = targets[..., 25:27].argmax(dim=-1)
        
        # 每个成员的有效活动数
        seq_lengths = activity_mask.sum(dim=-1).clamp(min=1).float()  # (batch, max_members)
        
        member_losses = torch.zeros(batch_size, max_members, device=targets.device)
        
        # 连续属性
        continuous_loss = self.smooth_l1(
            predictions['continuous'], target_continuous
        ).sum(dim=-1)  # (batch, max_members, max_activities)
        continuous_loss = (continuous_loss * activity_mask.float()).sum(dim=-1) / seq_lengths
        member_losses += self.weights['continuous'] * continuous_loss
        
        # 目的
        purpose_loss = self._compute_member_focal_loss(
            predictions['purpose'], target_purpose, 
            self.model_config.num_purposes, activity_mask, seq_lengths
        )
        member_losses += self.weights['purpose'] * purpose_loss
        
        # 方式
        mode_loss = self._compute_member_focal_loss(
            predictions['mode'], target_mode,
            self.model_config.num_modes, activity_mask, seq_lengths
        )
        member_losses += self.weights['mode'] * mode_loss
        
        # 驾驶状态
        driver_loss = self._compute_member_focal_loss(
            predictions['driver'], target_driver,
            self.model_config.num_driver_status, activity_mask, seq_lengths
        )
        member_losses += self.weights['driver'] * driver_loss
        
        # 联合出行
        joint_loss = self._compute_member_focal_loss(
            predictions['joint'], target_joint,
            self.model_config.num_joint_status, activity_mask, seq_lengths
        )
        member_losses += self.weights['joint'] * joint_loss
        
        # mask无效成员
        member_losses = member_losses * member_mask.float()
        
        return member_losses
    
    def _compute_member_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        activity_mask: torch.BoolTensor,
        seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个成员的Focal Loss
        
        Args:
            logits: (batch, max_members, max_activities, num_classes)
            targets: (batch, max_members, max_activities)
            num_classes: 类别数
            activity_mask: (batch, max_members, max_activities)
            seq_lengths: (batch, max_members)
        
        Returns:
            (batch, max_members)
        """
        batch_size, max_members, max_activities = targets.shape
        
        # 展平计算focal loss
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        loss = self.focal_loss(logits_flat, targets_flat)
        loss = loss.view(batch_size, max_members, max_activities)
        
        # 对每个成员的序列取平均
        loss = (loss * activity_mask.float()).sum(dim=-1) / seq_lengths
        
        return loss


class MetricsCalculator:
    """计算评估指标"""
    
    @staticmethod
    def compute_accuracy(
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        activity_mask: torch.BoolTensor
    ) -> Dict[str, float]:
        """
        计算各属性的准确率
        """
        with torch.no_grad():
            valid_mask = activity_mask
            num_valid = valid_mask.sum().item()
            
            if num_valid == 0:
                return {k: 0.0 for k in ['purpose', 'mode', 'driver', 'joint']}
            
            # 解析目标
            target_purpose = targets[..., 2:12].argmax(dim=-1)
            target_mode = targets[..., 12:23].argmax(dim=-1)
            target_driver = targets[..., 23:25].argmax(dim=-1)
            target_joint = targets[..., 25:27].argmax(dim=-1)
            
            # 预测
            pred_purpose = predictions['purpose'].argmax(dim=-1)
            pred_mode = predictions['mode'].argmax(dim=-1)
            pred_driver = predictions['driver'].argmax(dim=-1)
            pred_joint = predictions['joint'].argmax(dim=-1)
            
            # 计算准确率
            acc = {}
            acc['purpose'] = ((pred_purpose == target_purpose) & valid_mask).sum().item() / num_valid
            acc['mode'] = ((pred_mode == target_mode) & valid_mask).sum().item() / num_valid
            acc['driver'] = ((pred_driver == target_driver) & valid_mask).sum().item() / num_valid
            acc['joint'] = ((pred_joint == target_joint) & valid_mask).sum().item() / num_valid
            
            return acc
    
    @staticmethod
    def compute_time_mae(
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        activity_mask: torch.BoolTensor
    ) -> Dict[str, float]:
        """
        计算时间的MAE (在z-score空间)
        """
        with torch.no_grad():
            valid_mask = activity_mask
            num_valid = valid_mask.sum().item()
            
            if num_valid == 0:
                return {'start_time': 0.0, 'end_time': 0.0}
            
            target_continuous = targets[..., :2]
            pred_continuous = predictions['continuous']
            
            # MAE
            mae_start = ((pred_continuous[..., 0] - target_continuous[..., 0]).abs() * valid_mask.float()).sum().item() / num_valid
            mae_end = ((pred_continuous[..., 1] - target_continuous[..., 1]).abs() * valid_mask.float()).sum().item() / num_valid
            
            return {'start_time': mae_start, 'end_time': mae_end}


def compute_rollout_loss(
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor,
        start_pos: int,
        rollout_length: int,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
) -> torch.Tensor:
    """
    计算 rollout 片段的损失

    Args:
        predictions: rollout 预测, 各项 shape 为 (B, max_members, rollout_length, *)
        targets: 完整目标 (B, max_members, max_activities, 27)
        member_mask: (B, max_members)
        activity_mask: (B, max_members, max_activities)
        start_pos: rollout 起始位置
        rollout_length: rollout 长度

    Returns:
        loss: scalar
    """
    end_pos = min(start_pos + rollout_length, targets.size(2))
    actual_length = end_pos - start_pos

    # 截取对应片段的目标
    target_slice = targets[:, :, start_pos:end_pos, :]
    mask_slice = activity_mask[:, :, start_pos:end_pos]
    valid_mask = member_mask.unsqueeze(-1) & mask_slice
    num_valid = valid_mask.sum().clamp(min=1)

    # 解析目标
    target_continuous = target_slice[..., :2]
    target_purpose = target_slice[..., 2:12].argmax(dim=-1)
    target_mode = target_slice[..., 12:23].argmax(dim=-1)
    target_driver = target_slice[..., 23:25].argmax(dim=-1)
    target_joint = target_slice[..., 25:27].argmax(dim=-1)

    # 截取预测（可能 rollout_length > actual_length）
    pred_continuous = predictions['continuous'][:, :, :actual_length]
    pred_purpose = predictions['purpose'][:, :, :actual_length]
    pred_mode = predictions['mode'][:, :, :actual_length]
    pred_driver = predictions['driver'][:, :, :actual_length]
    pred_joint = predictions['joint'][:, :, :actual_length]

    # 连续属性: Smooth L1
    continuous_loss = F.smooth_l1_loss(pred_continuous, target_continuous, reduction='none')
    continuous_loss = (continuous_loss.sum(dim=-1) * valid_mask.float()).sum() / num_valid

    # 离散属性: Focal Loss
    def focal_loss(logits, targets, num_classes):
        logits_flat = logits.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        p_t = F.softmax(logits_flat, dim=-1).gather(-1, targets_flat.unsqueeze(-1)).squeeze(-1)
        focal_weight = focal_alpha * (1 - p_t) ** focal_gamma
        return (focal_weight * ce).view_as(targets)

    purpose_loss = focal_loss(pred_purpose, target_purpose, pred_purpose.size(-1))
    purpose_loss = (purpose_loss * valid_mask.float()).sum() / num_valid

    mode_loss = focal_loss(pred_mode, target_mode, pred_mode.size(-1))
    mode_loss = (mode_loss * valid_mask.float()).sum() / num_valid

    driver_loss = focal_loss(pred_driver, target_driver, pred_driver.size(-1))
    driver_loss = (driver_loss * valid_mask.float()).sum() / num_valid

    joint_loss = focal_loss(pred_joint, target_joint, pred_joint.size(-1))
    joint_loss = (joint_loss * valid_mask.float()).sum() / num_valid

    # 加权求和
    total_loss = continuous_loss + purpose_loss + mode_loss + 0.5 * driver_loss + 0.5 * joint_loss

    return total_loss


# ==================== 新增：活动模式预测损失 ====================

class PatternPredictionLoss(nn.Module):
    """
    活动模式预测损失

    用 GMM 得到的目标分布监督模式预测专家
    """

    def __init__(
            self,
            family_weight: float = 1.0,
            individual_weight: float = 1.0
    ):
        super().__init__()
        self.family_weight = family_weight
        self.individual_weight = individual_weight

    def forward(
            self,
            pattern_outputs: Dict[str, torch.Tensor],
            family_pattern_target: torch.Tensor,
            individual_pattern_target: torch.Tensor,
            member_mask: torch.BoolTensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pattern_outputs: 模型预测
                - 'family_pattern_prob': (batch, num_family_patterns)
                - 'individual_pattern_prob': (batch, max_members, num_individual_patterns)
            family_pattern_target: (batch, num_family_patterns) GMM 目标分布
            individual_pattern_target: (batch, max_members, num_individual_patterns) GMM 目标分布
            member_mask: (batch, max_members) 有效成员掩码

        Returns:
            total_loss: scalar
            loss_dict: 各部分损失
        """
        losses = {}

        # 家庭模式损失: KL 散度
        family_pred = pattern_outputs['family_pattern_prob'].clamp(min=1e-8)
        family_loss = F.kl_div(
            family_pred.log(),
            family_pattern_target,
            reduction='batchmean'
        )
        losses['family_pattern'] = family_loss

        # 个人模式损失: KL 散度
        individual_pred = pattern_outputs['individual_pattern_prob'].clamp(min=1e-8)

        if member_mask is not None:
            num_valid = member_mask.sum().clamp(min=1)
            kl = F.kl_div(
                individual_pred.log(),
                individual_pattern_target,
                reduction='none'
            ).sum(dim=-1)  # (batch, max_members)
            individual_loss = (kl * member_mask.float()).sum() / num_valid
        else:
            individual_loss = F.kl_div(
                individual_pred.log(),
                individual_pattern_target,
                reduction='batchmean'
            )
        losses['individual_pattern'] = individual_loss

        # 总损失
        total_loss = self.family_weight * family_loss + self.individual_weight * individual_loss

        return total_loss, losses

# ==================== 新增结束 ====================