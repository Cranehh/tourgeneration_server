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


def destination_loss(
        pred_logits: torch.Tensor,
        target_zones: torch.Tensor,
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor,
        num_zones: int = 2006,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
) -> torch.Tensor:
    valid_mask = member_mask.unsqueeze(-1) & activity_mask
    dest_valid_mask = (target_zones < num_zones) & valid_mask

    if dest_valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

    pred_flat = pred_logits.view(-1, pred_logits.size(-1))
    target_flat = target_zones.view(-1)
    mask_flat = dest_valid_mask.view(-1)

    pred_valid = pred_flat[mask_flat]
    target_valid = target_flat[mask_flat]
    a = torch.isnan(pred_valid).any()
    b = torch.isinf(pred_valid).any()
    c = pred_valid.min()
    d = pred_valid.max()
    e = target_valid.min()
    f = target_valid.max()
    ce = F.cross_entropy(pred_valid, target_valid, reduction='none')
    pt = torch.exp(-ce)
    focal = focal_alpha * (1 - pt) ** focal_gamma * ce

    return focal.mean()
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

        # 添加不确定性权重
        self.use_uw = model_config.use_uncertainty_weight
        if self.use_uw:
            num_tasks = len(self.weights)
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
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

        # ===== 新增：目的地损失 =====
        if 'destination' in predictions:
            # 从targets提取目标目的地（最后一维）
            target_destinations = targets[..., -1].long()

            dest_loss = destination_loss(
                predictions['destination'],
                target_destinations,
                member_mask,
                activity_mask,
                num_zones=2006,
                focal_alpha= self.train_config.focal_alpha,
                focal_gamma= self.train_config.focal_gamma
            )
            losses['destination'] = dest_loss
            total_loss = total_loss + self.weights.get('destination', 1) * dest_loss

        if self.use_uw:
            total_loss = 0
            for i, (name, loss) in enumerate(losses.items()):
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * loss + 0.5 * self.log_vars[i]
        else:
            total_loss = sum(self.weights[k] * v for k, v in losses.items())

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


    # ===== 新增：目的地损失 =====
    if 'destination' in predictions:
        target_dest = targets[:, :, start_pos:end_pos, -1].long()
        pred_dest = predictions['destination'][:, :, :actual_length]

        dest_valid = (target_dest < 2006) & valid_mask
        if dest_valid.sum() > 0:
            dest_loss = F.cross_entropy(
                pred_dest[dest_valid].view(-1, pred_dest.size(-1)),
                target_dest[dest_valid].view(-1),
                reduction='mean'
            )
            total_loss = total_loss + dest_loss

    return total_loss


# ==================== 新增：活动模式预测损失 ====================

# ==================== 新增：活动模式预测损失 ====================

class PatternPredictionLoss(nn.Module):
    """使用Focal Loss的模式预测损失"""

    def __init__(
            self,
            family_weight: float = 1.0,
            individual_weight: float = 1.0,
            focal_gamma: float = 2.0,  # 聚焦参数
            label_smoothing: float = 0.05  # 轻微平滑
    ):
        super().__init__()
        self.family_weight = family_weight
        self.individual_weight = individual_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

    def _smooth_target(self, target: torch.Tensor) -> torch.Tensor:
        n_classes = target.size(-1)
        return (1 - self.label_smoothing) * target + self.label_smoothing / n_classes

    def _focal_soft_ce(
            self,
            logits: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal Loss for soft labels

        FL = -α * (1 - p_t)^γ * log(p_t)
        对于软标签：FL = -Σ target_i * (1 - pred_i)^γ * log(pred_i)
        """
        log_pred = F.log_softmax(logits, dim=-1)
        pred = torch.exp(log_pred)

        # Focal weight: (1 - p)^gamma，让模型更关注预测错误的样本
        focal_weight = (1 - pred) ** self.focal_gamma

        # 软交叉熵 + focal weight
        loss = -(target * focal_weight * log_pred).sum(dim=-1)

        return loss

    def forward(
            self,
            pattern_outputs: Dict[str, torch.Tensor],
            family_pattern_target: torch.Tensor,
            individual_pattern_target: torch.Tensor,
            member_mask: torch.BoolTensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        losses = {}

        # 平滑目标
        smoothed_family = self._smooth_target(family_pattern_target)
        smoothed_individual = self._smooth_target(individual_pattern_target)

        # 家庭模式 Focal Loss
        family_logits = pattern_outputs['family_pattern_logits']
        family_loss = self._focal_soft_ce(family_logits, smoothed_family).mean()
        losses['family_pattern'] = family_loss

        # 个人模式 Focal Loss
        individual_logits = pattern_outputs['individual_pattern_logits']
        individual_ce = self._focal_soft_ce(individual_logits, smoothed_individual)

        if member_mask is not None:
            num_valid = member_mask.sum().clamp(min=1)
            individual_loss = (individual_ce * member_mask.float()).sum() / num_valid
        else:
            individual_loss = individual_ce.mean()
        losses['individual_pattern'] = individual_loss

        total_loss = self.family_weight * family_loss + self.individual_weight * individual_loss

        return total_loss, losses

# ==================== 新增结束 ====================
# ==================== CAGrad 梯度调整 ====================

import numpy as np
from scipy.optimize import minimize


def cagrad_backward(
        model: torch.nn.Module,
        loss: torch.Tensor,
        losses: Dict[str, torch.Tensor],
        c: float = 0.5
) -> None:
    """
    使用 CAGrad 进行反向传播（替换原来的 loss.backward()）

    Args:
        model: 模型
        loss: 总损失（用于 decoder 的普通梯度）
        losses: 各任务损失字典
        c: CAGrad 约束半径 (推荐 0.4-0.6)

    使用示例:
        loss, losses = criterion(predictions, targets, ...)
        # loss.backward()  ← 删掉这行
        cagrad_backward(model, loss, losses, c=0.5)  # ← 用这个替换
        optimizer.step()
    """
    # 获取 encoder 参数
    # if hasattr(model, 'encoder'):
    #     shared_params = [p for p in model.encoder.parameters() if p.requires_grad]
    # else:
    shared_params = [p for p in model.parameters() if p.requires_grad]

    # 计算每个任务对 encoder 的梯度
    task_grads = []
    task_names = list(losses.keys())
    num_tasks = len(task_names)

    for i, (name, task_loss) in enumerate(losses.items()):
        grads = torch.autograd.grad(
            task_loss,
            shared_params,
            retain_graph=True,  # 保留计算图
            allow_unused=True
        )
        grad_vec = torch.cat([
            g.view(-1) if g is not None else torch.zeros(p.numel(), device=task_loss.device)
            for g, p in zip(grads, shared_params)
        ])
        task_grads.append(grad_vec)

    # 堆叠为矩阵 (num_tasks, grad_dim)
    grads_matrix = torch.stack(task_grads, dim=0)

    # 应用 CAGrad 算法
    cagrad_vec = _cagrad_core(grads_matrix, c)

    # 写回 encoder 梯度
    offset = 0
    for p in shared_params:
        numel = p.numel()
        p.grad = cagrad_vec[offset:offset + numel].view_as(p).clone()
        offset += numel

    # 对 decoder 用普通 backward（最后一次，不需要 retain_graph）
    if hasattr(model, 'decoder'):
        decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
        decoder_grads = torch.autograd.grad(
            loss,
            decoder_params,
            allow_unused=True
        )
        for p, g in zip(decoder_params, decoder_grads):
            if g is not None:
                p.grad = g


def _cagrad_core(grads: torch.Tensor, c: float = 0.5) -> torch.Tensor:
    """CAGrad 核心算法"""
    num_tasks = grads.shape[0]
    g0 = grads.mean(dim=0)
    g0_norm = g0.norm()

    if g0_norm < 1e-8:
        return g0

    GG = grads @ grads.T
    x_init = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for _ in range(num_tasks))
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    A = GG.detach().cpu().numpy()
    b = (grads @ g0).detach().cpu().numpy()

    def obj(x):
        return x @ A @ x - 2 * x @ b

    result = minimize(obj, x_init, method='SLSQP', bounds=bnds,
                      constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})

    w = torch.tensor(result.x, dtype=grads.dtype, device=grads.device)
    g_w = w @ grads

    diff = g_w - g0
    diff_norm = diff.norm()

    if diff_norm > 1e-8:
        cagrad_vec = g0 + c * g0_norm * diff / diff_norm
    else:
        cagrad_vec = g0

    # 缩放到原始范数
    cagrad_vec = cagrad_vec * g0_norm / (cagrad_vec.norm() + 1e-8)

    return cagrad_vec


def cagrad_backward_amp(
        model: torch.nn.Module,
        loss: torch.Tensor,
        losses: Dict[str, torch.Tensor],
        scaler: torch.cuda.amp.GradScaler,
        c: float = 0.5
) -> None:
    """
    AMP 混合精度版本的 CAGrad 反向传播
    """
    # 获取 encoder 参数
    # if hasattr(model, 'encoder'):
    #     shared_params = [p for p in model.encoder.parameters() if p.requires_grad]
    # else:
    shared_params = [p for p in model.parameters() if p.requires_grad]

    # 计算每个任务对 encoder 的梯度
    task_grads = []

    for i, (name, task_loss) in enumerate(losses.items()):
        # 对每个任务损失进行 scale
        scaled_loss = scaler.scale(task_loss)
        grads = torch.autograd.grad(
            scaled_loss,
            shared_params,
            retain_graph=True,
            allow_unused=True
        )
        grad_vec = torch.cat([
            g.view(-1) if g is not None else torch.zeros(p.numel(), device=task_loss.device)
            for g, p in zip(grads, shared_params)
        ])
        task_grads.append(grad_vec)

    # 堆叠为矩阵
    grads_matrix = torch.stack(task_grads, dim=0)

    # 应用 CAGrad 算法
    cagrad_vec = _cagrad_core(grads_matrix, c)

    # 写回 encoder 梯度
    offset = 0
    for p in shared_params:
        numel = p.numel()
        p.grad = cagrad_vec[offset:offset + numel].view_as(p).clone()
        offset += numel

    # 对 decoder 用普通 backward
    if hasattr(model, 'decoder'):
        decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
        scaled_loss = scaler.scale(loss)
        decoder_grads = torch.autograd.grad(
            scaled_loss,
            decoder_params,
            allow_unused=True
        )
        for p, g in zip(decoder_params, decoder_grads):
            if g is not None:
                p.grad = g