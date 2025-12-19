"""
Exposure Bias 处理策略

包含:
1. Scheduled Sampling: 训练时逐渐从 teacher forcing 过渡到自回归
2. Data Noising: 给输入添加噪声增强鲁棒性
3. Mixed Training: 混合使用真实标签和模型预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable
import numpy as np
import math

from config import ModelConfig
from mtan_decoder import autoregressive_rollout
from losses import compute_rollout_loss


class ScheduledSamplingScheduler:
    """
    Scheduled Sampling 调度器
    
    控制训练过程中使用 teacher forcing 的概率
    随着训练进行，逐渐降低使用真实标签的概率，增加使用模型预测的概率
    """
    
    def __init__(
        self,
        schedule_type: str = 'linear',
        initial_prob: float = 1.0,
        final_prob: float = 0.1,
        decay_steps: int = 10000,
        warmup_steps: int = 10
    ):
        """
        Args:
            schedule_type: 调度类型 ('linear', 'exponential', 'sigmoid', 'cosine')
            initial_prob: 初始使用真实标签的概率
            final_prob: 最终使用真实标签的概率
            decay_steps: 衰减步数
            warmup_steps: 预热步数 (在此期间保持 initial_prob)
        """
        self.schedule_type = schedule_type
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def get_teacher_forcing_prob(self) -> float:
        """获取当前步的 teacher forcing 概率"""
        if self.current_step < self.warmup_steps:
            return self.initial_prob
        
        # 计算衰减进度
        progress = min(
            (self.current_step - self.warmup_steps) / self.decay_steps, 
            1.0
        )
        
        if self.schedule_type == 'linear':
            prob = self.initial_prob - progress * (self.initial_prob - self.final_prob)
        
        elif self.schedule_type == 'exponential':
            # 指数衰减: prob = initial * decay^progress
            decay_rate = self.final_prob / self.initial_prob
            prob = self.initial_prob * (decay_rate ** progress)
        
        elif self.schedule_type == 'sigmoid':
            # Sigmoid 衰减: 中间快两端慢
            k = 10  # 控制曲线陡峭程度
            sigmoid_progress = 1 / (1 + math.exp(-k * (progress - 0.5)))
            prob = self.initial_prob - sigmoid_progress * (self.initial_prob - self.final_prob)
        
        elif self.schedule_type == 'cosine':
            # 余弦衰减: 平滑过渡
            prob = self.final_prob + 0.5 * (self.initial_prob - self.final_prob) * \
                   (1 + math.cos(math.pi * progress))
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return max(prob, self.final_prob)
    
    def step(self):
        """更新步数"""
        self.current_step += 1
    
    def reset(self):
        """重置"""
        self.current_step = 0
    
    def state_dict(self) -> dict:
        """保存状态"""
        return {'current_step': self.current_step}
    
    def load_state_dict(self, state_dict: dict):
        """加载状态"""
        self.current_step = state_dict['current_step']


class InputNoising:
    """
    输入噪声模块
    
    给输入添加噪声，使模型对输入扰动更鲁棒
    模拟推理时误差累积的情况
    """
    
    def __init__(
        self,
        continuous_noise_std: float = 0.1,
        discrete_flip_prob: float = 0.1,
        noise_schedule: str = 'constant'
    ):
        """
        Args:
            continuous_noise_std: 连续属性的高斯噪声标准差
            discrete_flip_prob: 离散属性的翻转概率
            noise_schedule: 噪声调度 ('constant', 'curriculum')
        """
        self.continuous_noise_std = continuous_noise_std
        self.discrete_flip_prob = discrete_flip_prob
        self.noise_schedule = noise_schedule
        self.current_step = 0
    
    def add_noise(
        self,
        activities: torch.Tensor,
        config: ModelConfig,
        training_progress: float = 0.0
    ) -> torch.Tensor:
        """
        给活动添加噪声
        
        Args:
            activities: (batch, max_members, max_activities, 27) 活动数据
            config: 模型配置
            training_progress: 训练进度 [0, 1]，用于 curriculum 调度
        
        Returns:
            noised_activities: 添加噪声后的活动
        """
        noised = activities.clone()
        
        # 根据训练进度调整噪声强度
        if self.noise_schedule == 'curriculum':
            # 课程学习: 训练初期少噪声，后期多噪声
            noise_scale = min(training_progress * 2, 1.0)
        else:
            noise_scale = 1.0
        
        # 连续属性加高斯噪声
        continuous_noise = torch.randn_like(noised[..., :2]) * \
                          self.continuous_noise_std * noise_scale
        noised[..., :2] = noised[..., :2] + continuous_noise
        
        # 离散属性随机翻转
        flip_prob = self.discrete_flip_prob * noise_scale
        
        # 目的 (2:12)
        noised[..., 2:12] = self._flip_onehot(
            noised[..., 2:12], config.num_purposes, flip_prob
        )
        
        # 方式 (12:23)
        noised[..., 12:23] = self._flip_onehot(
            noised[..., 12:23], config.num_modes, flip_prob
        )
        
        # 驾驶状态 (23:25)
        noised[..., 23:25] = self._flip_onehot(
            noised[..., 23:25], config.num_driver_status, flip_prob
        )
        
        # 联合出行 (25:27)
        noised[..., 25:27] = self._flip_onehot(
            noised[..., 25:27], config.num_joint_status, flip_prob
        )
        
        return noised
    
    def _flip_onehot(
        self,
        onehot: torch.Tensor,
        num_classes: int,
        flip_prob: float
    ) -> torch.Tensor:
        """
        随机翻转 one-hot 编码
        
        Args:
            onehot: (..., num_classes) one-hot 张量
            num_classes: 类别数
            flip_prob: 翻转概率
        """
        if flip_prob <= 0:
            return onehot
        
        # 决定哪些位置需要翻转
        flip_mask = torch.rand(onehot.shape[:-1], device=onehot.device) < flip_prob
        
        # 生成新的随机类别
        new_indices = torch.randint(0, num_classes, onehot.shape[:-1], device=onehot.device)
        new_onehot = F.one_hot(new_indices, num_classes).float()
        
        # 应用翻转
        flip_mask = flip_mask.unsqueeze(-1).expand_as(onehot)
        result = torch.where(flip_mask, new_onehot, onehot)
        
        return result


class ScheduledSamplingDecoder(nn.Module):
    """
    支持 Scheduled Sampling 的 Decoder 包装器
    
    在训练时根据调度概率决定使用真实标签还是模型预测作为下一步输入
    """
    
    def __init__(
        self,
        base_decoder: nn.Module,
        config: ModelConfig,
        scheduler: ScheduledSamplingScheduler,
        input_noising: Optional[InputNoising] = None
    ):
        """
        Args:
            base_decoder: 基础 decoder (MTANDecoder)
            config: 模型配置
            scheduler: Scheduled Sampling 调度器
            input_noising: 输入噪声模块 (可选)
        """
        super().__init__()
        self.decoder = base_decoder
        self.config = config
        self.scheduler = scheduler
        self.input_noising = input_noising
    
    def forward(
        self,
        member_repr: torch.Tensor,
        family_repr: torch.Tensor,
        target_activities: torch.Tensor,
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor,
        training: bool = True,
        pattern_outputs: Dict[str, torch.Tensor] = None  # 新增
    ) -> Dict[str, torch.Tensor]:
        """
        带 Scheduled Sampling 的前向传播
        
        Args:
            member_repr: (batch, max_members, d_model)
            family_repr: (batch, d_model)
            target_activities: (batch, max_members, max_activities, 27)
            member_mask: (batch, max_members)
            activity_mask: (batch, max_members, max_activities)
            training: 是否训练模式
        
        Returns:
            predictions: 预测结果
        """
        if not training:
            # 推理模式: 纯自回归
            return self.decoder.generate(member_repr, family_repr, member_mask,
            pattern_outputs=pattern_outputs
            )
        
        # 获取当前 teacher forcing 概率
        tf_prob = self.scheduler.get_teacher_forcing_prob()
        
        # 如果概率为 1，使用纯 teacher forcing
        if tf_prob >= 1.0:
            activities_input = target_activities
            if self.input_noising is not None:
                activities_input = self.input_noising.add_noise(
                    activities_input, self.config
                )
            return self.decoder(
                member_repr, family_repr, activities_input,
                member_mask, activity_mask,
                pattern_outputs=pattern_outputs
            )
        
        # 混合模式: 逐步决定使用真实标签还是预测
        return self._forward_scheduled_sampling(
            member_repr, family_repr, target_activities,
            member_mask, activity_mask, tf_prob,
            pattern_outputs=pattern_outputs
        )
    
    def _forward_scheduled_sampling(
        self,
        member_repr: torch.Tensor,
        family_repr: torch.Tensor,
        target_activities: torch.Tensor,
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor,
        tf_prob: float,
        pattern_outputs: Dict[str, torch.Tensor] = None  # 新增
    ) -> Dict[str, torch.Tensor]:
        """
        Scheduled Sampling 前向传播
        
        对每个时间步，以 tf_prob 概率使用真实标签，否则使用模型预测
        """
        batch_size = member_repr.size(0)
        max_members = self.config.max_members
        max_activities = self.config.max_activities
        device = member_repr.device
        
        # 初始化
        all_predictions = {
            'continuous': [],
            'purpose': [],
            'mode': [],
            'driver': [],
            'joint': []
        }
        
        # 准备任务特定注意力和 Cross-Role 注意力
        member_states = member_repr.clone()
        member_states = self.decoder.task_attention(member_states, family_repr)
        member_states = self.decoder.cross_role_attention(member_states, member_mask)
        
        # 当前输入序列 (从 start token 开始)
        current_input = self.decoder.start_token.expand(batch_size, max_members, 1, -1)
        
        # 上一个活动时间 (用于时间约束)
        prev_continuous = torch.zeros(batch_size, max_members, 2, device=device)
        
        for t in range(max_activities):
            # 位置编码
            seq_len = current_input.size(2)
            pos_input = self.decoder.pos_encoding(
                current_input.view(batch_size * max_members, seq_len, -1)
            )
            pos_input = pos_input.view(batch_size, max_members, seq_len, -1)
            
            # 加入成员上下文
            member_context = member_states.unsqueeze(2).expand(-1, -1, seq_len, -1)
            decoder_input = pos_input + member_context
            
            # Memory
            memory = member_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)
            
            # 重塑并解码
            decoder_input_flat = decoder_input.view(batch_size * max_members, seq_len, -1)
            memory_flat = memory.view(batch_size * max_members, seq_len, -1)
            
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            
            decoded = self.decoder.transformer_decoder(
                tgt=decoder_input_flat,
                memory=memory_flat,
                tgt_mask=causal_mask
            )

            # 新增: 对 decoder 输出进行模式调制
            if self.decoder.use_pattern_condition and pattern_outputs is not None:
                # 恢复形状以便调制
                decoded_reshaped = decoded.view(batch_size, max_members, seq_len, -1)
                decoded_reshaped = self.decoder.pattern_condition(
                    decoded_reshaped,
                    pattern_outputs['family_pattern_prob'],
                    pattern_outputs['individual_pattern_prob']
                )
                decoded = decoded_reshaped.view(batch_size * max_members, seq_len, -1)
            
            # 取最后位置输出
            last_hidden = decoded[:, -1, :].view(batch_size, max_members, -1)
            
            # 预测
            step_pred = self.decoder.output_heads(last_hidden,
                                                  pattern_outputs['family_pattern_prob'],
                                                  pattern_outputs['individual_pattern_prob']
                                                  )
            
            # 时间约束
            is_first = (t == 0)
            is_first_tensor = torch.full(
                (batch_size, max_members), is_first, device=device, dtype=torch.bool
            )
            constrained_continuous = self.decoder.time_constraint.apply_constraint(
                step_pred['continuous'], prev_continuous, is_first_tensor
            )
            step_pred['continuous'] = constrained_continuous
            
            # 保存预测
            all_predictions['continuous'].append(constrained_continuous)
            all_predictions['purpose'].append(step_pred['purpose'])
            all_predictions['mode'].append(step_pred['mode'])
            all_predictions['driver'].append(step_pred['driver'])
            all_predictions['joint'].append(step_pred['joint'])
            
            # 决定下一步输入: teacher forcing 还是用预测
            if t < max_activities - 1:
                use_teacher = torch.rand(batch_size, max_members, device=device) < tf_prob
                use_teacher = use_teacher.unsqueeze(-1)  # (batch, max_members, 1)
                
                # 真实标签
                real_activity = target_activities[:, :, t, :]
                real_emb = self.decoder.activity_embedding(real_activity)
                
                # 模型预测
                pred_continuous = constrained_continuous
                pred_purpose_idx = step_pred['purpose'].argmax(dim=-1)
                pred_mode_idx = step_pred['mode'].argmax(dim=-1)
                pred_driver_idx = step_pred['driver'].argmax(dim=-1)
                pred_joint_idx = step_pred['joint'].argmax(dim=-1)
                
                pred_emb = self.decoder.activity_embedding.embed_from_indices(
                    pred_continuous, pred_purpose_idx, pred_mode_idx,
                    pred_driver_idx, pred_joint_idx
                )
                
                # 混合
                next_emb = torch.where(use_teacher, real_emb, pred_emb)
                
                # 可选: 添加噪声
                if self.input_noising is not None and torch.rand(1).item() < 0.5:
                    # 构造活动张量用于加噪
                    noise_activity = self._construct_activity_tensor(
                        pred_continuous, pred_purpose_idx, pred_mode_idx,
                        pred_driver_idx, pred_joint_idx
                    )
                    noised_activity = self.input_noising.add_noise(
                        noise_activity.unsqueeze(2), self.config
                    ).squeeze(2)
                    noised_emb = self.decoder.activity_embedding(noised_activity)
                    
                    # 只对使用预测的部分加噪声
                    next_emb = torch.where(use_teacher, real_emb, noised_emb)
                
                # 更新输入序列
                current_input = torch.cat([
                    current_input,
                    next_emb.unsqueeze(2)
                ], dim=2)
                
                # 更新 prev_continuous
                prev_continuous = torch.where(
                    use_teacher.expand_as(pred_continuous),
                    real_activity[..., :2],
                    pred_continuous
                )
        
        # 堆叠结果
        return {
            'continuous': torch.stack(all_predictions['continuous'], dim=2),
            'purpose': torch.stack(all_predictions['purpose'], dim=2),
            'mode': torch.stack(all_predictions['mode'], dim=2),
            'driver': torch.stack(all_predictions['driver'], dim=2),
            'joint': torch.stack(all_predictions['joint'], dim=2)
        }
    
    def _construct_activity_tensor(
        self,
        continuous: torch.Tensor,
        purpose_idx: torch.Tensor,
        mode_idx: torch.Tensor,
        driver_idx: torch.Tensor,
        joint_idx: torch.Tensor
    ) -> torch.Tensor:
        """从预测构造活动张量"""
        batch_size, max_members = continuous.shape[:2]
        device = continuous.device
        
        activity = torch.zeros(
            batch_size, max_members, self.config.activity_dim,
            device=device
        )
        
        # 连续属性
        activity[..., :2] = continuous
        
        # One-hot 编码
        activity[..., 2:12] = F.one_hot(purpose_idx, self.config.num_purposes).float()
        activity[..., 12:23] = F.one_hot(mode_idx, self.config.num_modes).float()
        activity[..., 23:25] = F.one_hot(driver_idx, self.config.num_driver_status).float()
        activity[..., 25:27] = F.one_hot(joint_idx, self.config.num_joint_status).float()
        
        return activity


class ExposureBiasTrainer:
    """
    处理 Exposure Bias 的训练器封装
    
    集成 Scheduled Sampling 和 Input Noising
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        ss_schedule_type: str = 'linear',
        ss_initial_prob: float = 1.0,
        ss_final_prob: float = 0.2,
        ss_decay_steps: int = 10000,
        ss_warmup_steps: int = 2000,
        use_input_noising: bool = True,
        noise_std: float = 0.1,
        noise_flip_prob: float = 0.1
    ):
        """
        Args:
            model: 模型
            config: 配置
            ss_*: Scheduled Sampling 参数
            use_input_noising: 是否使用输入噪声
            noise_*: 噪声参数
        """
        self.model = model
        self.config = config
        
        # Scheduled Sampling 调度器
        self.ss_scheduler = ScheduledSamplingScheduler(
            schedule_type=ss_schedule_type,
            initial_prob=ss_initial_prob,
            final_prob=ss_final_prob,
            decay_steps=ss_decay_steps,
            warmup_steps=ss_warmup_steps
        )
        
        # 输入噪声
        self.input_noising = InputNoising(
            continuous_noise_std=noise_std,
            discrete_flip_prob=noise_flip_prob,
            noise_schedule='curriculum'
        ) if use_input_noising else None
        
        # 包装 decoder
        if hasattr(model, 'decoder'):
            self.ss_decoder = ScheduledSamplingDecoder(
                base_decoder=model.decoder,
                config=config,
                scheduler=self.ss_scheduler,
                input_noising=self.input_noising
            )
    
    def train_step(
        self,
        batch,
        criterion,
        optimizer,
        current_epoch,
        scaler=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        单步训练
        
        Returns:
            loss: 损失
            losses: 各部分损失
        """
        # 编码
        member_repr, family_repr, pattern_prob = self.model.encoder(
            batch.family_attr, batch.member_attr, batch.member_mask
        )
        
        # 解码 (使用 Scheduled Sampling)
        predictions = self.ss_decoder(
            member_repr, family_repr, batch.activities,
            batch.member_mask, batch.activity_mask,
            training=True,
            pattern_outputs=pattern_prob  # 传递模式概率
        )
        pattern_prob.update({
            'family_pattern_target': batch.family_pattern,
            'individual_pattern_target': batch.member_pattern,
        })
        # 计算损失
        loss, losses = criterion(
            predictions, batch.activities,
            batch.member_mask, batch.activity_mask, pattern_prob
        )

        ar_stage = {1:{'prob': 0.3, 'length':2, 'ar_ratio':0.3},
                    2:{'prob': 0.5, 'length':3, 'ar_ratio':0.4},
                    3:{'prob': 0.8, 'length':3, 'ar_ratio':0.5}}
        if current_epoch < 10:
            stage = 1
        elif current_epoch < 50:
            stage = 2
        else:
            stage = 3
        # 设置 rollout 参数
        rollout_prob = ar_stage[stage]['prob']
        rollout_length = ar_stage[stage]['length']
        ar_ratio = ar_stage[stage]['ar_ratio']
        
        # 添加 rollout 损失 (epoch > 5 后开始)
        ar_loss = torch.tensor(0.0, device=member_repr.device)
        if torch.rand(1).item() < rollout_prob:  # 概率做 rollout

            start_pos = torch.randint(0, max(1, batch.activities.size(2) - rollout_length), (1,)).item()
            rollout_preds = autoregressive_rollout(
                self.model.decoder, member_repr, family_repr,
                batch.activities, batch.member_mask,
                start_pos=start_pos, rollout_length=rollout_length,
                pattern_probs=pattern_prob
            )
            ar_loss = compute_rollout_loss(
                rollout_preds, batch.activities,
                batch.member_mask, batch.activity_mask,
                start_pos=start_pos, rollout_length=rollout_length
            )
        # 更新调度器
        self.ss_scheduler.step()
        loss = loss + ar_ratio * ar_loss
        return loss, losses
    
    def get_current_tf_prob(self) -> float:
        """获取当前 teacher forcing 概率"""
        return self.ss_scheduler.get_teacher_forcing_prob()
    
    def state_dict(self) -> dict:
        """保存状态"""
        return {
            'ss_scheduler': self.ss_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict: dict):
        """加载状态"""
        self.ss_scheduler.load_state_dict(state_dict['ss_scheduler'])


# 便捷函数
def create_exposure_bias_handler(
    model: nn.Module,
    config: ModelConfig,
    strategy: str = 'scheduled_sampling',
    **kwargs
) -> ExposureBiasTrainer:
    """
    创建 Exposure Bias 处理器
    
    Args:
        model: 模型
        config: 配置
        strategy: 策略 ('scheduled_sampling', 'aggressive', 'conservative')
        **kwargs: 额外参数
    
    Returns:
        ExposureBiasTrainer 实例
    """
    if strategy == 'scheduled_sampling':
        # 默认配置
        return ExposureBiasTrainer(
            model, config,
            ss_schedule_type='linear',
            ss_initial_prob=1.0,
            ss_final_prob=0.2,
            ss_decay_steps=60000,
            ss_warmup_steps=5000,
            use_input_noising=True,
            **kwargs
        )
    
    elif strategy == 'aggressive':
        # 激进配置: 快速降低 teacher forcing
        return ExposureBiasTrainer(
            model, config,
            ss_schedule_type='exponential',
            ss_initial_prob=1.0,
            ss_final_prob=0.05,
            ss_decay_steps=10000,
            ss_warmup_steps=300,
            use_input_noising=True,
            noise_std=0.15,
            noise_flip_prob=0.15,
            **kwargs
        )
    
    elif strategy == 'conservative':
        # 保守配置: 缓慢降低 teacher forcing
        return ExposureBiasTrainer(
            model, config,
            ss_schedule_type='cosine',
            ss_initial_prob=1.0,
            ss_final_prob=0.3,
            ss_decay_steps=20000,
            ss_warmup_steps=5000,
            use_input_noising=True,
            noise_std=0.05,
            noise_flip_prob=0.05,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
