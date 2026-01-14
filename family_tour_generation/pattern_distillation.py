"""
活动模式蒸馏模型

核心思路：
1. 教师模型：从完整活动序列识别模式（能看到"答案"）
2. 学生模型：从人口属性预测模式（只能看到输入特征）
3. 蒸馏：让学生模仿教师的表征和输出

使用方法：
1. 先训练教师模型（使用真实活动序列）
2. 冻结教师，蒸馏训练学生
3. 将学生模型集成到主生成模型中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import math


# ==================== 配置 ====================

@dataclass
class PatternDistillConfig:
    """蒸馏模型配置"""
    # 活动序列特征
    activity_dim: int = 27          # 每个活动的特征维度
    max_activities: int = 6         # 最大活动数
    max_members: int = 8            # 最大家庭成员数
    
    # 人口属性特征
    family_dim: int = 10            # 家庭属性维度
    member_dim: int = 52            # 成员属性维度（不含最后两列）
    
    # 模式数量
    num_family_patterns: int = 20   # 家庭模式数
    num_person_patterns: int = 40   # 个人模式数
    
    # 模型结构
    d_model: int = 256              # 隐藏层维度
    d_emb: int = 128                # embedding维度
    num_heads: int = 8              # 注意力头数
    num_encoder_layers: int = 3     # 教师编码器层数
    dropout: float = 0.1
    
    # 蒸馏参数
    temperature: float = 3.0        # 蒸馏温度
    alpha_soft: float = 0.4         # 软标签损失权重
    alpha_repr: float = 0.5         # 表征蒸馏权重
    alpha_hard: float = 0.1         # 硬标签损失权重
    
    # 训练参数
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 128
    num_epochs: int = 100
    device: str = 'cuda'


# ==================== 教师模型：从活动序列识别模式 ====================

class ActivityEmbedding(nn.Module):
    """活动特征嵌入"""
    
    def __init__(self, config: PatternDistillConfig):
        super().__init__()
        self.config = config
        
        # 活动各部分的维度
        # [时间2, 目的10, 方式11, 驾驶2, 联合2] = 27
        self.time_dim = 2
        self.purpose_dim = 10
        self.mode_dim = 11
        self.driver_dim = 2
        self.joint_dim = 2
        
        # 各特征的embedding
        self.time_proj = nn.Linear(self.time_dim, config.d_emb // 4)
        self.purpose_emb = nn.Linear(self.purpose_dim, config.d_emb // 4)
        self.mode_emb = nn.Linear(self.mode_dim, config.d_emb // 4)
        self.driver_emb = nn.Linear(self.driver_dim, config.d_emb // 8)
        self.joint_emb = nn.Linear(self.joint_dim, config.d_emb // 8)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(config.d_emb, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_activities, config.d_model) * 0.02
        )
        
    def forward(self, activities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activities: [B, T, 27] 活动序列
            
        Returns:
            [B, T, d_model] 活动嵌入
        """
        # 分解各部分
        time_feat = activities[..., :2]
        purpose_feat = activities[..., 2:12]
        mode_feat = activities[..., 12:23]
        driver_feat = activities[..., 23:25]
        joint_feat = activities[..., 25:27]
        
        # 各部分embedding
        time_emb = self.time_proj(time_feat)
        purpose_emb = self.purpose_emb(purpose_feat)
        mode_emb = self.mode_emb(mode_feat)
        driver_emb = self.driver_emb(driver_feat)
        joint_emb = self.joint_emb(joint_feat)
        
        # 拼接
        combined = torch.cat([time_emb, purpose_emb, mode_emb, driver_emb, joint_emb], dim=-1)
        
        # 融合
        output = self.fusion(combined)
        
        # 加位置编码
        output = output + self.pos_encoding[:, :output.size(1), :]
        
        return output


class PatternTeacher(nn.Module):
    """
    教师模型：从活动序列识别模式
    
    输入：活动序列 [B, M, T, 27]
    输出：
        - person_repr: 个人模式表征 [B, M, d_model]
        - family_repr: 家庭模式表征 [B, d_model]
        - person_logits: 个人模式分类 [B, M, num_person_patterns]
        - family_logits: 家庭模式分类 [B, num_family_patterns]
    """
    
    def __init__(self, config: PatternDistillConfig):
        super().__init__()
        self.config = config
        
        # 活动embedding
        self.activity_embedding = ActivityEmbedding(config)
        
        # 个人活动序列编码器（Transformer）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.person_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        self.person_encoder.enable_nested_tensor = False
        
        # 个人表征聚合（加权池化）
        self.person_pool = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # 家庭成员间交互编码器
        family_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.family_encoder = nn.TransformerEncoder(
            family_encoder_layer,
            num_layers=2
        )
        self.family_encoder.enable_nested_tensor = False
        
        # 家庭表征聚合
        self.family_pool = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类头
        self.person_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_person_patterns)
        )
        
        self.family_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_family_patterns)
        )
        
    def forward(
        self,
        activities: torch.Tensor,
        member_mask: torch.Tensor,
        activity_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            activities: [B, M, T, 27] 活动序列
            member_mask: [B, M] 有效成员mask
            activity_mask: [B, M, T] 有效活动mask
            
        Returns:
            Dict包含:
                - person_repr: [B, M, d_model]
                - family_repr: [B, d_model]
                - person_logits: [B, M, num_person_patterns]
                - family_logits: [B, num_family_patterns]
        """
        B, M, T, _ = activities.shape
        
        # 编码每个成员的活动序列
        person_reprs = []

        for m in range(M):
            member_acts = activities[:, m, :, :]
            act_mask = activity_mask[:, m, :]  # [B, T]

            # 检测哪些样本的该成员是"空的"（无任何活动）
            valid_samples = act_mask.any(dim=1)  # [B]，True表示至少有一个活动

            if valid_samples.all():
                # 正常路径：所有样本的该成员都有活动
                act_emb = self.activity_embedding(member_acts)
                padding_mask = ~act_mask
                encoded = self.person_encoder(act_emb, src_key_padding_mask=padding_mask)

                weights = self.person_pool(encoded)
                weights = weights.masked_fill(~act_mask.unsqueeze(-1), -1e9)
                weights = F.softmax(weights, dim=1)
                person_repr = (encoded * weights).sum(dim=1)
            else:
                # 混合路径：部分样本的该成员是空的
                B = member_acts.size(0)
                person_repr = torch.zeros(B, self.config.d_model, device=member_acts.device)

                if valid_samples.any():
                    # 只处理有效的样本
                    valid_acts = member_acts[valid_samples]
                    valid_act_mask = act_mask[valid_samples]

                    act_emb = self.activity_embedding(valid_acts)
                    padding_mask = ~valid_act_mask
                    encoded = self.person_encoder(act_emb, src_key_padding_mask=padding_mask)

                    weights = self.person_pool(encoded)
                    weights = weights.masked_fill(~valid_act_mask.unsqueeze(-1), -1e9)
                    weights = F.softmax(weights, dim=1)
                    valid_repr = (encoded * weights).sum(dim=1)

                    # 放回原位置
                    person_repr[valid_samples] = valid_repr

                # 无效样本的person_repr保持为零向量（或可用其他默认值）

            person_reprs.append(person_repr)
        
        # Stack成 [B, M, d_model]
        person_reprs = torch.stack(person_reprs, dim=1)
        
        # 家庭成员间交互
        family_padding_mask = ~member_mask
        family_encoded = self.family_encoder(
            person_reprs,
            src_key_padding_mask=family_padding_mask
        )
        
        # 家庭表征聚合 [B, d_model]
        family_weights = self.family_pool(family_encoded)
        family_weights = family_weights.masked_fill(~member_mask.unsqueeze(-1), -1e9)
        family_weights = F.softmax(family_weights, dim=1)
        family_repr = (family_encoded * family_weights).sum(dim=1)
        
        # 分类
        person_logits = self.person_classifier(person_reprs)  # [B, M, num_person_patterns]
        family_logits = self.family_classifier(family_repr)   # [B, num_family_patterns]
        
        return {
            'person_repr': person_reprs,
            'family_repr': family_repr,
            'person_logits': person_logits,
            'family_logits': family_logits
        }


# ==================== 学生模型：从人口属性预测模式 ====================

class PersonAttributeEncoder(nn.Module):
    """个人属性编码器"""
    
    def __init__(self, config: PatternDistillConfig):
        super().__init__()
        self.config = config
        
        # 成员属性编码
        # member_dim包含：age(1) + 性别(2) + 驾照(2) + 关系(10) + 学历(8) + 职业(16) + valid_flag(1) + work_pos(1)
        # 这里我们只用前面的属性，不含最后两列
        self.member_encoder = nn.Sequential(
            nn.Linear(config.member_dim - 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
    def forward(self, member_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            member_attr: [B, M, member_dim] 成员属性
            
        Returns:
            [B, M, d_model] 成员编码
        """
        # 只取前member_dim-2列
        attr = member_attr[..., :-2]
        return self.member_encoder(attr)


class FamilyAttributeEncoder(nn.Module):
    """家庭属性编码器"""
    
    def __init__(self, config: PatternDistillConfig):
        super().__init__()
        self.config = config
        
        # 家庭属性编码
        self.family_encoder = nn.Sequential(
            nn.Linear(config.family_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
    def forward(self, family_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            family_attr: [B, family_dim] 家庭属性
            
        Returns:
            [B, d_model] 家庭编码
        """
        return self.family_encoder(family_attr)


class PatternStudent(nn.Module):
    """
    学生模型：从人口属性预测模式
    
    输入：家庭属性 + 成员属性
    输出：
        - person_repr: 个人模式表征 [B, M, d_model]
        - family_repr: 家庭模式表征 [B, d_model]
        - person_logits: 个人模式分类 [B, M, num_person_patterns]
        - family_logits: 家庭模式分类 [B, num_family_patterns]
    """
    
    def __init__(self, config: PatternDistillConfig):
        super().__init__()
        self.config = config
        
        # 属性编码器
        self.person_encoder = PersonAttributeEncoder(config)
        self.family_encoder = FamilyAttributeEncoder(config)
        
        # 家庭-成员交互（让成员感知家庭上下文）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 成员间交互
        member_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.member_interaction = nn.TransformerEncoder(
            member_encoder_layer,
            num_layers=2
        )
        self.member_interaction.enable_nested_tensor = False
        
        # 表征投影（对齐到教师的表征空间）
        self.person_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        self.family_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # 分类头（与教师共享结构，但参数独立）
        self.person_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_person_patterns)
        )
        
        self.family_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_family_patterns)
        )
        
    def forward(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            family_attr: [B, family_dim] 家庭属性
            member_attr: [B, M, member_dim] 成员属性
            member_mask: [B, M] 有效成员mask
            
        Returns:
            Dict包含:
                - person_repr: [B, M, d_model]
                - family_repr: [B, d_model]
                - person_logits: [B, M, num_person_patterns]
                - family_logits: [B, num_family_patterns]
        """
        B, M, _ = member_attr.shape
        
        # 编码属性
        person_emb = self.person_encoder(member_attr)  # [B, M, d_model]
        family_emb = self.family_encoder(family_attr)  # [B, d_model]
        
        # 家庭上下文注入到成员
        family_context = family_emb.unsqueeze(1)  # [B, 1, d_model]
        person_emb, _ = self.cross_attention(
            query=person_emb,
            key=family_context,
            value=family_context
        )
        
        # 成员间交互
        padding_mask = ~member_mask
        person_encoded = self.member_interaction(
            person_emb,
            src_key_padding_mask=padding_mask
        )
        
        # 表征投影
        person_repr = self.person_proj(person_encoded)
        
        # 家庭表征：聚合成员表征
        # 使用attention加权
        family_query = family_emb.unsqueeze(1)  # [B, 1, d_model]
        family_repr, _ = self.cross_attention(
            query=family_query,
            key=person_repr,
            value=person_repr,
            key_padding_mask=padding_mask
        )
        family_repr = family_repr.squeeze(1)  # [B, d_model]
        family_repr = self.family_proj(family_repr)
        
        # 分类
        person_logits = self.person_classifier(person_repr)
        family_logits = self.family_classifier(family_repr)
        
        return {
            'person_repr': person_repr,
            'family_repr': family_repr,
            'person_logits': person_logits,
            'family_logits': family_logits
        }


# ==================== 蒸馏损失 ====================

class DistillationLoss(nn.Module):
    """蒸馏损失函数"""
    
    def __init__(self, config: PatternDistillConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.alpha_soft = config.alpha_soft
        self.alpha_repr = config.alpha_repr
        self.alpha_hard = config.alpha_hard
        
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        family_pattern_labels: torch.Tensor,
        person_pattern_labels: torch.Tensor,
        member_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算蒸馏损失
        
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出（detached）
            family_pattern_labels: [B] 家庭模式标签（hard label）
            person_pattern_labels: [B, M] 个人模式标签（hard label）
            member_mask: [B, M] 有效成员mask
            
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失
        """
        T = self.temperature
        
        # ========== 1. 表征蒸馏损失 ==========
        # 个人表征
        person_repr_loss = F.mse_loss(
            student_outputs['person_repr'],
            teacher_outputs['person_repr'].detach(),
            reduction='none'
        )
        # mask掉无效成员
        person_repr_loss = (person_repr_loss * member_mask.unsqueeze(-1)).sum() / (member_mask.sum() * self.config.d_model + 1e-8)
        
        # 家庭表征
        family_repr_loss = F.mse_loss(
            student_outputs['family_repr'],
            teacher_outputs['family_repr'].detach()
        )
        
        repr_loss = person_repr_loss + family_repr_loss
        
        # ========== 2. 软标签蒸馏损失（KL散度） ==========
        # 个人模式
        student_person_log_prob = F.log_softmax(
            student_outputs['person_logits'] / T, dim=-1
        )
        teacher_person_prob = F.softmax(
            teacher_outputs['person_logits'].detach() / T, dim=-1
        )
        person_kl = F.kl_div(
            student_person_log_prob,
            teacher_person_prob,
            reduction='none'
        ).sum(dim=-1)  # [B, M]
        person_soft_loss = (person_kl * member_mask).sum() / (member_mask.sum() + 1e-8)
        person_soft_loss = person_soft_loss * (T * T)  # 温度缩放
        
        # 家庭模式
        student_family_log_prob = F.log_softmax(
            student_outputs['family_logits'] / T, dim=-1
        )
        teacher_family_prob = F.softmax(
            teacher_outputs['family_logits'].detach() / T, dim=-1
        )
        family_soft_loss = F.kl_div(
            student_family_log_prob,
            teacher_family_prob,
            reduction='batchmean'
        ) * (T * T)
        
        soft_loss = person_soft_loss + family_soft_loss
        
        # ========== 3. 硬标签损失（Focal Loss） ==========
        # 个人模式
        person_hard_loss = self._focal_loss(
            student_outputs['person_logits'].view(-1, self.config.num_person_patterns),
            person_pattern_labels.view(-1),
            member_mask.view(-1)
        )
        
        # 家庭模式
        family_hard_loss = F.cross_entropy(
            student_outputs['family_logits'],
            family_pattern_labels
        )
        
        hard_loss = person_hard_loss + family_hard_loss
        
        # ========== 总损失 ==========
        total_loss = (
            self.alpha_repr * repr_loss +
            self.alpha_soft * soft_loss +
            self.alpha_hard * hard_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'repr': repr_loss.item(),
            'soft': soft_loss.item(),
            'hard': hard_loss.item(),
            'person_repr': person_repr_loss.item(),
            'family_repr': family_repr_loss.item(),
            'person_soft': person_soft_loss.item(),
            'family_soft': family_soft_loss.item(),
            'person_hard': person_hard_loss.item(),
            'family_hard': family_hard_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25
    ) -> torch.Tensor:
        """Focal Loss for handling class imbalance"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        # 应用mask
        masked_loss = focal_loss * mask
        return masked_loss.sum() / (mask.sum() + 1e-8)


class TeacherOnlyLoss(nn.Module):
    """教师模型单独训练时的损失"""
    
    def __init__(self, config: PatternDistillConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        family_pattern_labels: torch.Tensor,
        person_pattern_labels: torch.Tensor,
        member_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        教师模型损失
        
        Args:
            outputs: 模型输出
            family_pattern_labels: [B] 家庭模式标签
            person_pattern_labels: [B, M] 个人模式标签
            member_mask: [B, M] 有效成员mask
        """
        # 个人模式损失
        person_logits = outputs['person_logits'].view(-1, self.config.num_person_patterns)
        person_targets = person_pattern_labels.view(-1)
        person_mask = member_mask.view(-1)
        
        person_loss = F.cross_entropy(person_logits, person_targets, reduction='none')
        person_loss = (person_loss * person_mask).sum() / (person_mask.sum() + 1e-8)
        
        # 家庭模式损失
        family_loss = F.cross_entropy(
            outputs['family_logits'],
            family_pattern_labels
        )
        
        total_loss = person_loss + family_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'person': person_loss.item(),
            'family': family_loss.item()
        }
        
        return total_loss, loss_dict


# ==================== 数据集 ====================

class PatternDistillDataset(torch.utils.data.Dataset):
    """蒸馏模型数据集"""
    
    def __init__(
        self,
        family_data: np.ndarray,      # [N, family_dim+1] 最后一列是home_zone
        member_data: np.ndarray,      # [N, M, member_dim]
        activity_data: np.ndarray,    # [N, M, T, activity_dim+2] 最后两列是位置
        member_mask: np.ndarray,      # [N, M]
        activity_mask: np.ndarray,    # [N, M, T]
        family_pattern: np.ndarray,   # [N, num_family_patterns] 概率分布
        person_pattern: np.ndarray    # [N, M, num_person_patterns] 概率分布
    ):
        self.family_data = torch.FloatTensor(family_data)
        self.member_data = torch.FloatTensor(member_data)
        self.activity_data = torch.FloatTensor(activity_data)
        self.member_mask = torch.BoolTensor(member_mask)
        self.activity_mask = torch.BoolTensor(activity_mask)
        self.family_pattern = torch.FloatTensor(family_pattern)
        self.person_pattern = torch.FloatTensor(person_pattern)
        
    def __len__(self):
        return len(self.family_data)
    
    def __getitem__(self, idx):
        # 提取活动特征（只取前27维，不含位置）
        activities = self.activity_data[idx, :, :, :27]
        
        return {
            'family_attr': self.family_data[idx, :-1],  # 不含home_zone
            'member_attr': self.member_data[idx],
            'activities': activities,
            'member_mask': self.member_mask[idx],
            'activity_mask': self.activity_mask[idx],
            'family_pattern': self.family_pattern[idx],
            'person_pattern': self.person_pattern[idx],
            # Hard labels (argmax of soft labels)
            'family_pattern_label': self.family_pattern[idx].argmax(),
            'person_pattern_label': self.person_pattern[idx].argmax(dim=-1)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """整理批次"""
    return {
        'family_attr': torch.stack([b['family_attr'] for b in batch]),
        'member_attr': torch.stack([b['member_attr'] for b in batch]),
        'activities': torch.stack([b['activities'] for b in batch]),
        'member_mask': torch.stack([b['member_mask'] for b in batch]),
        'activity_mask': torch.stack([b['activity_mask'] for b in batch]),
        'family_pattern': torch.stack([b['family_pattern'] for b in batch]),
        'person_pattern': torch.stack([b['person_pattern'] for b in batch]),
        'family_pattern_label': torch.stack([b['family_pattern_label'] for b in batch]),
        'person_pattern_label': torch.stack([b['person_pattern_label'] for b in batch])
    }


# ==================== 训练器 ====================

class PatternDistillTrainer:
    """蒸馏训练器"""
    
    def __init__(
        self,
        config: PatternDistillConfig,
        teacher: PatternTeacher,
        student: PatternStudent,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        save_dir: str = './checkpoints_distill'
    ):
        self.config = config
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.teacher.to(self.device)
        self.student.to(self.device)
        
        # 冻结教师
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 损失函数
        self.criterion = DistillationLoss(config)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.student.train()
        self.teacher.eval()
        
        total_losses = {}
        num_batches = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            # 移动数据到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 教师前向（不计算梯度）
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    batch['activities'],
                    batch['member_mask'],
                    batch['activity_mask']
                )
            
            # 学生前向
            student_outputs = self.student(
                batch['family_attr'],
                batch['member_attr'],
                batch['member_mask']
            )
            
            # 计算损失
            loss, loss_dict = self.criterion(
                student_outputs,
                teacher_outputs,
                batch['family_pattern_label'],
                batch['person_pattern_label'],
                batch['member_mask']
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            
            # 累积损失
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'soft': f"{loss_dict['soft']:.4f}",
                'repr': f"{loss_dict['repr']:.4f}"
            })
        
        # 平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        self.scheduler.step()
        self.current_epoch += 1
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
            
        self.student.eval()
        self.teacher.eval()
        
        total_losses = {}
        num_batches = 0
        
        # 准确率统计
        person_correct = 0
        person_total = 0
        family_correct = 0
        family_total = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 教师前向
            teacher_outputs = self.teacher(
                batch['activities'],
                batch['member_mask'],
                batch['activity_mask']
            )
            
            # 学生前向
            student_outputs = self.student(
                batch['family_attr'],
                batch['member_attr'],
                batch['member_mask']
            )
            
            # 计算损失
            _, loss_dict = self.criterion(
                student_outputs,
                teacher_outputs,
                batch['family_pattern_label'],
                batch['person_pattern_label'],
                batch['member_mask']
            )
            
            # 计算准确率
            # 个人模式
            person_pred = student_outputs['person_logits'].argmax(dim=-1)
            person_target = batch['person_pattern_label']
            mask = batch['member_mask']
            person_correct += ((person_pred == person_target) & mask).sum().item()
            person_total += mask.sum().item()
            
            # 家庭模式
            family_pred = student_outputs['family_logits'].argmax(dim=-1)
            family_target = batch['family_pattern_label']
            family_correct += (family_pred == family_target).sum().item()
            family_total += len(family_target)
            
            # 累积损失
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v
            num_batches += 1
        
        # 平均损失
        avg_losses = {f'val_{k}': v / num_batches for k, v in total_losses.items()}
        
        # 准确率
        avg_losses['val_person_acc'] = person_correct / (person_total + 1e-8)
        avg_losses['val_family_acc'] = family_correct / (family_total + 1e-8)
        
        return avg_losses
    
    def train(self, num_epochs: int = None):
        """完整训练"""
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"Starting distillation training on {self.device}")
        print(f"Teacher parameters: {sum(p.numel() for p in self.teacher.parameters()):,}")
        print(f"Student parameters: {sum(p.numel() for p in self.student.parameters()):,}")
        
        for epoch in range(num_epochs):
            # 训练
            train_losses = self.train_epoch()
            print(f"\nEpoch {epoch} - Train Loss: {train_losses['total']:.4f}")
            
            # 验证
            if self.val_loader is not None:
                val_losses = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_losses['val_total']:.4f}")
                print(f"  Person Acc: {val_losses['val_person_acc']:.4f}")
                print(f"  Family Acc: {val_losses['val_family_acc']:.4f}")
                
                # 保存最佳模型
                if val_losses['val_total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['val_total']
                    self.save_checkpoint('best_student.pt')
                    print("  New best model saved!")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'student_epoch_{epoch}.pt')
        
        print("Distillation training completed!")
        
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, f'{self.save_dir}/{filename}')
        
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


class TeacherTrainer:
    """教师模型训练器"""
    
    def __init__(
        self,
        config: PatternDistillConfig,
        teacher: PatternTeacher,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        save_dir: str = './checkpoints_teacher'
    ):
        self.config = config
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.teacher.to(self.device)
        
        # 损失函数
        self.criterion = TeacherOnlyLoss(config)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.teacher.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.current_epoch = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.teacher.train()
        
        total_losses = {}
        num_batches = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f'Teacher Epoch {self.current_epoch}')
        
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向
            outputs = self.teacher(
                batch['activities'],
                batch['member_mask'],
                batch['activity_mask']
            )
            
            # 计算损失
            loss, loss_dict = self.criterion(
                outputs,
                batch['family_pattern_label'],
                batch['person_pattern_label'],
                batch['member_mask']
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), 1.0)
            self.optimizer.step()
            
            # 累积损失
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
        
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        self.scheduler.step()
        self.current_epoch += 1
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
            
        self.teacher.eval()
        
        total_losses = {}
        num_batches = 0
        
        person_correct = 0
        person_total = 0
        family_correct = 0
        family_total = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.teacher(
                batch['activities'],
                batch['member_mask'],
                batch['activity_mask']
            )
            
            _, loss_dict = self.criterion(
                outputs,
                batch['family_pattern_label'],
                batch['person_pattern_label'],
                batch['member_mask']
            )
            
            # 准确率
            person_pred = outputs['person_logits'].argmax(dim=-1)
            person_target = batch['person_pattern_label']
            mask = batch['member_mask']
            person_correct += ((person_pred == person_target) & mask).sum().item()
            person_total += mask.sum().item()
            
            family_pred = outputs['family_logits'].argmax(dim=-1)
            family_target = batch['family_pattern_label']
            family_correct += (family_pred == family_target).sum().item()
            family_total += len(family_target)
            
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v
            num_batches += 1
        
        avg_losses = {f'val_{k}': v / num_batches for k, v in total_losses.items()}
        avg_losses['val_person_acc'] = person_correct / (person_total + 1e-8)
        avg_losses['val_family_acc'] = family_correct / (family_total + 1e-8)
        
        return avg_losses
    
    def train(self, num_epochs: int = None):
        """完整训练"""
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"Starting teacher training on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.teacher.parameters()):,}")
        
        for epoch in range(num_epochs):
            train_losses = self.train_epoch()
            print(f"\nEpoch {epoch} - Train Loss: {train_losses['total']:.4f}")
            
            if self.val_loader is not None:
                val_losses = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_losses['val_total']:.4f}")
                print(f"  Person Acc: {val_losses['val_person_acc']:.4f}")
                print(f"  Family Acc: {val_losses['val_family_acc']:.4f}")
                
                # 保存最佳模型（按准确率）
                avg_acc = (val_losses['val_person_acc'] + val_losses['val_family_acc']) / 2
                if avg_acc > self.best_val_acc:
                    self.best_val_acc = avg_acc
                    self.save_checkpoint('best_teacher.pt')
                    print("  New best model saved!")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'teacher_epoch_{epoch}.pt')
        
        print("Teacher training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        torch.save(checkpoint, f'{self.save_dir}/{filename}')
        
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


# ==================== 工厂函数 ====================

def create_teacher(config: PatternDistillConfig) -> PatternTeacher:
    """创建教师模型"""
    return PatternTeacher(config)


def create_student(config: PatternDistillConfig) -> PatternStudent:
    """创建学生模型"""
    return PatternStudent(config)


def load_teacher(checkpoint_path: str, config: PatternDistillConfig = None) -> PatternTeacher:
    """加载训练好的教师模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if config is None:
        config = checkpoint.get('config', PatternDistillConfig())
    
    teacher = create_teacher(config)
    teacher.load_state_dict(checkpoint['teacher_state_dict'])
    teacher.eval()
    return teacher


def load_student(checkpoint_path: str, config: PatternDistillConfig = None) -> PatternStudent:
    """加载训练好的学生模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if config is None:
        config = checkpoint.get('config', PatternDistillConfig())
    
    student = create_student(config)
    student.load_state_dict(checkpoint['student_state_dict'])
    student.eval()
    return student
