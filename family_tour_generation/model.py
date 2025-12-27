"""
完整的家庭活动链生成模型
整合: PLE编码器 + MTAN Decoder
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from torch import Tensor

from config import ModelConfig
from ple_encoder import PLEEncoder
from mtan_decoder import MTANDecoder
from data import FamilyTourBatch


class FamilyTourGenerator(nn.Module):
    """
    家庭活动链生成模型
    
    架构:
    1. PLE编码器: 提取家庭、成员集合、个体信息
    2. MTAN Decoder: 任务特定注意力 + Cross-Role注意力 + Transformer Decoder
    
    数据流:
    输入:
        - 家庭属性: (B, Ff)
        - 成员属性: (B, max_members, Fm)
        - 成员掩码: (B, max_members)
    
    编码:
        - PLE: 家庭属性 + 成员属性 -> 成员表示 (B, max_members, d_model)
    
    解码:
        - MTAN Decoder: 自回归生成活动链 (B, max_members, max_activities, *)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # PLE编码器
        self.encoder = PLEEncoder(config)
        
        # MTAN Decoder
    def forward(
        self,
        batch: FamilyTourBatch,
        teacher_forcing: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            batch: FamilyTourBatch数据
            teacher_forcing: 是否使用teacher forcing (训练时True, 推理时False)
        
        Returns:
            predictions: dict
                - 'continuous': (B, max_members, max_activities, 2)
                - 'purpose': (B, max_members, max_activities, num_purposes)
                - 'mode': (B, max_members, max_activities, num_modes)
                - 'driver': (B, max_members, max_activities, num_driver)
                - 'joint': (B, max_members, max_activities, num_joint)
        """
        # PLE编码
        member_repr, family_repr, pattern_probs = self.encoder(
            batch.family_attr,
            batch.member_attr,
            batch.member_mask
        )
        
        if teacher_forcing:
            # 训练模式: teacher forcing
            predictions = self.decoder(
                member_repr=member_repr,
                family_repr=family_repr,
                target_activities=batch.activities,
                member_mask=batch.member_mask,
                activity_mask=batch.activity_mask,
                pattern_outputs=pattern_probs,
                home_zones = batch.home_zones,  # 新增
                target_destinations = batch.target_destinations  # 新增
            )
        else:
            # 推理模式: 自回归生成
            predictions = self.decoder.generate(
                member_repr=member_repr,
                family_repr=family_repr,
                member_mask=batch.member_mask,
                home_zones=batch.home_zones  # 新增
            )
        
        return predictions, pattern_probs
    
    def generate(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.BoolTensor,
        max_length: int = None,
        home_zones=None
    ) -> Tuple[Dict[str, Tensor], Any]:
        """
        生成活动链 (推理接口)
        
        Args:
            family_attr: (B, Ff)
            member_attr: (B, max_members, Fm)
            member_mask: (B, max_members)
            max_length: 最大生成长度
        
        Returns:
            generated: dict of tensors
        """
        # PLE编码
        member_repr, family_repr, pattern_prob = self.encoder(
            family_attr, member_attr, member_mask
        )
        
        # 自回归生成
        return self.decoder.generate(
            member_repr=member_repr,
            family_repr=family_repr,
            member_mask=member_mask,
            max_length=max_length,
            pattern_outputs=pattern_prob,
            home_zones=batch.home_zones), pattern_prob
    
    def get_encoder_output(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.BoolTensor
    ) -> tuple:
        """
        获取编码器输出 (用于分析)
        
        Returns:
            member_repr: (B, max_members, d_model)
            family_repr: (B, d_model)
        """
        return self.encoder(family_attr, member_attr, member_mask)
    
    def count_parameters(self) -> Dict[str, int]:
        """统计参数量"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params
        }


def create_model(config: ModelConfig) -> FamilyTourGenerator:
    """创建模型实例"""
    model = FamilyTourGenerator(config)
    
    # 初始化权重
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model


# 测试代码
if __name__ == '__main__':
    # 配置
    config = ModelConfig(
        family_dim=32,
        member_dim=48,
        activity_dim=27,
        max_members=8,
        max_activities=6,
        d_model=256,
        num_heads=8,
        num_decoder_layers=20,
        num_inducing_points=16
    )
    
    # 创建模型
    model = create_model(config)
    print(f"Model parameters: {model.count_parameters()}")
    
    # 测试数据
    batch_size = 4
    family_attr = torch.randn(batch_size, config.family_dim)
    member_attr = torch.randn(batch_size, config.max_members, config.member_dim)
    member_mask = torch.ones(batch_size, config.max_members, dtype=torch.bool)
    member_mask[:, -2:] = False  # 最后两个成员无效
    
    activities = torch.randn(batch_size, config.max_members, config.max_activities, config.activity_dim)
    # 构造有效的one-hot
    activities[..., 2:12] = torch.zeros_like(activities[..., 2:12])
    activities[..., 2] = 1  # 目的
    activities[..., 12:23] = torch.zeros_like(activities[..., 12:23])
    activities[..., 12] = 1  # 方式
    activities[..., 23:25] = torch.zeros_like(activities[..., 23:25])
    activities[..., 23] = 1  # 驾驶状态
    activities[..., 25:27] = torch.zeros_like(activities[..., 25:27])
    activities[..., 25] = 1  # 联合出行
    
    activity_mask = torch.ones(batch_size, config.max_members, config.max_activities, dtype=torch.bool)
    activity_mask[:, :, -1] = False  # 最后一个活动无效
    
    # 创建batch
    batch = FamilyTourBatch(
        family_attr=family_attr,
        member_attr=member_attr,
        member_mask=member_mask,
        activities=activities,
        activity_mask=activity_mask
    )
    
    # 前向传播 (teacher forcing)
    model.eval()
    with torch.no_grad():
        predictions = model(batch, teacher_forcing=True)
    
    print("\nPrediction shapes:")
    for k, v in predictions.items():
        print(f"  {k}: {v.shape}")
    
    # 生成模式
    with torch.no_grad():
        generated = model.generate(family_attr, member_attr, member_mask)
    
    print("\nGenerated shapes:")
    for k, v in generated.items():
        print(f"  {k}: {v.shape}")
