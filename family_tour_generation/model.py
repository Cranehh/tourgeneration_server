"""
完整的家庭活动链生成模型
整合: PLE编码器 + MTAN Decoder
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from torch import Tensor
import torch.nn.functional as F
from config import ModelConfig
from ple_encoder import PLEEncoder
from mtan_decoder import MTANDecoder
from data import FamilyTourBatch
from integration import IntegratedPatternPredictor, PatternConditionInjector
from nash_bargaining_sqp_optnet import HouseholdNashBargainingLayer, UtilityParams



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
        self.encoder = PLEEncoder(
            config=config,
            num_family_patterns=getattr(config, 'num_family_patterns', 118),
            num_individual_patterns=getattr(config, 'num_individual_patterns', 207),
            num_shared_experts=getattr(config, 'num_shared_experts', 2),
            num_task_experts=getattr(config, 'num_task_experts', 1),
            num_cgc_layers=getattr(config, 'num_cgc_layers', 3)
        )

        self.pattern_predictor = IntegratedPatternPredictor(
            student_checkpoint_path="./checkpoints_student/best_student.pt",
            target_d_model=config.d_model,  # 主模型的d_model
            freeze_student=True   # 建议先冻结，后期可以解冻微调
        )
        
        # MTAN Decoder

        self.decoder = MTANDecoder(config)

        # 新增: Nash Bargaining 层
        if getattr(config, 'use_nash_bargaining', False):
            self.nash_layer = HouseholdNashBargainingLayer(
                num_zones=2006,
                num_modes=config.num_modes,
                max_members=config.max_members,
                max_tours=config.max_activities,
                utility_params=UtilityParams(
                    alpha_joint=config.nash_config.get('alpha_joint', 0.1),
                    alpha_social=config.nash_config.get('alpha_social', 0.3),
                    theta_escort=config.nash_config.get('theta_escort', -0.58),
                    theta_car=config.nash_config.get('theta_car', -1.0),
                    theta_pt=config.nash_config.get('theta_pt', -0.4)
                ),
                sqp_max_iter=config.nash_config.get('sqp_max_iter', 15),
                sqp_tol=config.nash_config.get('sqp_tol', 1e-4),
                enabled=True
            )
        else:
            self.nash_layer = None
        # 加载出行时间矩阵（需要预先准备）
        self.travel_time_matrix = None  # 需要从外部加载

    def _apply_nash_bargaining(
            self,
            predictions: Dict[str, torch.Tensor],
            batch
    ) -> Dict[str, torch.Tensor]:
        """应用 Nash Bargaining 协调层"""

        # 准备输入
        # 注意：Nash 层需要 departure_time 和 duration，但模型输出是 continuous[..., 0] 和 continuous[..., 1]
        departure_time = predictions['continuous'][..., 0]  # [B, M, T]
        duration = predictions['continuous'][..., 1] - predictions['continuous'][..., 0]  # 计算持续时间

        # 确保 duration 为正
        duration = torch.clamp(duration, min=0.01)

        # 准备 member_is_adult（需要从 batch 或 member_attr 中提取）
        # 假设 member_attr 中第 0 维是年龄或类似指标
        member_is_adult = (batch.member_attr[..., 0] > 18).float()  # 需要根据实际数据调整

        # 调用 Nash 层
        nash_output = self.nash_layer(
            departure_time=departure_time,
            duration=duration,
            destination_logits=predictions['destination'],
            mode_logits=predictions['mode'],
            is_joint_logit=predictions['joint'][..., 1] - predictions['joint'][..., 0],  # 转为单值 logit
            is_driver_logit=predictions['driver'][..., 1] - predictions['driver'][..., 0],
            member_mask=batch.member_mask.float(),
            tour_mask=batch.activity_mask.float(),
            num_vehicles=batch.num_vehicles if batch.num_vehicles is not None else torch.ones(batch.batch_size,
                                                                                              device=batch.device),
            home_zone=batch.home_zones,
            member_is_adult=member_is_adult,
            travel_time_matrix=self.travel_time_matrix,
        )

        # 将 Nash 输出转换回原格式
        coordinated_predictions = predictions.copy()

        # 更新 continuous（从 departure + duration 恢复）
        coordinated_continuous = torch.stack([
            nash_output.departure_time,
            nash_output.departure_time + nash_output.duration
        ], dim=-1)
        coordinated_predictions['continuous'] = coordinated_continuous

        # 更新 destination (logits)
        coordinated_predictions['destination'] = nash_output.destination_logits

        # 更新 mode (logits)
        coordinated_predictions['mode'] = nash_output.mode_logits

        # 更新 joint (转回 2-class logits)
        joint_logit = nash_output.is_joint_logit
        coordinated_predictions['joint'] = torch.stack([
            -joint_logit / 2, joint_logit / 2
        ], dim=-1)

        # 更新 driver (转回 2-class logits)
        driver_logit = nash_output.is_driver_logit
        coordinated_predictions['driver'] = torch.stack([
            -driver_logit / 2, driver_logit / 2
        ], dim=-1)

        return coordinated_predictions
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
            batch.member_mask,
            batch.home_zones,
            batch.work_positions
        )

        # 在主模型中使用
        member_attr_padded = F.pad(batch.member_attr, (0, 1))
        outputs = self.pattern_predictor(batch.family_attr, member_attr_padded, batch.member_mask)
        pattern_probs = {}
        # 获取模式概率分布（可选，用于监督）
        pattern_probs['family_pattern_prob'] = outputs['family_pattern_prob']  # [B, 20]
        pattern_probs['individual_pattern_prob'] = outputs['person_pattern_prob']  # [B, M, 40]
        
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

        # Nash 协调
        if self.nash_layer is not None and self.training:
            predictions = self._apply_nash_bargaining(predictions, batch)
        
        return predictions, pattern_probs
    
    def generate(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.BoolTensor,
        max_length: int = None,
        home_zones=None,
        work_positions=None,
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
            family_attr, member_attr, member_mask, home_zones, work_positions
        )

        member_attr_padded = F.pad(member_attr, (0, 1))
        outputs = self.pattern_predictor(family_attr, member_attr_padded, member_mask)
        pattern_prob = {}
        # 获取模式概率分布（可选，用于监督）
        pattern_prob['family_pattern_prob'] = outputs['family_pattern_prob']  # [B, 20]
        pattern_prob['individual_pattern_prob'] = outputs['person_pattern_prob']  # [B, M, 40]
        
        # 自回归生成
        return self.decoder.generate(
            member_repr=member_repr,
            family_repr=family_repr,
            member_mask=member_mask,
            max_length=max_length,
            pattern_outputs=pattern_prob,
            home_zones=home_zones), pattern_prob
    
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
