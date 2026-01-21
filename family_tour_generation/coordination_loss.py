"""
家庭协调损失函数模块
- 联合出行一致性损失
- 车辆约束损失
- 家庭起止点约束损失

添加方式：在 losses.py 中导入并集成到 FamilyTourLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class FamilyCoordinationLoss(nn.Module):
    """
    家庭协调损失
    
    包含:
    1. 联合出行一致性: 联合出行的成员应有相同的目的地、出行方式、时间
    2. 车辆约束: 同一时段驾车人数不超过家庭车辆数
    3. 家庭起止约束: 第一个和最后一个活动地点应为家
    """
    
    def __init__(
        self,
        lambda_joint: float = 1.0,      # 联合出行一致性权重
        lambda_vehicle: float = 2.0,     # 车辆约束权重
        lambda_home: float = 1.0,        # 家庭起止约束权重
        num_zones: int = 2006,
        home_zone_idx: int = None,       # 如果有固定的home zone编码
    ):
        super().__init__()
        self.lambda_joint = lambda_joint
        self.lambda_vehicle = lambda_vehicle
        self.lambda_home = lambda_home
        self.num_zones = num_zones
        self.home_zone_idx = home_zone_idx
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor,
        num_vehicles: torch.Tensor = None,      # (batch,) 家庭车辆数
        home_zones: torch.Tensor = None,        # (batch,) 家庭所在zone
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: 模型预测
                - 'continuous': (batch, max_members, max_activities, 2) 时间
                - 'mode': (batch, max_members, max_activities, num_modes) 方式logits
                - 'driver': (batch, max_members, max_activities, num_driver) 驾驶状态logits
                - 'joint': (batch, max_members, max_activities, num_joint) 联合出行logits
                - 'destination': (batch, max_members, max_activities, num_zones) 目的地logits (可选)
            member_mask: (batch, max_members) 有效成员
            activity_mask: (batch, max_members, max_activities) 有效活动
            num_vehicles: (batch,) 家庭车辆数
            home_zones: (batch,) 家庭所在zone索引
        
        Returns:
            total_loss: 协调损失总和
            loss_dict: 各部分损失
        """
        losses = {}
        
        # 1. 联合出行一致性损失
        joint_loss = self.compute_joint_consistency_loss(
            predictions, member_mask, activity_mask
        )
        losses['joint_consistency'] = joint_loss
        
        # 2. 车辆约束损失
        if num_vehicles is not None:
            vehicle_loss = self.compute_vehicle_constraint_loss(
                predictions, member_mask, activity_mask, num_vehicles
            )
            losses['vehicle_constraint'] = vehicle_loss
        else:
            vehicle_loss = torch.tensor(0.0, device=member_mask.device)
            losses['vehicle_constraint'] = vehicle_loss
        
        # 3. 家庭起止约束损失
        if home_zones is not None and 'destination' in predictions:
            home_loss = self.compute_home_constraint_loss(
                predictions, member_mask, activity_mask, home_zones
            )
            losses['home_constraint'] = home_loss
        else:
            home_loss = torch.tensor(0.0, device=member_mask.device)
            losses['home_constraint'] = home_loss
        
        # 加权总损失
        total_loss = (
            self.lambda_joint * joint_loss +
            self.lambda_vehicle * vehicle_loss +
            self.lambda_home * home_loss
        )
        
        return total_loss, losses
    
    def compute_joint_consistency_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        联合出行一致性损失
        
        当两个成员在同一时间段都预测为联合出行时，
        他们的目的地、出行方式、时间应该一致
        
        使用软概率计算，完全可微
        """
        batch_size, max_members, max_activities = activity_mask.shape
        device = activity_mask.device
        
        # 获取联合出行概率 (softmax后取joint=1的概率)
        joint_logits = predictions['joint']  # (B, M, A, 2)
        joint_prob = F.softmax(joint_logits, dim=-1)[..., 1]  # (B, M, A) 联合出行概率
        
        # 获取时间预测
        time_pred = predictions['continuous']  # (B, M, A, 2)
        
        # 获取方式概率分布
        mode_logits = predictions['mode']  # (B, M, A, num_modes)
        mode_prob = F.softmax(mode_logits, dim=-1)  # (B, M, A, num_modes)
        
        # 获取目的地概率分布 (如果有)
        if 'destination' in predictions:
            dest_logits = predictions['destination']  # (B, M, A, num_zones)
            dest_prob = F.softmax(dest_logits, dim=-1)
        else:
            dest_prob = None
        
        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0
        
        # 遍历所有成员对
        for i in range(max_members):
            for j in range(i + 1, max_members):
                # 两个成员都有效的mask
                pair_valid = member_mask[:, i] & member_mask[:, j]  # (B,)
                if not pair_valid.any():
                    continue
                
                # 两个成员在每个活动位置的联合出行概率乘积
                # (越高表示越可能是联合出行)
                joint_weight = joint_prob[:, i, :] * joint_prob[:, j, :]  # (B, A)
                
                # 两个活动都有效
                activity_valid = activity_mask[:, i, :] & activity_mask[:, j, :]  # (B, A)
                weight_mask = pair_valid.unsqueeze(-1) & activity_valid  # (B, A)
                
                if not weight_mask.any():
                    continue
                
                # === 时间一致性 ===
                time_i = time_pred[:, i, :]  # (B, A, 2)
                time_j = time_pred[:, j, :]  # (B, A, 2)
                time_diff = ((time_i - time_j) ** 2).sum(dim=-1)  # (B, A)
                
                # 加权时间损失 (联合出行概率越高，时间差异惩罚越大)
                time_loss = (joint_weight * time_diff * weight_mask.float()).sum()
                
                # === 方式一致性 (KL散度) ===
                mode_i = mode_prob[:, i, :]  # (B, A, num_modes)
                mode_j = mode_prob[:, j, :]  # (B, A, num_modes)
                # 对称KL散度
                kl_ij = F.kl_div(mode_i.log().clamp(min=-100), mode_j, reduction='none').sum(dim=-1)
                kl_ji = F.kl_div(mode_j.log().clamp(min=-100), mode_i, reduction='none').sum(dim=-1)
                mode_kl = 0.5 * (kl_ij + kl_ji)  # (B, A)
                
                mode_loss = (joint_weight * mode_kl * weight_mask.float()).sum()
                
                # === 目的地一致性 (如果有) ===
                if dest_prob is not None:
                    dest_i = dest_prob[:, i, :]  # (B, A, num_zones)
                    dest_j = dest_prob[:, j, :]  # (B, A, num_zones)
                    # 对称KL散度
                    dest_kl_ij = F.kl_div(dest_i.log().clamp(min=-100), dest_j, reduction='none').sum(dim=-1)
                    dest_kl_ji = F.kl_div(dest_j.log().clamp(min=-100), dest_i, reduction='none').sum(dim=-1)
                    dest_kl = 0.5 * (dest_kl_ij + dest_kl_ji)
                    
                    dest_loss = (joint_weight * dest_kl * weight_mask.float()).sum()
                else:
                    dest_loss = 0.0
                
                total_loss = total_loss + time_loss + mode_loss + dest_loss
                num_pairs += weight_mask.sum().item()
        
        # 归一化
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        
        return total_loss
    
    def compute_vehicle_constraint_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor,
        num_vehicles: torch.Tensor
    ) -> torch.Tensor:
        """
        车辆约束损失
        
        同一时段内驾车人数不应超过家庭车辆数
        使用软约束: ReLU(驾车人数 - 车辆数)
        """
        batch_size, max_members, max_activities = activity_mask.shape
        device = activity_mask.device
        
        # 获取驾驶状态概率 (driver=1表示驾车)
        driver_logits = predictions['driver']  # (B, M, A, 2)
        driver_prob = F.softmax(driver_logits, dim=-1)[..., 1]  # (B, M, A) 驾车概率
        
        # 获取时间预测 (用于判断同一时段)
        time_pred = predictions['continuous']  # (B, M, A, 2) [start, end]
        
        # 简化处理: 对每个活动位置，统计该位置所有成员的驾车概率之和
        # (假设同一活动位置大致对应同一时段)
        
        valid_mask = member_mask.unsqueeze(-1) & activity_mask  # (B, M, A)
        
        # 每个活动位置的驾车人数期望
        driver_count = (driver_prob * valid_mask.float()).sum(dim=1)  # (B, A)
        
        # 每个活动位置的有效成员数
        valid_count = valid_mask.float().sum(dim=1)  # (B, A)
        valid_positions = valid_count > 0  # (B, A)
        
        # 超出车辆数的惩罚
        excess = F.relu(driver_count - num_vehicles.unsqueeze(-1).float())  # (B, A)
        
        # 只在有效位置计算
        if valid_positions.sum() > 0:
            loss = (excess * valid_positions.float()).sum() / valid_positions.sum()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
    
    def compute_home_constraint_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor,
        home_zones: torch.Tensor
    ) -> torch.Tensor:
        """
        家庭起止约束损失
        
        第一个活动的起点和最后一个活动的终点应为家
        (这里简化为: 第一个和最后一个活动的目的地应接近家)
        """
        batch_size, max_members, max_activities = activity_mask.shape
        device = activity_mask.device
        
        if 'destination' not in predictions:
            return torch.tensor(0.0, device=device)
        
        dest_logits = predictions['destination']  # (B, M, A, num_zones)
        dest_prob = F.softmax(dest_logits, dim=-1)  # (B, M, A, num_zones)
        
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0
        
        for b in range(batch_size):
            home_zone = home_zones[b].long()
            
            for m in range(max_members):
                if not member_mask[b, m]:
                    continue
                
                # 找到该成员的有效活动范围
                valid_acts = activity_mask[b, m]  # (A,)
                valid_indices = valid_acts.nonzero(as_tuple=True)[0]
                
                if len(valid_indices) == 0:
                    continue
                
                first_idx = valid_indices[0]
                last_idx = valid_indices[-1]
                
                # 第一个活动目的地应为家 (或接近家)
                first_home_prob = dest_prob[b, m, first_idx, home_zone]
                
                # 最后一个活动目的地应为家
                last_home_prob = dest_prob[b, m, last_idx, home_zone]
                
                # 负对数似然 (希望home概率高)
                loss = -torch.log(first_home_prob.clamp(min=1e-8)) - torch.log(last_home_prob.clamp(min=1e-8))
                
                total_loss = total_loss + loss
                num_valid += 1
        
        if num_valid > 0:
            total_loss = total_loss / num_valid
        
        return total_loss


# ==================== 集成到 FamilyTourLoss 的代码 ====================

def add_coordination_loss_to_family_tour_loss():
    """
    如何在 FamilyTourLoss 中集成协调损失的示例代码
    
    在 losses.py 的 FamilyTourLoss.__init__ 中添加:
    
    ```python
    from coordination_loss import FamilyCoordinationLoss
    
    # 在 __init__ 中:
    self.coordination_loss = FamilyCoordinationLoss(
        lambda_joint=train_config.lambda_joint if hasattr(train_config, 'lambda_joint') else 1.0,
        lambda_vehicle=train_config.lambda_vehicle if hasattr(train_config, 'lambda_vehicle') else 2.0,
        lambda_home=train_config.lambda_home if hasattr(train_config, 'lambda_home') else 1.0,
    )
    self.use_coordination = getattr(train_config, 'use_coordination_loss', False)
    self.coordination_weight = getattr(train_config, 'coordination_weight', 0.5)
    ```
    
    在 forward 方法的 return 之前添加:
    
    ```python
    # ===== 协调损失 =====
    if self.use_coordination:
        coord_loss, coord_losses = self.coordination_loss(
            predictions,
            member_mask,
            activity_mask,
            num_vehicles=getattr(batch, 'num_vehicles', None),  # 需要在batch中添加
            home_zones=getattr(batch, 'home_zones', None),      # 需要在batch中添加
        )
        losses.update({f'coord_{k}': v for k, v in coord_losses.items()})
        total_loss = total_loss + self.coordination_weight * coord_loss
    ```
    """
    pass


# ==================== 简化版：直接作为函数使用 ====================

def compute_coordination_loss(
    predictions: Dict[str, torch.Tensor],
    member_mask: torch.BoolTensor,
    activity_mask: torch.BoolTensor,
    num_vehicles: Optional[torch.Tensor] = None,
    home_zones: Optional[torch.Tensor] = None,
    lambda_joint: float = 1.0,
    lambda_vehicle: float = 2.0,
    lambda_home: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算协调损失的函数版本 (无需实例化类)
    
    可以直接在 FamilyTourLoss.forward 中调用
    """
    coord_module = FamilyCoordinationLoss(
        lambda_joint=lambda_joint,
        lambda_vehicle=lambda_vehicle,
        lambda_home=lambda_home
    )
    return coord_module(predictions, member_mask, activity_mask, num_vehicles, home_zones)


# ==================== 在 config.py 中添加的配置 ====================

"""
在 TrainConfig 中添加:

@dataclass
class TrainConfig:
    # ... 现有配置 ...
    
    # 协调损失配置
    use_coordination_loss: bool = True
    coordination_weight: float = 0.5
    lambda_joint: float = 1.0       # 联合出行一致性权重
    lambda_vehicle: float = 2.0     # 车辆约束权重  
    lambda_home: float = 1.0        # 家庭起止约束权重
"""
