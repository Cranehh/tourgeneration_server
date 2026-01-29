"""
家庭层面 Nash Bargaining 协调层 (改进版)
基于 SQP + OptNet 的可微优化实现

改进点:
1. 支持 mask 机制 - 处理不存在的成员和活动
2. 与 OptNet (qpth) 深度整合 - 利用 batch 并行优化
3. 直接输出 x* - 与原有损失函数结合进行端到端反向传播

核心流程:
    神经网络输出 → SQP求解Nash Bargaining → x*(θ) → 原有损失函数 → 反向传播
                                              ↑
                              OptNet: 通过KKT隐函数求 dx*/dθ (精确梯度)

Author: 郝赫
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np

# 导入 qpth (OptNet)
try:
    from qpth.qp import QPFunction
    QPTH_AVAILABLE = True
except ImportError:
    QPTH_AVAILABLE = False
    print("Warning: qpth not available. Install with: pip install qpth")


# =============================================================================
#                              数据结构定义
# =============================================================================

@dataclass
class UtilityParams:
    """效用函数参数 (Rezvany et al. 2023)"""
    alpha_joint: float = 0.1        # 联合活动奖励
    alpha_social: float = 0.3       # 外出奖励
    theta_escort: float = -0.58     # 接送惩罚
    theta_car: float = -1.0         # 小汽车时间系数
    theta_pt: float = -0.4          # 公交时间系数
    theta_walk: float = -1.5        # 步行时间系数
    theta_bike: float = -0.8        # 骑行时间系数
    beta_time: float = 1.0          # 时间偏差惩罚
    gamma_duration: float = 1.0     # 持续时间偏差惩罚
    utility_baseline: float = 10.0  # 基准效用 (确保U>0)


class NashBargainingOutput(NamedTuple):
    """输出结构"""
    # 协调后的决策变量 (与输入同形状，可直接用于原有损失函数)
    departure_time: Tensor      # [batch, max_members, max_tours]
    duration: Tensor            # [batch, max_members, max_tours]
    destination_logits: Tensor  # [batch, max_members, max_tours, num_zones]
    mode_logits: Tensor         # [batch, max_members, max_tours, num_modes]
    is_joint_logit: Tensor      # [batch, max_members, max_tours]
    is_driver_logit: Tensor     # [batch, max_members, max_tours]
    
    # 附加信息
    nash_welfare: Tensor        # [batch] Nash社会福利
    converged: Tensor           # [batch] 是否收敛


# =============================================================================
#                          核心：Nash Bargaining QP Layer
# =============================================================================

class NashBargainingQPLayer(nn.Module):
    """
    Nash Bargaining 协调层
    
    将神经网络输出作为参数，通过SQP求解Nash Bargaining问题
    最后一步QP使用OptNet实现batch并行和精确梯度反传
    
    输入输出格式与原有模型一致，支持mask
    """
    
    def __init__(
        self,
        num_zones: int,
        num_modes: int = 4,
        max_members: int = 6,
        max_tours: int = 8,
        utility_params: Optional[UtilityParams] = None,
        sqp_max_iter: int = 20,
        sqp_tol: float = 1e-4,
    ):
        super().__init__()
        
        self.num_zones = num_zones
        self.num_modes = num_modes
        self.max_members = max_members
        self.max_tours = max_tours
        self.sqp_max_iter = sqp_max_iter
        self.sqp_tol = sqp_tol
        
        # 效用参数
        self.params = utility_params or UtilityParams()
        
        # 注册效用参数为可学习参数 (可选)
        self.register_buffer('theta_travel', torch.tensor([
            self.params.theta_car,
            self.params.theta_pt,
            self.params.theta_walk,
            self.params.theta_bike,
        ][:num_modes]))
        
        # 变量维度计算
        # 每个成员每个tour: departure(1) + duration(1) + dest(Z) + mode(M) + joint(1) + driver(1)
        self.vars_per_tour = 2 + num_zones + num_modes + 2
        self.vars_per_member = max_tours * self.vars_per_tour
        self.vars_per_household = max_members * self.vars_per_member
        
    def forward(
        self,
        # 神经网络输出 (作为初始点和参数θ)
        departure_time: Tensor,         # [batch, max_members, max_tours]
        duration: Tensor,               # [batch, max_members, max_tours]
        destination_logits: Tensor,     # [batch, max_members, max_tours, num_zones]
        mode_logits: Tensor,            # [batch, max_members, max_tours, num_modes]
        is_joint_logit: Tensor,         # [batch, max_members, max_tours]
        is_driver_logit: Tensor,        # [batch, max_members, max_tours]
        # Mask
        member_mask: Tensor,            # [batch, max_members] 有效成员
        tour_mask: Tensor,              # [batch, max_members, max_tours] 有效tour
        # 家庭信息
        num_vehicles: Tensor,           # [batch]
        home_zone: Tensor,              # [batch]
        member_is_adult: Tensor,        # [batch, max_members] 是否成人
        # 出行时间矩阵
        travel_time_matrix: Tensor,     # [num_zones, num_zones, num_modes] 或 [batch, ...]
        # 偏好 (可选，默认使用NN输出)
        preferred_departure: Optional[Tensor] = None,  # [batch, max_members, max_tours]
        preferred_duration: Optional[Tensor] = None,
        activity_flexibility: Optional[Tensor] = None, # [batch, max_members, max_tours]
    ) -> NashBargainingOutput:
        """
        前向传播
        
        Returns:
            NashBargainingOutput: 协调后的变量，形状与输入一致
        """
        batch_size = departure_time.shape[0]
        device = departure_time.device
        dtype = departure_time.dtype
        
        # 默认偏好
        if preferred_departure is None:
            preferred_departure = departure_time.detach()
        if preferred_duration is None:
            preferred_duration = duration.detach()
        if activity_flexibility is None:
            activity_flexibility = torch.ones_like(departure_time)
        
        # 打包输入为优化向量
        x0 = self._pack_variables(
            departure_time, duration, destination_logits,
            mode_logits, is_joint_logit, is_driver_logit
        )  # [batch, total_vars]
        
        # 构建mask向量
        var_mask = self._build_variable_mask(member_mask, tour_mask)  # [batch, total_vars]
        
        # SQP迭代求解
        x_opt, converged = self._sqp_solve(
            x0=x0,
            var_mask=var_mask,
            member_mask=member_mask,
            tour_mask=tour_mask,
            num_vehicles=num_vehicles,
            home_zone=home_zone,
            member_is_adult=member_is_adult,
            travel_time_matrix=travel_time_matrix,
            preferred_departure=preferred_departure,
            preferred_duration=preferred_duration,
            activity_flexibility=activity_flexibility,
        )
        
        # 解包为原始形状
        (out_departure, out_duration, out_dest_logits,
         out_mode_logits, out_joint_logit, out_driver_logit) = self._unpack_variables(x_opt)
        
        # 应用mask (无效位置保持原值)
        full_mask = tour_mask.unsqueeze(-1)  # [batch, max_members, max_tours, 1]
        
        out_departure = torch.where(tour_mask.bool(), out_departure, departure_time)
        out_duration = torch.where(tour_mask.bool(), out_duration, duration)
        out_dest_logits = torch.where(full_mask.bool(), out_dest_logits, destination_logits)
        out_mode_logits = torch.where(
            full_mask[..., :self.num_modes].bool() if full_mask.shape[-1] >= self.num_modes 
            else full_mask.expand(-1,-1,-1,self.num_modes).bool(),
            out_mode_logits, mode_logits
        )
        out_joint_logit = torch.where(tour_mask.bool(), out_joint_logit, is_joint_logit)
        out_driver_logit = torch.where(tour_mask.bool(), out_driver_logit, is_driver_logit)
        
        # 计算Nash福利
        nash_welfare = self._compute_nash_welfare(
            out_departure, out_duration, out_dest_logits, out_mode_logits,
            out_joint_logit, out_driver_logit,
            member_mask, tour_mask, home_zone, member_is_adult,
            travel_time_matrix, preferred_departure, preferred_duration, activity_flexibility,
        )
        
        return NashBargainingOutput(
            departure_time=out_departure,
            duration=out_duration,
            destination_logits=out_dest_logits,
            mode_logits=out_mode_logits,
            is_joint_logit=out_joint_logit,
            is_driver_logit=out_driver_logit,
            nash_welfare=nash_welfare,
            converged=converged,
        )
    
    # =========================================================================
    #                          变量打包/解包
    # =========================================================================
    
    def _pack_variables(
        self,
        departure: Tensor,      # [batch, M, T]
        duration: Tensor,       # [batch, M, T]
        dest_logits: Tensor,    # [batch, M, T, Z]
        mode_logits: Tensor,    # [batch, M, T, Modes]
        joint_logit: Tensor,    # [batch, M, T]
        driver_logit: Tensor,   # [batch, M, T]
    ) -> Tensor:
        """打包为 [batch, total_vars]"""
        batch_size = departure.shape[0]
        
        # 转换logits为概率 (优化在概率空间进行)
        dest_prob = F.softmax(dest_logits, dim=-1)      # [batch, M, T, Z]
        mode_prob = F.softmax(mode_logits, dim=-1)      # [batch, M, T, Modes]
        joint_prob = torch.sigmoid(joint_logit)         # [batch, M, T]
        driver_prob = torch.sigmoid(driver_logit)       # [batch, M, T]
        
        # 展平并拼接
        # 顺序: [departure, duration, dest_prob, mode_prob, joint_prob, driver_prob] for each member, tour
        packed = []
        for m in range(self.max_members):
            for t in range(self.max_tours):
                packed.append(departure[:, m, t:t+1])                    # [batch, 1]
                packed.append(duration[:, m, t:t+1])                     # [batch, 1]
                packed.append(dest_prob[:, m, t, :])                     # [batch, Z]
                packed.append(mode_prob[:, m, t, :])                     # [batch, Modes]
                packed.append(joint_prob[:, m, t:t+1])                   # [batch, 1]
                packed.append(driver_prob[:, m, t:t+1])                  # [batch, 1]
        
        return torch.cat(packed, dim=-1)  # [batch, total_vars]
    
    def _unpack_variables(self, x: Tensor) -> Tuple[Tensor, ...]:
        """解包为原始形状"""
        batch_size = x.shape[0]
        device = x.device
        
        departure = torch.zeros(batch_size, self.max_members, self.max_tours, device=device)
        duration = torch.zeros(batch_size, self.max_members, self.max_tours, device=device)
        dest_prob = torch.zeros(batch_size, self.max_members, self.max_tours, self.num_zones, device=device)
        mode_prob = torch.zeros(batch_size, self.max_members, self.max_tours, self.num_modes, device=device)
        joint_prob = torch.zeros(batch_size, self.max_members, self.max_tours, device=device)
        driver_prob = torch.zeros(batch_size, self.max_members, self.max_tours, device=device)
        
        idx = 0
        for m in range(self.max_members):
            for t in range(self.max_tours):
                departure[:, m, t] = x[:, idx]
                idx += 1
                duration[:, m, t] = x[:, idx]
                idx += 1
                dest_prob[:, m, t, :] = x[:, idx:idx+self.num_zones]
                idx += self.num_zones
                mode_prob[:, m, t, :] = x[:, idx:idx+self.num_modes]
                idx += self.num_modes
                joint_prob[:, m, t] = x[:, idx]
                idx += 1
                driver_prob[:, m, t] = x[:, idx]
                idx += 1
        
        # 概率转回logits
        eps = 1e-6
        dest_logits = torch.log(dest_prob.clamp(min=eps) / (1 - dest_prob.clamp(max=1-eps).sum(dim=-1, keepdim=True) + eps).clamp(min=eps))
        # 简化: 直接用log-ratio近似
        dest_logits = torch.log(dest_prob.clamp(min=eps))
        mode_logits = torch.log(mode_prob.clamp(min=eps))
        joint_logit = torch.log(joint_prob.clamp(min=eps) / (1 - joint_prob).clamp(min=eps))
        driver_logit = torch.log(driver_prob.clamp(min=eps) / (1 - driver_prob).clamp(min=eps))
        
        return departure, duration, dest_logits, mode_logits, joint_logit, driver_logit
    
    def _build_variable_mask(
        self,
        member_mask: Tensor,    # [batch, M]
        tour_mask: Tensor,      # [batch, M, T]
    ) -> Tensor:
        """构建变量级mask [batch, total_vars]"""
        batch_size = member_mask.shape[0]
        device = member_mask.device
        
        var_masks = []
        for m in range(self.max_members):
            for t in range(self.max_tours):
                # 该tour的mask
                t_mask = tour_mask[:, m, t:t+1]  # [batch, 1]
                
                # 扩展到该tour的所有变量
                var_masks.append(t_mask)                                    # departure
                var_masks.append(t_mask)                                    # duration
                var_masks.append(t_mask.expand(-1, self.num_zones))        # dest
                var_masks.append(t_mask.expand(-1, self.num_modes))        # mode
                var_masks.append(t_mask)                                    # joint
                var_masks.append(t_mask)                                    # driver
        
        return torch.cat(var_masks, dim=-1)  # [batch, total_vars]
    
    # =========================================================================
    #                          效用函数计算
    # =========================================================================
    
    def _compute_member_utility(
        self,
        departure: Tensor,          # [batch, T]
        duration: Tensor,           # [batch, T]
        dest_prob: Tensor,          # [batch, T, Z]
        mode_prob: Tensor,          # [batch, T, Modes]
        joint_prob: Tensor,         # [batch, T]
        driver_prob: Tensor,        # [batch, T]
        tour_mask: Tensor,          # [batch, T]
        is_adult: Tensor,           # [batch] bool
        home_zone: Tensor,          # [batch]
        travel_time: Tensor,        # [Z, Z, Modes] or [batch, Z, Z, Modes]
        pref_departure: Tensor,     # [batch, T]
        pref_duration: Tensor,      # [batch, T]
        flexibility: Tensor,        # [batch, T]
    ) -> Tensor:
        """
        计算单个成员的效用 (batch化)
        
        Returns:
            utility: [batch]
        """
        batch_size = departure.shape[0]
        device = departure.device
        
        # 基准效用
        U = torch.full((batch_size,), self.params.utility_baseline, device=device)
        
        # 有效tour数
        valid_tours = tour_mask.sum(dim=-1).clamp(min=1)  # [batch]
        
        # -----------------------------------------------------------------
        # 1. 位置效用: 偏好外出
        # -----------------------------------------------------------------
        # home_zone: [batch] -> [batch, 1, 1]
        home_idx = home_zone.long().view(-1, 1, 1).expand(-1, departure.shape[1], 1)
        at_home_prob = dest_prob.gather(-1, home_idx).squeeze(-1)  # [batch, T]
        
        U_location = self.params.alpha_social * ((1 - at_home_prob) * tour_mask).sum(dim=-1)
        
        # -----------------------------------------------------------------
        # 2. 联合活动效用
        # -----------------------------------------------------------------
        U_joint = self.params.alpha_joint * (joint_prob * tour_mask).sum(dim=-1)
        
        # -----------------------------------------------------------------
        # 3. 接送效用 (仅adult)
        # -----------------------------------------------------------------
        escort_duration = (driver_prob * duration * tour_mask).sum(dim=-1)
        U_escort = self.params.theta_escort * escort_duration * is_adult.float()
        
        # -----------------------------------------------------------------
        # 4. 时间偏差效用
        # -----------------------------------------------------------------
        time_deviation = (departure - pref_departure).pow(2)
        duration_deviation = (duration - pref_duration).pow(2)
        
        # 根据灵活性调整惩罚 (flexibility: 0=rigid, 1=moderate, 2=flexible)
        beta = 2.0 - flexibility * 0.75  # rigid:2.0, moderate:1.25, flexible:0.5
        
        U_timing = -self.params.beta_time * (beta * time_deviation * tour_mask).sum(dim=-1)
        U_duration = -self.params.gamma_duration * (duration_deviation * tour_mask).sum(dim=-1)
        
        # -----------------------------------------------------------------
        # 5. 出行效用
        # -----------------------------------------------------------------
        # 处理travel_time维度
        if travel_time.dim() == 3:
            # [Z, Z, M] -> [1, Z, Z, M]
            tt = travel_time.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            tt = travel_time  # [batch, Z, Z, M]
        
        # 从home出发的出行时间: [batch, Z, M]
        home_tt = tt[torch.arange(batch_size, device=device), home_zone.long(), :, :]
        
        # 期望出行时间: E[TT] = Σ_z P(z) * TT(home, z, m)
        # [batch, T, Z] @ [batch, Z, M] -> [batch, T, M]
        expected_tt = torch.einsum('btz,bzm->btm', dest_prob, home_tt)
        
        # 出行效用: Σ_m P(m) * θ_m * E[TT|m]
        # [batch, T, M] * [M] -> [batch, T]
        travel_utility = (mode_prob * expected_tt * self.theta_travel).sum(dim=-1)
        U_travel = (travel_utility * tour_mask).sum(dim=-1)
        
        # -----------------------------------------------------------------
        # 汇总
        # -----------------------------------------------------------------
        total_U = U + U_location + U_joint + U_escort + U_timing + U_duration + U_travel
        
        return total_U
    
    def _compute_household_utilities(
        self,
        x: Tensor,                  # [batch, total_vars]
        member_mask: Tensor,        # [batch, M]
        tour_mask: Tensor,          # [batch, M, T]
        home_zone: Tensor,          # [batch]
        member_is_adult: Tensor,    # [batch, M]
        travel_time: Tensor,
        pref_departure: Tensor,     # [batch, M, T]
        pref_duration: Tensor,
        flexibility: Tensor,
    ) -> Tensor:
        """
        计算所有成员效用
        
        Returns:
            utilities: [batch, M]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 解包变量
        departure, duration, dest_logits, mode_logits, joint_logit, driver_logit = self._unpack_variables(x)
        
        # 转为概率
        dest_prob = F.softmax(dest_logits, dim=-1)
        mode_prob = F.softmax(mode_logits, dim=-1)
        joint_prob = torch.sigmoid(joint_logit)
        driver_prob = torch.sigmoid(driver_logit)
        
        utilities = torch.zeros(batch_size, self.max_members, device=device)
        
        for m in range(self.max_members):
            U_m = self._compute_member_utility(
                departure=departure[:, m, :],
                duration=duration[:, m, :],
                dest_prob=dest_prob[:, m, :, :],
                mode_prob=mode_prob[:, m, :, :],
                joint_prob=joint_prob[:, m, :],
                driver_prob=driver_prob[:, m, :],
                tour_mask=tour_mask[:, m, :],
                is_adult=member_is_adult[:, m],
                home_zone=home_zone,
                travel_time=travel_time,
                pref_departure=pref_departure[:, m, :],
                pref_duration=pref_duration[:, m, :],
                flexibility=flexibility[:, m, :],
            )
            utilities[:, m] = U_m
        
        # 应用member_mask
        utilities = utilities * member_mask
        
        return utilities
    
    def _compute_nash_welfare(
        self,
        departure: Tensor,
        duration: Tensor,
        dest_logits: Tensor,
        mode_logits: Tensor,
        joint_logit: Tensor,
        driver_logit: Tensor,
        member_mask: Tensor,
        tour_mask: Tensor,
        home_zone: Tensor,
        member_is_adult: Tensor,
        travel_time: Tensor,
        pref_departure: Tensor,
        pref_duration: Tensor,
        flexibility: Tensor,
    ) -> Tensor:
        """计算Nash福利 Σ log(U_n)"""
        x = self._pack_variables(
            departure, duration, dest_logits, mode_logits, joint_logit, driver_logit
        )
        
        utilities = self._compute_household_utilities(
            x, member_mask, tour_mask, home_zone, member_is_adult,
            travel_time, pref_departure, pref_duration, flexibility
        )
        
        # Nash福利: Σ_n log(U_n), 只对有效成员
        log_U = torch.log(utilities.clamp(min=1e-6))
        nash_welfare = (log_U * member_mask).sum(dim=-1)  # [batch]
        
        return nash_welfare
    
    # =========================================================================
    #                          SQP 求解
    # =========================================================================
    
    def _sqp_solve(
        self,
        x0: Tensor,
        var_mask: Tensor,
        member_mask: Tensor,
        tour_mask: Tensor,
        num_vehicles: Tensor,
        home_zone: Tensor,
        member_is_adult: Tensor,
        travel_time_matrix: Tensor,
        preferred_departure: Tensor,
        preferred_duration: Tensor,
        activity_flexibility: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        SQP求解Nash Bargaining问题
        
        最后一步使用OptNet (QPFunction) 实现精确梯度反传
        """
        batch_size = x0.shape[0]
        n_vars = x0.shape[1]
        device = x0.device
        dtype = x0.dtype
        
        x = x0.clone()
        
        # BFGS Hessian初始化
        H = torch.eye(n_vars, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        prev_x = None
        prev_grad = None
        
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for iteration in range(self.sqp_max_iter):
            # 计算目标函数和梯度
            x_var = x.clone().requires_grad_(True)
            
            utilities = self._compute_household_utilities(
                x_var, member_mask, tour_mask, home_zone, member_is_adult,
                travel_time_matrix, preferred_departure, preferred_duration, activity_flexibility
            )
            
            # Nash目标: -Σ log(U_n) (最小化)
            log_U = torch.log(utilities.clamp(min=1e-6))
            nash_obj = -(log_U * member_mask).sum(dim=-1)  # [batch]
            
            # 梯度
            grad = torch.zeros(batch_size, n_vars, device=device, dtype=dtype)
            for b in range(batch_size):
                nash_obj[b].backward(retain_graph=True)
                grad[b] = x_var.grad[b].clone()
                x_var.grad.zero_()
            
            # 收敛检查
            grad_norm = (grad * var_mask).norm(dim=-1)
            converged = grad_norm < self.sqp_tol
            
            if converged.all():
                break
            
            # BFGS更新
            if prev_x is not None:
                s = x - prev_x  # [batch, n]
                y = grad - prev_grad  # [batch, n]
                
                # 逐样本BFGS
                for b in range(batch_size):
                    sy = torch.dot(s[b], y[b])
                    if sy > 1e-8:
                        rho = 1.0 / sy
                        I = torch.eye(n_vars, device=device, dtype=dtype)
                        s_b = s[b:b+1].T  # [n, 1]
                        y_b = y[b:b+1].T  # [n, 1]
                        
                        H[b] = (I - rho * s_b @ y_b.T) @ H[b] @ (I - rho * y_b @ s_b.T) + rho * s_b @ s_b.T
            
            prev_x = x.clone()
            prev_grad = grad.clone()
            
            # 最后一次迭代或倒数第二次: 使用OptNet
            if iteration == self.sqp_max_iter - 1 or (iteration >= self.sqp_max_iter - 2 and not converged.all()):
                x = self._optnet_qp_step(x, grad, H, var_mask, num_vehicles, member_is_adult, tour_mask)
            else:
                # 普通QP步
                x = self._simple_qp_step(x, grad, H, var_mask)
            
            # 投影到可行域
            x = self._project_to_feasible(x, var_mask)
        
        return x, converged
    
    def _simple_qp_step(
        self,
        x: Tensor,          # [batch, n]
        grad: Tensor,       # [batch, n]
        H: Tensor,          # [batch, n, n]
        var_mask: Tensor,   # [batch, n]
    ) -> Tensor:
        """简单QP步 (无约束，用于中间迭代)"""
        batch_size = x.shape[0]
        n = x.shape[1]
        device = x.device
        
        # 正则化Hessian
        H_reg = H + 1e-4 * torch.eye(n, device=device).unsqueeze(0)
        
        # 求解方向: d = -H^{-1} g
        try:
            d = -torch.linalg.solve(H_reg, grad.unsqueeze(-1)).squeeze(-1)
        except:
            d = -grad  # 退化为梯度下降
        
        # 线搜索步长
        alpha = 0.1
        
        # 更新 (只更新有效变量)
        x_new = x + alpha * d * var_mask
        
        return x_new
    
    def _optnet_qp_step(
        self,
        x: Tensor,              # [batch, n]
        grad: Tensor,           # [batch, n]
        H: Tensor,              # [batch, n, n]
        var_mask: Tensor,       # [batch, n]
        num_vehicles: Tensor,   # [batch]
        member_is_adult: Tensor,# [batch, M]
        tour_mask: Tensor,      # [batch, M, T]
    ) -> Tensor:
        """
        使用OptNet (qpth) 求解QP子问题
        
        min  0.5 * d^T H d + grad^T d
        s.t. G d <= h
             A d = b
             lb <= x + d <= ub
        """
        batch_size = x.shape[0]
        n = x.shape[1]
        device = x.device
        dtype = x.dtype
        
        # 正则化Hessian确保正定
        Q = H + 1e-4 * torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
        p = grad
        
        # 构建约束
        G, h, A, b = self._build_qp_constraints(x, var_mask, num_vehicles, member_is_adult, tour_mask)
        
        if QPTH_AVAILABLE and Q.shape[0] > 0:
            # 使用qpth求解
            try:
                # QPFunction期望: Q[batch,n,n], p[batch,n], G[batch,m,n], h[batch,m], A[batch,k,n], b[batch,k]
                d = QPFunction(verbose=False, maxIter=100, eps=1e-6)(
                    Q.double(), p.double(), 
                    G.double(), h.double(), 
                    A.double(), b.double()
                ).float()
            except Exception as e:
                print(f"QPFunction failed: {e}, using fallback")
                d = self._fallback_qp_solve(Q, p, G, h, A, b)
        else:
            d = self._fallback_qp_solve(Q, p, G, h, A, b)
        
        # 步长限制
        alpha = torch.clamp(torch.ones(batch_size, 1, device=device), max=1.0)
        
        x_new = x + alpha * d * var_mask
        
        return x_new
    
    def _build_qp_constraints(
        self,
        x: Tensor,              # [batch, n]
        var_mask: Tensor,       # [batch, n]
        num_vehicles: Tensor,   # [batch]
        member_is_adult: Tensor,# [batch, M]
        tour_mask: Tensor,      # [batch, M, T]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        构建QP约束矩阵
        
        不等式约束 Gd <= h:
            1. 边界约束: -d <= x (即 x + d >= 0)
                        d <= 1 - x (即 x + d <= 1)
            2. 车辆约束: Σ(driver_prob * car_mode_prob) <= num_vehicles
        
        等式约束 Ad = b:
            1. 概率归一化 (dest, mode概率和为1)
        """
        batch_size = x.shape[0]
        n = x.shape[1]
        device = x.device
        dtype = x.dtype
        
        # -----------------------------------------------------------------
        # 不等式约束
        # -----------------------------------------------------------------
        G_list = []
        h_list = []
        
        # 1. 边界约束: 0 <= x + d <= 1
        # -d <= x  =>  -I @ d <= x
        G_list.append(-torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1))
        h_list.append(x)
        
        # d <= 1 - x  =>  I @ d <= 1 - x
        G_list.append(torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1))
        h_list.append(1 - x)
        
        # 2. 车辆约束 (简化版)
        # 这里省略复杂的车辆约束构建，保持代码简洁
        
        G = torch.cat(G_list, dim=1)  # [batch, m, n]
        h = torch.cat(h_list, dim=1)  # [batch, m]
        
        # -----------------------------------------------------------------
        # 等式约束 (概率归一化)
        # -----------------------------------------------------------------
        # 简化: 不显式约束，依赖投影
        A = torch.zeros(batch_size, 1, n, device=device, dtype=dtype)
        b = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        
        return G, h, A, b
    
    def _fallback_qp_solve(
        self,
        Q: Tensor,  # [batch, n, n]
        p: Tensor,  # [batch, n]
        G: Tensor,  # [batch, m, n]
        h: Tensor,  # [batch, m]
        A: Tensor,  # [batch, k, n]
        b: Tensor,  # [batch, k]
    ) -> Tensor:
        """备选QP求解 (简单梯度投影)"""
        # 无约束解: d = -Q^{-1} p
        try:
            d = -torch.linalg.solve(Q, p.unsqueeze(-1)).squeeze(-1)
        except:
            d = -p / (torch.diagonal(Q, dim1=-2, dim2=-1) + 1e-6)
        
        # 投影到约束
        # Gd <= h => 如果违反则截断
        violations = (G @ d.unsqueeze(-1)).squeeze(-1) - h  # [batch, m]
        max_violation = violations.max(dim=-1, keepdim=True)[0].clamp(min=0)
        
        # 简单缩放
        if max_violation.max() > 0:
            scale = 1.0 / (1.0 + max_violation)
            d = d * scale
        
        return d
    
    def _project_to_feasible(self, x: Tensor, var_mask: Tensor) -> Tensor:
        """投影到可行域"""
        batch_size = x.shape[0]
        device = x.device
        
        # 解包
        departure, duration, dest_logits, mode_logits, joint_logit, driver_logit = self._unpack_variables(x)
        
        # 边界投影
        departure = torch.clamp(departure, 0, 1)
        duration = torch.clamp(duration, 0, 0.5)
        
        # 重新打包 (softmax/sigmoid自动处理概率约束)
        # 将logits转为概率再打包
        dest_prob = F.softmax(dest_logits, dim=-1)
        mode_prob = F.softmax(mode_logits, dim=-1)
        joint_prob = torch.sigmoid(joint_logit)
        driver_prob = torch.sigmoid(driver_logit)
        
        # 边界投影概率
        dest_prob = torch.clamp(dest_prob, 1e-6, 1-1e-6)
        dest_prob = dest_prob / dest_prob.sum(dim=-1, keepdim=True)
        
        mode_prob = torch.clamp(mode_prob, 1e-6, 1-1e-6)
        mode_prob = mode_prob / mode_prob.sum(dim=-1, keepdim=True)
        
        joint_prob = torch.clamp(joint_prob, 1e-6, 1-1e-6)
        driver_prob = torch.clamp(driver_prob, 1e-6, 1-1e-6)
        
        # 重新打包
        packed = []
        for m in range(self.max_members):
            for t in range(self.max_tours):
                packed.append(departure[:, m, t:t+1])
                packed.append(duration[:, m, t:t+1])
                packed.append(dest_prob[:, m, t, :])
                packed.append(mode_prob[:, m, t, :])
                packed.append(joint_prob[:, m, t:t+1])
                packed.append(driver_prob[:, m, t:t+1])
        
        return torch.cat(packed, dim=-1)


# =============================================================================
#                    包装类：便于集成到现有模型
# =============================================================================

class HouseholdNashBargainingLayer(nn.Module):
    """
    家庭Nash Bargaining协调层 - 便于集成的包装类
    
    使用方式:
    ```python
    # 初始化
    nash_layer = HouseholdNashBargainingLayer(
        num_zones=2006,
        num_modes=4,
        max_members=6,
        max_tours=8,
    )
    
    # 前向传播 (在decoder之后)
    coordinated = nash_layer(
        departure_time=decoder_out['departure'],
        duration=decoder_out['duration'],
        destination_logits=decoder_out['destination'],
        mode_logits=decoder_out['mode'],
        is_joint_logit=decoder_out['is_joint'],
        is_driver_logit=decoder_out['is_driver'],
        member_mask=batch['member_mask'],
        tour_mask=batch['tour_mask'],
        num_vehicles=batch['num_vehicles'],
        home_zone=batch['home_zone'],
        member_is_adult=batch['is_adult'],
        travel_time_matrix=self.travel_time_matrix,
    )
    
    # 使用协调后的输出计算原有损失
    loss = criterion(
        pred_departure=coordinated.departure_time,
        pred_destination=coordinated.destination_logits,
        pred_mode=coordinated.mode_logits,
        target_departure=batch['gt_departure'],
        target_destination=batch['gt_destination'],
        target_mode=batch['gt_mode'],
        mask=batch['tour_mask'],
    )
    
    # 反向传播 (梯度自动通过Nash层传回decoder)
    loss.backward()
    ```
    """
    
    def __init__(
        self,
        num_zones: int,
        num_modes: int = 4,
        max_members: int = 6,
        max_tours: int = 8,
        utility_params: Optional[UtilityParams] = None,
        sqp_max_iter: int = 15,
        sqp_tol: float = 1e-4,
        enabled: bool = True,
    ):
        super().__init__()
        
        self.enabled = enabled
        
        self.nash_qp_layer = NashBargainingQPLayer(
            num_zones=num_zones,
            num_modes=num_modes,
            max_members=max_members,
            max_tours=max_tours,
            utility_params=utility_params,
            sqp_max_iter=sqp_max_iter,
            sqp_tol=sqp_tol,
        )
    
    def forward(
        self,
        departure_time: Tensor,
        duration: Tensor,
        destination_logits: Tensor,
        mode_logits: Tensor,
        is_joint_logit: Tensor,
        is_driver_logit: Tensor,
        member_mask: Tensor,
        tour_mask: Tensor,
        num_vehicles: Tensor,
        home_zone: Tensor,
        member_is_adult: Tensor,
        travel_time_matrix: Tensor,
        preferred_departure: Optional[Tensor] = None,
        preferred_duration: Optional[Tensor] = None,
        activity_flexibility: Optional[Tensor] = None,
    ) -> NashBargainingOutput:
        """
        前向传播
        
        如果 enabled=False，直接返回输入（不进行协调）
        """
        if not self.enabled:
            return NashBargainingOutput(
                departure_time=departure_time,
                duration=duration,
                destination_logits=destination_logits,
                mode_logits=mode_logits,
                is_joint_logit=is_joint_logit,
                is_driver_logit=is_driver_logit,
                nash_welfare=torch.zeros(departure_time.shape[0], device=departure_time.device),
                converged=torch.ones(departure_time.shape[0], dtype=torch.bool, device=departure_time.device),
            )
        
        return self.nash_qp_layer(
            departure_time=departure_time,
            duration=duration,
            destination_logits=destination_logits,
            mode_logits=mode_logits,
            is_joint_logit=is_joint_logit,
            is_driver_logit=is_driver_logit,
            member_mask=member_mask,
            tour_mask=tour_mask,
            num_vehicles=num_vehicles,
            home_zone=home_zone,
            member_is_adult=member_is_adult,
            travel_time_matrix=travel_time_matrix,
            preferred_departure=preferred_departure,
            preferred_duration=preferred_duration,
            activity_flexibility=activity_flexibility,
        )
    
    def set_enabled(self, enabled: bool):
        """动态开启/关闭协调层"""
        self.enabled = enabled


# =============================================================================
#                    工具函数
# =============================================================================

def create_nash_layer_from_config(config: dict) -> HouseholdNashBargainingLayer:
    """从配置创建Nash层"""
    utility_params = UtilityParams(
        alpha_joint=config.get('alpha_joint', 0.1),
        alpha_social=config.get('alpha_social', 0.3),
        theta_escort=config.get('theta_escort', -0.58),
        theta_car=config.get('theta_car', -1.0),
        theta_pt=config.get('theta_pt', -0.4),
        beta_time=config.get('beta_time', 1.0),
        gamma_duration=config.get('gamma_duration', 1.0),
        utility_baseline=config.get('utility_baseline', 10.0),
    )
    
    return HouseholdNashBargainingLayer(
        num_zones=config['num_zones'],
        num_modes=config.get('num_modes', 4),
        max_members=config.get('max_members', 6),
        max_tours=config.get('max_tours', 8),
        utility_params=utility_params,
        sqp_max_iter=config.get('sqp_max_iter', 15),
        sqp_tol=config.get('sqp_tol', 1e-4),
        enabled=config.get('enabled', True),
    )