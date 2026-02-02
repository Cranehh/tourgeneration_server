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
            self.params.theta_walk,
            self.params.theta_car,
            self.params.theta_car,
            self.params.theta_pt,
            self.params.theta_pt,
            self.params.theta_bike,
            self.params.theta_bike,
            self.params.theta_car,
            self.params.theta_car,
            self.params.theta_car,
            self.params.theta_car
        ][:num_modes]))
        
        # 变量维度计算
        # 每个成员每个tour: departure(1) + duration(1) + dest(Z) + mode(M) + joint(1) + driver(1)
        self.vars_per_tour = 2 + 1 + num_modes + 2  # departure + duration + at_home + mode + joint + driver
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

        # 从目的地logits提取在家概率作为初始值
        dest_prob = F.softmax(destination_logits, dim=-1)  # [batch, M, T, Z]
        home_idx = home_zone.long().view(-1, 1, 1, 1).expand(-1, self.max_members, self.max_tours, 1)
        at_home_prob_init = dest_prob.gather(-1, home_idx).squeeze(-1)  # [batch, M, T]

        # 预计算外出时的期望TT (固定参数)
        expected_tt_out = self._precompute_expected_tt_out(
            dest_prob, home_zone, travel_time_matrix
        )  # [batch, M, T, num_modes]

        # 打包变量时用 at_home_prob 替代 destination_logits
        x0 = self._pack_variables(
            departure_time, duration, at_home_prob_init,  # 改这里
            mode_logits, is_joint_logit, is_driver_logit
        )
        
        # 构建mask向量
        var_mask = self._build_variable_mask(member_mask, tour_mask)  # [batch, total_vars]
        
        # SQP迭代求解
        x_opt, converged = self._sqp_solve(
            x0=x0,
            var_mask=var_mask,
            member_mask=member_mask,
            tour_mask=tour_mask,
            num_vehicles=num_vehicles,
            member_is_adult=member_is_adult,
            expected_tt_out=expected_tt_out,  # 新增
            pref_departure=preferred_departure,
            pref_duration=preferred_duration,
            pref_at_home=at_home_prob_init,  # 新增：用初始在家概率作为偏好
            flexibility=activity_flexibility,
        )

        # 解包协调后的变量
        out_departure, out_duration, out_at_home, out_mode_logits, out_joint_logit, out_driver_logit = \
            self._unpack_variables(x_opt)

        # 根据协调后的在家概率更新目的地logits
        out_dest_logits = self._update_destination_logits(
            destination_logits, out_at_home, home_zone
        )
        
        # 应用mask (无效位置保持原值)
        full_mask = tour_mask.unsqueeze(-1)  # [batch, max_members, max_tours, 1]
        
        out_departure = torch.where(tour_mask.bool(), out_departure, departure_time)
        out_duration = torch.where(tour_mask.bool(), out_duration, duration)
        out_dest_logits = torch.where(
            tour_mask.unsqueeze(-1).bool(), out_dest_logits, destination_logits
        )
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
            destination_logits=out_dest_logits,  # 更新后的目的地
            mode_logits=out_mode_logits,
            is_joint_logit=out_joint_logit,
            is_driver_logit=out_driver_logit,
            nash_welfare=nash_welfare,
            converged=converged,
        )
    
    # =========================================================================
    #                          变量打包/解包
    # =========================================================================
    def _precompute_expected_tt_out(
            self,
            dest_prob: Tensor,  # [batch, M, T, Z]
            home_zone: Tensor,  # [batch]
            travel_time_matrix: Tensor,  # [Z, Z, num_modes]
    ) -> Tensor:
        """
        预计算外出时的期望出行时间
        E[TT | out, m] = Σ_z P(z) * TT(home, z, m) / (1 - P(home))
        """
        batch_size = dest_prob.shape[0]
        device = dest_prob.device

        # [batch, Z, num_modes]
        if travel_time_matrix.dim() == 3:
            home_tt = travel_time_matrix[home_zone.long(), :, :]
        else:
            home_tt = travel_time_matrix[torch.arange(batch_size, device=device), home_zone.long(), :, :]

        # 在家概率 [batch, M, T]
        home_idx = home_zone.long().view(-1, 1, 1, 1).expand(-1, self.max_members, self.max_tours, 1)
        at_home_prob = dest_prob.gather(-1, home_idx).squeeze(-1)
        out_prob = (1 - at_home_prob).clamp(min=1e-6)

        # 期望TT (含在家): [batch, M, T, Z] @ [batch, Z, M] -> [batch, M, T, M]
        expected_tt_all = torch.einsum('bntz,bzm->bntm', dest_prob, home_tt)

        # 条件期望 (外出时)
        expected_tt_out = expected_tt_all / out_prob.unsqueeze(-1)

        return expected_tt_out

    def _pack_variables(
            self,
            departure: Tensor,  # [batch, M, T]
            duration: Tensor,  # [batch, M, T]
            at_home_prob: Tensor,  # [batch, M, T]  # 改：替代 dest_logits
            mode_logits: Tensor,  # [batch, M, T, num_modes]
            joint_logit: Tensor,  # [batch, M, T]
            driver_logit: Tensor,  # [batch, M, T]
    ) -> Tensor:
        """打包为 [batch, total_vars]"""

        mode_prob = F.softmax(mode_logits, dim=-1)
        joint_prob = torch.sigmoid(joint_logit)
        driver_prob = torch.sigmoid(driver_logit)

        # 拼接为 [batch, M, T, vars_per_tour]
        packed = torch.cat([
            departure.unsqueeze(-1),  # [batch, M, T, 1]
            duration.unsqueeze(-1),  # [batch, M, T, 1]
            at_home_prob.unsqueeze(-1),  # [batch, M, T, 1]
            mode_prob,  # [batch, M, T, num_modes]
            joint_prob.unsqueeze(-1),  # [batch, M, T, 1]
            driver_prob.unsqueeze(-1),  # [batch, M, T, 1]
        ], dim=-1)

        # 展平为 [batch, M * T * vars_per_tour]
        return packed.view(packed.shape[0], -1)

    def _unpack_variables(self, x: Tensor) -> Tuple[Tensor, ...]:
        """解包为原始形状"""
        batch_size = x.shape[0]

        # 总变量布局: [M * T * vars_per_tour]
        # vars_per_tour = 2 + 1 + num_modes + 2 = 5 + num_modes

        # 重塑为 [batch, M, T, vars_per_tour]
        x_reshaped = x.view(batch_size, self.max_members, self.max_tours, self.vars_per_tour)

        # 切片提取各变量
        idx = 0
        departure = x_reshaped[..., idx]  # [batch, M, T]
        idx += 1
        duration = x_reshaped[..., idx]  # [batch, M, T]
        idx += 1
        at_home_prob = x_reshaped[..., idx]  # [batch, M, T]
        idx += 1
        mode_prob = x_reshaped[..., idx:idx + self.num_modes]  # [batch, M, T, num_modes]
        idx += self.num_modes
        joint_prob = x_reshaped[..., idx]  # [batch, M, T]
        idx += 1
        driver_prob = x_reshaped[..., idx]  # [batch, M, T]

        # 转回logits
        eps = 1e-6
        mode_logits = torch.log(mode_prob.clamp(min=eps))
        joint_logit = torch.log(joint_prob.clamp(min=eps) / (1 - joint_prob).clamp(min=eps))
        driver_logit = torch.log(driver_prob.clamp(min=eps) / (1 - driver_prob).clamp(min=eps))

        return departure, duration, at_home_prob, mode_logits, joint_logit, driver_logit

    def _build_variable_mask(
            self,
            member_mask: Tensor,
            tour_mask: Tensor,
    ) -> Tensor:
        """构建变量级mask [batch, total_vars]"""
        batch_size = member_mask.shape[0]
        device = member_mask.device

        var_masks = []
        for m in range(self.max_members):
            for t in range(self.max_tours):
                t_mask = tour_mask[:, m, t:t + 1]

                var_masks.append(t_mask)  # departure
                var_masks.append(t_mask)  # duration
                var_masks.append(t_mask)  # at_home  # 改
                var_masks.append(t_mask.expand(-1, self.num_modes))  # mode
                var_masks.append(t_mask)  # joint
                var_masks.append(t_mask)  # driver

        return torch.cat(var_masks, dim=-1)
    
    # =========================================================================
    #                          效用函数计算
    # =========================================================================

    def _compute_member_utility(
            self,
            departure: Tensor,  # [batch, T]
            duration: Tensor,  # [batch, T]
            at_home_prob: Tensor,  # [batch, T]  # 改：替代 dest_prob
            mode_prob: Tensor,  # [batch, T, Modes]
            joint_prob: Tensor,  # [batch, T]
            driver_prob: Tensor,  # [batch, T]
            tour_mask: Tensor,  # [batch, T]
            is_adult: Tensor,  # [batch]
            expected_tt_out: Tensor,  # [batch, T, num_modes]  # 改：预计算的外出期望TT
            pref_departure: Tensor,
            pref_duration: Tensor,
            pref_at_home: Tensor,  # [batch, T]  # 改：NN输出的在家概率
            flexibility: Tensor,
    ) -> Tensor:
        batch_size = departure.shape[0]
        device = departure.device

        U = torch.full((batch_size,), self.params.utility_baseline, device=device)
        valid_tours = tour_mask.sum(dim=-1).clamp(min=1)

        # 1. 位置效用：偏好外出
        out_of_home_prob = 1 - at_home_prob  # [batch, T]
        U_location = self.params.alpha_social * (out_of_home_prob * tour_mask).sum(dim=-1)

        # 2. 联合活动效用
        U_joint = self.params.alpha_joint * (joint_prob * tour_mask).sum(dim=-1)

        # 3. 接送效用
        escort_duration = (driver_prob * duration * tour_mask).sum(dim=-1)
        U_escort = self.params.theta_escort * escort_duration * is_adult.float()

        # 4. 时间偏差效用
        time_deviation = (departure - pref_departure).pow(2)
        beta = 2.0 - flexibility * 0.75
        U_timing = -self.params.beta_time * (beta * time_deviation * tour_mask).sum(dim=-1)

        # 5. 持续时间偏差效用
        duration_deviation = (duration - pref_duration).pow(2)
        U_duration = -self.params.gamma_duration * (duration_deviation * tour_mask).sum(dim=-1)

        # 6. 在家概率偏差效用（鼓励接近NN输出）
        at_home_deviation = (at_home_prob - pref_at_home).pow(2)
        U_at_home_dev = -0.5 * (at_home_deviation * tour_mask).sum(dim=-1)

        # 7. 出行效用：外出时才有出行
        # [batch, T, M] * [M] -> [batch, T]
        travel_utility_per_tour = (mode_prob * expected_tt_out * self.theta_travel).sum(dim=-1)
        U_travel = (out_of_home_prob * travel_utility_per_tour * tour_mask).sum(dim=-1)

        return U + U_location + U_joint + U_escort + U_timing + U_duration + U_at_home_dev + U_travel

    def _compute_household_utilities(
            self,
            x: Tensor,
            member_mask: Tensor,
            tour_mask: Tensor,
            member_is_adult: Tensor,
            expected_tt_out: Tensor,  # [batch, M, T, num_modes]  # 新增
            pref_departure: Tensor,
            pref_duration: Tensor,
            pref_at_home: Tensor,  # [batch, M, T]  # 新增
            flexibility: Tensor,
    ) -> Tensor:
        batch_size = x.shape[0]
        device = x.device

        # 解包
        departure, duration, at_home_prob, mode_logits, joint_logit, driver_logit = self._unpack_variables(x)

        mode_prob = F.softmax(mode_logits, dim=-1)
        joint_prob = torch.sigmoid(joint_logit)
        driver_prob = torch.sigmoid(driver_logit)

        utilities = torch.zeros(batch_size, self.max_members, device=device)

        for m in range(self.max_members):
            U_m = self._compute_member_utility(
                departure=departure[:, m, :],
                duration=duration[:, m, :],
                at_home_prob=at_home_prob[:, m, :],  # 改
                mode_prob=mode_prob[:, m, :, :],
                joint_prob=joint_prob[:, m, :],
                driver_prob=driver_prob[:, m, :],
                tour_mask=tour_mask[:, m, :],
                is_adult=member_is_adult[:, m],
                expected_tt_out=expected_tt_out[:, m, :, :],  # 改
                pref_departure=pref_departure[:, m, :],
                pref_duration=pref_duration[:, m, :],
                pref_at_home=pref_at_home[:, m, :],  # 改
                flexibility=flexibility[:, m, :],
            )
            utilities[:, m] = U_m

        return utilities * member_mask

    def _compute_nash_welfare(
            self,
            departure: Tensor,
            duration: Tensor,
            dest_logits: Tensor,  # 保留，用于提取 at_home
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

        # 从 dest_logits 提取 at_home_prob
        dest_prob = F.softmax(dest_logits, dim=-1)
        home_idx = home_zone.long().view(-1, 1, 1, 1).expand(-1, self.max_members, self.max_tours, 1)
        at_home_prob = dest_prob.gather(-1, home_idx).squeeze(-1)

        # 预计算 expected_tt_out
        expected_tt_out = self._precompute_expected_tt_out(dest_prob, home_zone, travel_time)

        # 打包变量
        x = self._pack_variables(
            departure, duration, at_home_prob, mode_logits, joint_logit, driver_logit
        )

        # 计算效用
        utilities = self._compute_household_utilities(
            x, member_mask, tour_mask, member_is_adult,
            expected_tt_out, pref_departure, pref_duration, at_home_prob, flexibility
        )

        # Nash福利
        log_U = torch.log(utilities.clamp(min=1e-6))
        nash_welfare = (log_U * member_mask).sum(dim=-1)

        return nash_welfare
    
    # =========================================================================
    #                          SQP 求解
    # =========================================================================
    def _update_destination_logits(
            self,
            original_dest_logits: Tensor,  # [batch, M, T, Z]
            coordinated_at_home: Tensor,  # [batch, M, T]
            home_zone: Tensor,  # [batch]
    ) -> Tensor:
        """根据协调后的在家概率更新目的地logits"""
        dest_prob = F.softmax(original_dest_logits, dim=-1)

        home_idx = home_zone.long().view(-1, 1, 1, 1).expand(
            -1, self.max_members, self.max_tours, 1
        )
        original_at_home = dest_prob.gather(-1, home_idx).squeeze(-1)

        # 缩放外出目的地概率
        scale = (1 - coordinated_at_home) / (1 - original_at_home).clamp(min=1e-6)

        new_prob = dest_prob * scale.unsqueeze(-1)
        new_prob.scatter_(-1, home_idx, coordinated_at_home.unsqueeze(-1))
        new_prob = new_prob / new_prob.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return torch.log(new_prob.clamp(min=1e-6))

    def _sqp_solve(
            self,
            x0: Tensor,
            var_mask: Tensor,
            member_mask: Tensor,
            tour_mask: Tensor,
            num_vehicles: Tensor,
            member_is_adult: Tensor,
            expected_tt_out: Tensor,
            pref_departure: Tensor,
            pref_duration: Tensor,
            pref_at_home: Tensor,
            flexibility: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        SQP求解Nash Bargaining问题
        最后一步必须经过OptNet以保证梯度反传
        """
        batch_size = x0.shape[0]
        n_vars = x0.shape[1]
        device = x0.device
        dtype = x0.dtype

        x = x0.clone().detach()  # 断开与输入的计算图
        H = torch.eye(n_vars, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()

        prev_x = None
        prev_grad = None
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # =========================================================================
        # SQP 迭代（无需保持计算图）
        # =========================================================================

        with torch.no_grad():  # 关键：中间迭代不需要梯度
            for iteration in range(self.sqp_max_iter):
                # 临时开启梯度计算（仅用于计算当前梯度值）
                x_var = x.clone().requires_grad_(True)

                with torch.enable_grad():
                    utilities = self._compute_household_utilities(
                        x_var, member_mask, tour_mask, member_is_adult,
                        expected_tt_out, pref_departure, pref_duration, pref_at_home, flexibility
                    )

                    log_U = torch.log(utilities.clamp(min=1e-6))
                    nash_obj = -(log_U * member_mask).sum(dim=-1)

                    # 计算梯度（不保留计算图）
                    grad = torch.autograd.grad(nash_obj.sum(), x_var, create_graph=False)[0]

                # 显式释放中间变量
                del utilities, log_U, nash_obj, x_var

                # 转为纯数值（断开计算图）
                grad = grad.detach()

                # 收敛检查
                grad_norm = (grad * var_mask).norm(dim=-1)
                converged = grad_norm < self.sqp_tol

                if converged.all():
                    break

                # BFGS 更新（阻尼版本）
                if prev_x is not None:
                    s = x - prev_x
                    y = grad - prev_grad

                    for b in range(batch_size):
                        s_b = s[b]
                        y_b = y[b]

                        # ====== 前置检查 ======
                        # 检查输入是否有效
                        if torch.isnan(s_b).any() or torch.isinf(s_b).any():
                            continue
                        if torch.isnan(y_b).any() or torch.isinf(y_b).any():
                            continue
                        if torch.isnan(H[b]).any() or torch.isinf(H[b]).any():
                            H[b] = torch.eye(n_vars, device=device, dtype=dtype)
                            continue

                        # 检查步长是否太小（说明已收敛）或太大（说明不稳定）
                        s_norm = s_b.norm()
                        if s_norm < 1e-12 or s_norm > 1e6:
                            continue

                        # ====== 计算关键量 ======
                        sy = torch.dot(s_b, y_b)
                        Hs = H[b] @ s_b
                        sHs = torch.dot(s_b, Hs)

                        # 检查 sHs 是否有效
                        if torch.isnan(sHs) or torch.isinf(sHs) or sHs < 1e-12:
                            continue

                        # ====== Powell's damping ======
                        if sy < 0.2 * sHs:
                            theta = 0.8 * sHs / (sHs - sy + 1e-10)  # 加小量防除零
                            theta = torch.clamp(theta, 0.0, 1.0)  # 确保在[0,1]
                            y_b = theta * y_b + (1 - theta) * Hs
                            sy = torch.dot(s_b, y_b)

                        # ====== 最终安全检查 ======
                        if sy < 1e-10 or torch.isnan(sy) or torch.isinf(sy):
                            continue

                        rho = 1.0 / sy

                        # 限制 rho 防止爆炸
                        if rho > 1e8:
                            continue

                        # ====== BFGS 更新 ======
                        I = torch.eye(n_vars, device=device, dtype=dtype)
                        s_col = s_b.unsqueeze(1)
                        y_col = y_b.unsqueeze(1)

                        H_new = (I - rho * s_col @ y_col.T) @ H[b] @ (I - rho * y_col @ s_col.T) + rho * s_col @ s_col.T

                        # 更新后检查
                        if torch.isnan(H_new).any() or torch.isinf(H_new).any():
                            print(f"Warning: batch {b}, H became nan/inf after update, resetting")
                            print(f'number_nans in H: {torch.isnan(H_new).sum().item()}')
                            print(f'number_infs in H: {torch.isinf(H_new).sum().item()}')
                            H[b] = I
                            continue

                        H[b] = H_new

                prev_x = x.clone()
                prev_grad = grad.clone()

                # 普通 QP 步
                x = self._simple_qp_step(x, grad, H, var_mask)
                x = self._project_to_feasible(x, var_mask)

        # =========================================================================
        # 最终 OptNet 步：建立计算图，保证梯度反传
        # =========================================================================
        # 这里需要计算图，不用 no_grad
        x_var = x.clone().requires_grad_(True)

        utilities = self._compute_household_utilities(
            x_var, member_mask, tour_mask, member_is_adult,
            expected_tt_out, pref_departure, pref_duration, pref_at_home, flexibility
        )

        log_U = torch.log(utilities.clamp(min=1e-6))
        nash_obj = -(log_U * member_mask).sum(dim=-1)

        # create_graph=True 保持计算图（用于反向传播）
        final_grad = torch.autograd.grad(nash_obj.sum(), x_var, create_graph=True)[0]

        # OptNet QP 步（带梯度）
        x_final = self._optnet_qp_step(
            x, final_grad, H, var_mask, num_vehicles, member_is_adult, tour_mask
        )

        x_final = self._project_to_feasible(x_final, var_mask)

        return x_final, converged
    
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

        # 解包 - 注意返回的是 at_home_prob
        departure, duration, at_home_prob, mode_logits, joint_logit, driver_logit = self._unpack_variables(x)

        # 边界投影
        departure = torch.clamp(departure, 0, 1)
        duration = torch.clamp(duration, 0, 0.5)
        at_home_prob = torch.clamp(at_home_prob, 1e-6, 1 - 1e-6)  # 概率边界

        # mode 概率归一化
        mode_prob = F.softmax(mode_logits, dim=-1)
        mode_prob = torch.clamp(mode_prob, 1e-6, 1 - 1e-6)
        mode_prob = mode_prob / mode_prob.sum(dim=-1, keepdim=True)

        joint_prob = torch.sigmoid(joint_logit)
        driver_prob = torch.sigmoid(driver_logit)
        joint_prob = torch.clamp(joint_prob, 1e-6, 1 - 1e-6)
        driver_prob = torch.clamp(driver_prob, 1e-6, 1 - 1e-6)

        # 重新打包
        packed = []
        for m in range(self.max_members):
            for t in range(self.max_tours):
                packed.append(departure[:, m, t:t + 1])
                packed.append(duration[:, m, t:t + 1])
                packed.append(at_home_prob[:, m, t:t + 1])  # 修改：at_home_prob
                packed.append(mode_prob[:, m, t, :])
                packed.append(joint_prob[:, m, t:t + 1])
                packed.append(driver_prob[:, m, t:t + 1])

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