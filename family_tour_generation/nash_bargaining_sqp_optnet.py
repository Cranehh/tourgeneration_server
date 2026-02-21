"""
家庭层面 Nash Bargaining 协调层 (重新设计版 V3)
基于 SQP + OptNet 的可微优化实现

修复问题:
1. SQP迭代过程中加入完整约束条件
2. 确保OptNet的QP参数(Q,p,G,h,A,b)显式依赖神经网络输出θ，保证梯度可传播
5. 完整构建约束：概率归一化、车辆约束、边界约束
6. 效用函数与讨论的数学模型一致

核心设计原则:
- SQP中间迭代: 无需保持计算图，用于找到好的收敛点
- 最后一步OptNet: 基于收敛点x_k和原始神经网络输出θ，重新构建QP参数
- QP参数(Q,p,A,b,G,h)显式依赖θ，OptNet通过KKT条件的隐函数定理计算dx*/dθ

Author: 郝赫
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import math

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
    alpha_social: float = 0.3       # 外出奖励 (偏好外出而非在家)
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
    departure_time: Tensor          # [batch, max_members, max_tours]
    duration: Tensor                # [batch, max_members, max_tours]
    destination_logits: Tensor      # [batch, max_members, max_tours, num_zones]
    mode_logits: Tensor             # [batch, max_members, max_tours, num_modes]
    is_joint_logit: Tensor          # [batch, max_members, max_tours]
    is_driver_logit: Tensor         # [batch, max_members, max_tours]
    nash_welfare: Tensor            # [batch] Nash社会福利
    converged: Tensor               # [batch] 是否收敛


# =============================================================================
#                          核心：Nash Bargaining QP Layer
# =============================================================================

class NashBargainingQPLayer(nn.Module):
    """
    Nash Bargaining 协调层 (重新设计版)

    关键改进:
    1. SQP迭代中包含完整约束
    2. 最后一步OptNet的QP参数显式依赖神经网络输出θ
    3. 完整的约束构建
    """

    def __init__(
        self,
        num_zones: int,
        num_modes: int = 4,
        max_members: int = 6,
        max_tours: int = 8,
        utility_params: Optional[UtilityParams] = None,
        sqp_max_iter: int = 10,
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

        # 出行时间系数 (按mode索引)
        self.register_buffer('theta_travel', torch.tensor([
          self.params.theta_walk,
          self.params.theta_car,
          self.params.theta_car,
          self.params.theta_pt,
          self.params.theta_walk,
          self.params.theta_pt,
          self.params.theta_bike,
          self.params.theta_bike,
          self.params.theta_car,
          self.params.theta_car,
          self.params.theta_car,
          self.params.theta_car
      ][:num_modes]))

        # 变量布局：每个tour的变量
        # [departure(1), duration(1), mode_prob(M), joint_prob(1), driver_prob(1)]
        # 注意：目的地单独处理(维度太高)，通过at_home_prob简化
        self.vars_per_tour = 2 + num_modes + 2  # departure + duration + mode + joint + driver
        self.total_vars_per_member = max_tours * self.vars_per_tour
        self.total_vars = max_members * self.total_vars_per_member

    def forward(
        self,
        # 神经网络输出 (θ)
        departure_time: Tensor,         # [batch, max_members, max_tours]
        duration: Tensor,               # [batch, max_members, max_tours]
        destination_logits: Tensor,     # [batch, max_members, max_tours, num_zones]
        mode_logits: Tensor,            # [batch, max_members, max_tours, num_modes]
        is_joint_logit: Tensor,         # [batch, max_members, max_tours]
        is_driver_logit: Tensor,        # [batch, max_members, max_tours]
        # Mask
        member_mask: Tensor,            # [batch, max_members]
        tour_mask: Tensor,              # [batch, max_members, max_tours]
        # 家庭信息
        num_vehicles: Tensor,           # [batch]
        home_zone: Tensor,              # [batch]
        member_is_adult: Tensor,        # [batch, max_members]
        # 出行时间矩阵
        travel_time_matrix: Tensor,     # [num_zones, num_zones, num_modes] 或 [batch, ...]
        # 偏好 (可选)
        preferred_departure: Optional[Tensor] = None,
        preferred_duration: Optional[Tensor] = None,
        activity_flexibility: Optional[Tensor] = None,
    ) -> NashBargainingOutput:
        """前向传播"""
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

        # 预计算从家到各zone的期望出行时间 [batch, num_zones, num_modes]
        expected_tt = self._get_home_to_zone_tt(home_zone, travel_time_matrix)

        # 打包神经网络输出为初始点
        x0 = self._pack_variables(
            departure_time, duration, mode_logits, is_joint_logit, is_driver_logit
        )

        # 构建变量mask
        var_mask = self._build_variable_mask(member_mask, tour_mask)

        # =====================================================================
        # 阶段1: SQP迭代找到收敛点 (无需计算图)
        # =====================================================================
        x_converged = self._sqp_iterations(
            x0=x0,
            var_mask=var_mask,
            member_mask=member_mask,
            tour_mask=tour_mask,
            num_vehicles=num_vehicles,
            member_is_adult=member_is_adult,
            expected_tt=expected_tt,
            destination_logits=destination_logits.detach(),  # 目的地固定
            home_zone=home_zone,
            pref_departure=preferred_departure,
            pref_duration=preferred_duration,
            flexibility=activity_flexibility,
        )

        # =====================================================================
        # 阶段2: 在收敛点构建最终QP，使用OptNet求解并反传梯度
        # =====================================================================
        x_final, converged = self._final_optnet_step(
            x_converged=x_converged,
            theta_departure=departure_time,      # 原始神经网络输出 (保持计算图)
            theta_duration=duration,
            theta_mode_logits=mode_logits,
            theta_joint_logit=is_joint_logit,
            theta_driver_logit=is_driver_logit,
            theta_destination_logits=destination_logits,
            var_mask=var_mask,
            member_mask=member_mask,
            tour_mask=tour_mask,
            num_vehicles=num_vehicles,
            member_is_adult=member_is_adult,
            expected_tt=expected_tt,
            home_zone=home_zone,
            pref_departure=preferred_departure,
            pref_duration=preferred_duration,
            flexibility=activity_flexibility,
        )

        # 解包
        out_departure, out_duration, out_mode_logits, out_joint_logit, out_driver_logit = \
            self._unpack_variables(x_final)

        # 应用mask
        out_departure = torch.where(tour_mask.bool(), out_departure, departure_time)
        out_duration = torch.where(tour_mask.bool(), out_duration, duration)
        out_mode_logits = torch.where(
            tour_mask.unsqueeze(-1).expand_as(mode_logits).bool(),
            out_mode_logits, mode_logits
        )
        out_joint_logit = torch.where(tour_mask.bool(), out_joint_logit, is_joint_logit)
        out_driver_logit = torch.where(tour_mask.bool(), out_driver_logit, is_driver_logit)

        # 计算Nash福利
        nash_welfare = self._compute_nash_welfare(
            out_departure, out_duration, destination_logits, out_mode_logits,
            out_joint_logit, out_driver_logit,
            member_mask, tour_mask, home_zone, member_is_adult,
            expected_tt, preferred_departure, preferred_duration, activity_flexibility
        )

        return NashBargainingOutput(
            departure_time=out_departure,
            duration=out_duration,
            destination_logits=destination_logits,  # 目的地不变
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
        mode_logits: Tensor,    # [batch, M, T, num_modes]
        joint_logit: Tensor,    # [batch, M, T]
        driver_logit: Tensor,   # [batch, M, T]
    ) -> Tensor:
        """打包为 [batch, total_vars]"""
        batch_size = departure.shape[0]

        # 转为概率
        mode_prob = F.softmax(mode_logits, dim=-1)
        joint_prob = torch.sigmoid(joint_logit)
        driver_prob = torch.sigmoid(driver_logit)

        # 拼接 [batch, M, T, vars_per_tour]
        packed = torch.cat([
            departure.unsqueeze(-1),
            duration.unsqueeze(-1),
            mode_prob,
            joint_prob.unsqueeze(-1),
            driver_prob.unsqueeze(-1),
        ], dim=-1)

        return packed.view(batch_size, -1)

    def _unpack_variables(self, x: Tensor) -> Tuple[Tensor, ...]:
        """解包为原始形状"""
        batch_size = x.shape[0]

        # [batch, M, T, vars_per_tour]
        x_reshaped = x.view(batch_size, self.max_members, self.max_tours, self.vars_per_tour)

        idx = 0
        departure = x_reshaped[..., idx]; idx += 1
        duration = x_reshaped[..., idx]; idx += 1
        mode_prob = x_reshaped[..., idx:idx+self.num_modes]; idx += self.num_modes
        joint_prob = x_reshaped[..., idx]; idx += 1
        driver_prob = x_reshaped[..., idx]

        # 转回logits
        eps = 1e-6
        mode_logits = torch.log(mode_prob.clamp(min=eps))
        joint_logit = torch.log(joint_prob.clamp(min=eps) / (1 - joint_prob).clamp(min=eps))
        driver_logit = torch.log(driver_prob.clamp(min=eps) / (1 - driver_prob).clamp(min=eps))

        return departure, duration, mode_logits, joint_logit, driver_logit

    def _build_variable_mask(self, member_mask: Tensor, tour_mask: Tensor) -> Tensor:
        """构建变量级mask [batch, total_vars]"""
        batch_size = member_mask.shape[0]
        device = member_mask.device

        # [batch, M, T] -> [batch, M, T, vars_per_tour] -> [batch, total_vars]
        mask_expanded = tour_mask.unsqueeze(-1).expand(-1, -1, -1, self.vars_per_tour)
        return mask_expanded.reshape(batch_size, -1)

    def _get_home_to_zone_tt(self, home_zone: Tensor, travel_time_matrix: Tensor) -> Tensor:
        """获取从家到各zone的出行时间 [batch, num_zones, num_modes]"""
        batch_size = home_zone.shape[0]

        if travel_time_matrix.dim() == 3:
            # [Z, Z, M] -> 选取 [home, :, :]
            return travel_time_matrix[home_zone.long(), :, :]
        else:
            # [batch, Z, Z, M]
            batch_idx = torch.arange(batch_size, device=home_zone.device)
            return travel_time_matrix[batch_idx, home_zone.long(), :, :]

    # =========================================================================
    #                          效用函数计算
    # =========================================================================

    def _compute_member_utility(
        self,
        departure: Tensor,      # [batch, T]
        duration: Tensor,       # [batch, T]
        mode_prob: Tensor,      # [batch, T, M]
        joint_prob: Tensor,     # [batch, T]
        driver_prob: Tensor,    # [batch, T]
        tour_mask: Tensor,      # [batch, T]
        is_adult: Tensor,       # [batch]
        dest_prob: Tensor,      # [batch, T, Z]
        home_zone: Tensor,      # [batch]
        expected_tt: Tensor,    # [batch, Z, M]
        pref_departure: Tensor,
        pref_duration: Tensor,
        flexibility: Tensor,
    ) -> Tensor:
        """计算单个成员的效用"""
        batch_size = departure.shape[0]
        device = departure.device

        # 基准效用
        U = torch.full((batch_size,), self.params.utility_baseline, device=device)

        # 1. 位置效用：偏好外出
        home_idx = home_zone.long().view(-1, 1, 1).expand(-1, self.max_tours, 1)
        at_home_prob = dest_prob.gather(-1, home_idx).squeeze(-1)  # [batch, T]
        out_of_home_prob = 1 - at_home_prob
        U_location = self.params.alpha_social * (out_of_home_prob * tour_mask).sum(dim=-1)

        # 2. 联合活动效用
        U_joint = self.params.alpha_joint * (joint_prob * tour_mask).sum(dim=-1)

        # 3. 接送效用 (成人)
        escort_duration = (driver_prob * duration * tour_mask).sum(dim=-1)
        U_escort = self.params.theta_escort * escort_duration * is_adult.float()

        # 4. 时间偏差效用
        time_deviation = (departure - pref_departure).pow(2)
        beta = 2.0 - flexibility * 0.75  # 灵活性越高惩罚越小
        U_timing = -self.params.beta_time * (beta * time_deviation * tour_mask).sum(dim=-1)

        # 5. 持续时间偏差效用
        duration_deviation = (duration - pref_duration).pow(2)
        U_duration = -self.params.gamma_duration * (duration_deviation * tour_mask).sum(dim=-1)

        # 6. 出行效用
        # 期望出行时间: E[TT|外出] = Σ_z Σ_m dest_prob[z] * mode_prob[m] * TT[home,z,m]
        # 简化：对于外出的tour，计算加权出行时间
        # [batch, T, Z] @ [batch, Z, M] -> [batch, T, M]
        expected_tt_per_tour = torch.einsum('btz,bzm->btm', dest_prob, expected_tt)
        # [batch, T, M] * [batch, T, M] * [M] -> [batch, T]
        travel_utility_per_tour = (mode_prob * expected_tt_per_tour * self.theta_travel).sum(dim=-1)
        # 只有外出才有出行
        U_travel = (out_of_home_prob * travel_utility_per_tour * tour_mask).sum(dim=-1)

        return U + U_location + U_joint + U_escort + U_timing + U_duration + U_travel

    def _compute_household_utilities(
        self,
        x: Tensor,                  # [batch, total_vars]
        member_mask: Tensor,
        tour_mask: Tensor,
        member_is_adult: Tensor,
        dest_logits: Tensor,        # [batch, M, T, Z]
        home_zone: Tensor,
        expected_tt: Tensor,
        pref_departure: Tensor,
        pref_duration: Tensor,
        flexibility: Tensor,
    ) -> Tensor:
        """计算家庭所有成员的效用 [batch, max_members]"""
        batch_size = x.shape[0]
        device = x.device

        departure, duration, mode_logits, joint_logit, driver_logit = self._unpack_variables(x)
        mode_prob = F.softmax(mode_logits, dim=-1)
        joint_prob = torch.sigmoid(joint_logit)
        driver_prob = torch.sigmoid(driver_logit)
        dest_prob = F.softmax(dest_logits, dim=-1)

        utilities = torch.zeros(batch_size, self.max_members, device=device)

        for m in range(self.max_members):
            U_m = self._compute_member_utility(
                departure=departure[:, m, :],
                duration=duration[:, m, :],
                mode_prob=mode_prob[:, m, :, :],
                joint_prob=joint_prob[:, m, :],
                driver_prob=driver_prob[:, m, :],
                tour_mask=tour_mask[:, m, :],
                is_adult=member_is_adult[:, m],
                dest_prob=dest_prob[:, m, :, :],
                home_zone=home_zone,
                expected_tt=expected_tt,
                pref_departure=pref_departure[:, m, :],
                pref_duration=pref_duration[:, m, :],
                flexibility=flexibility[:, m, :],
            )
            utilities[:, m] = U_m

        return utilities * member_mask

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
        expected_tt: Tensor,
        pref_departure: Tensor,
        pref_duration: Tensor,
        flexibility: Tensor,
    ) -> Tensor:
        """计算Nash福利 Σ log(U_n)"""
        x = self._pack_variables(departure, duration, mode_logits, joint_logit, driver_logit)
        utilities = self._compute_household_utilities(
            x, member_mask, tour_mask, member_is_adult,
            dest_logits, home_zone, expected_tt,
            pref_departure, pref_duration, flexibility
        )
        log_U = torch.log(utilities.clamp(min=1e-6))
        return (log_U * member_mask).sum(dim=-1)

    # =========================================================================
    #                          约束构建
    # =========================================================================

    def _build_constraints(
        self,
        x: Tensor,              # [batch, n] 当前点
        var_mask: Tensor,       # [batch, n]
        member_mask: Tensor,    # [batch, M]
        tour_mask: Tensor,      # [batch, M, T]
        num_vehicles: Tensor,   # [batch]
        member_is_adult: Tensor,# [batch, M]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        构建QP约束

        不等式约束 Gd <= h:
            1. 边界约束: 0 <= x + d <= 1 (针对概率变量)
            2. 车辆约束: Σ(driver_prob * car_mode_prob) <= num_vehicles

        等式约束 Ad = b:
            1. mode概率归一化: Σ mode_prob = 1

        Returns:
            G: [batch, num_ineq, n]
            h: [batch, num_ineq]
            A: [batch, num_eq, n]
            b: [batch, num_eq]
        """
        batch_size = x.shape[0]
        n = x.shape[1]
        device = x.device
        dtype = x.dtype

        G_list = []
        h_list = []
        A_list = []
        b_list = []

        # ---------------------------------------------------------------------
        # 不等式约束1: 边界约束 0 <= x + d <= 1
        # 等价于: -x <= d 且 d <= 1 - x
        # ---------------------------------------------------------------------

        # 下界: -d <= x => -I @ d <= x
        G_lower = -torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        h_lower = x.clone()

        # 上界: d <= 1 - x => I @ d <= 1 - x
        G_upper = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        h_upper = 1 - x

        # 对于时间变量(departure, duration)，设置更合理的边界
        # departure: [0, 1], duration: [0, 0.5]
        for m in range(self.max_members):
            for t in range(self.max_tours):
                base_idx = m * self.total_vars_per_member + t * self.vars_per_tour
                # duration的上界是0.5而非1
                duration_idx = base_idx + 1
                h_upper[:, duration_idx] = 10 - x[:, duration_idx]

                h_upper[:, base_idx] = (24-12.94601186)/ 4.69132532 - x[:, base_idx]

                h_lower[:, base_idx] = x[:, base_idx] - (1 - 12.94601186) / 4.69132532

        G_list.extend([G_lower, G_upper])
        h_list.extend([h_lower, h_upper])

        # ---------------------------------------------------------------------
        # 不等式约束2: 车辆约束
        # Σ_{m是成人} Σ_t driver_prob[m,t] * car_mode_prob[m,t] <= num_vehicles
        #
        # 线性化: 在当前点展开双线性项
        # f(d,c) ≈ f(d0,c0) + (∂f/∂d)(d-d0) + (∂f/∂c)(c-c0)
        # = d0*c0 + c0*(d-d0) + d0*(c-c0)
        # = c0*d + d0*c - d0*c0
        #
        # 约束: Σ (c0*Δd + d0*Δc) <= num_vehicles - Σ d0*c0
        # ---------------------------------------------------------------------

        # 解包当前点
        x_reshaped = x.view(batch_size, self.max_members, self.max_tours, self.vars_per_tour)
        driver_prob_0 = x_reshaped[..., -1]  # [batch, M, T]
        car_mode_idx = 7  # 假设car是第3个mode (index=2)
        car_prob_0 = x_reshaped[..., 2 + car_mode_idx]  # [batch, M, T]

        G_vehicle = torch.zeros(batch_size, 1, n, device=device, dtype=dtype)

        for m in range(self.max_members):
            for t in range(self.max_tours):
                base_idx = m * self.total_vars_per_member + t * self.vars_per_tour
                driver_idx = base_idx + 2 + self.num_modes + 1  # driver_prob位置
                car_idx = base_idx + 2 + car_mode_idx  # car_mode_prob位置

                # 系数: c0 * ∂/∂d + d0 * ∂/∂c
                c0 = car_prob_0[:, m, t]      # [batch]
                d0 = driver_prob_0[:, m, t]   # [batch]
                adult_mask = member_is_adult[:, m].float()  # [batch]
                tour_valid = tour_mask[:, m, t].float()     # [batch]

                G_vehicle[:, 0, driver_idx] = c0 * adult_mask * tour_valid
                G_vehicle[:, 0, car_idx] = d0 * adult_mask * tour_valid

        # RHS: num_vehicles - Σ d0*c0
        current_vehicle_usage = (driver_prob_0 * car_prob_0 * member_is_adult.unsqueeze(-1).float() * tour_mask).sum(dim=(1, 2))
        h_vehicle = (num_vehicles.float() - current_vehicle_usage).unsqueeze(-1)

        G_list.append(G_vehicle)
        h_list.append(h_vehicle)

        # ---------------------------------------------------------------------
        # 等式约束: mode概率归一化
        # Σ_m mode_prob[m] = 1 对每个有效的(member, tour)
        #
        # 由于初始点已经满足归一化，我们约束增量也满足: Σ Δmode_prob = 0
        # ---------------------------------------------------------------------

        num_eq = self.max_members * self.max_tours
        A_mode = torch.zeros(batch_size, num_eq, n, device=device, dtype=dtype)
        b_mode = torch.zeros(batch_size, num_eq, device=device, dtype=dtype)

        eq_idx = 0
        for m in range(self.max_members):
            for t in range(self.max_tours):
                base_idx = m * self.total_vars_per_member + t * self.vars_per_tour
                mode_start = base_idx + 2  # mode概率起始位置

                # Σ mode_prob = 1 => Σ Δmode_prob = 0
                for mode_i in range(self.num_modes):
                    A_mode[:, eq_idx, mode_start + mode_i] = 1.0

                # b = 0 (因为是增量约束)
                b_mode[:, eq_idx] = 0.0
                eq_idx += 1

        A_list.append(A_mode)
        b_list.append(b_mode)

        # 合并约束
        G = torch.cat(G_list, dim=1)
        h = torch.cat(h_list, dim=1)
        A = torch.cat(A_list, dim=1) if A_list else torch.zeros(batch_size, 0, n, device=device, dtype=dtype)
        b = torch.cat(b_list, dim=1) if b_list else torch.zeros(batch_size, 0, device=device, dtype=dtype)

        return G, h, A, b

    # =========================================================================
    #                          SQP迭代 (阶段1)
    # =========================================================================

    def _sqp_iterations(
        self,
        x0: Tensor,
        var_mask: Tensor,
        member_mask: Tensor,
        tour_mask: Tensor,
        num_vehicles: Tensor,
        member_is_adult: Tensor,
        expected_tt: Tensor,
        destination_logits: Tensor,
        home_zone: Tensor,
        pref_departure: Tensor,
        pref_duration: Tensor,
        flexibility: Tensor,
    ) -> Tensor:
        """
        SQP迭代找到收敛点

        这一阶段不需要保持计算图，目的是找到一个好的收敛点
        """
        batch_size = x0.shape[0]
        n = x0.shape[1]
        device = x0.device
        dtype = x0.dtype

        x = x0.clone().detach()
        H = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()

        prev_x = None
        prev_grad = None

        with torch.no_grad():
            for iteration in range(self.sqp_max_iter):
                # 计算梯度
                x_var = x.clone().requires_grad_(True)

                with torch.enable_grad():
                    utilities = self._compute_household_utilities(
                        x_var, member_mask, tour_mask, member_is_adult,
                        destination_logits, home_zone, expected_tt,
                        pref_departure, pref_duration, flexibility
                    )
                    log_U = torch.log(utilities.clamp(min=1e-6))
                    nash_obj = -(log_U * member_mask).sum(dim=-1)
                    grad = torch.autograd.grad(nash_obj.sum(), x_var)[0]

                grad = grad.detach()

                # 收敛检查
                grad_norm = (grad * var_mask).norm(dim=-1)
                if (grad_norm < self.sqp_tol).all():
                    break

                # BFGS更新Hessian近似
                if prev_x is not None:
                    s = x - prev_x
                    y = grad - prev_grad
                    H = self._bfgs_update(H, s, y)

                prev_x = x.clone()
                prev_grad = grad.clone()

                # 构建约束
                G, h, A, b = self._build_constraints(
                    x, var_mask, member_mask, tour_mask, num_vehicles, member_is_adult
                )

                # 求解带约束的QP子问题
                d = self._solve_qp_subproblem(H, grad, G, h, A, b, var_mask)

                # 线搜索
                alpha = self._line_search(
                    x, d, var_mask, member_mask, tour_mask, member_is_adult,
                    destination_logits, home_zone, expected_tt,
                    pref_departure, pref_duration, flexibility
                )

                # 更新
                x = x + alpha * d * var_mask
                x = self._project_to_feasible(x)

        return x

    def _bfgs_update(self, H: Tensor, s: Tensor, y: Tensor) -> Tensor:
        """BFGS更新Hessian近似"""
        batch_size = H.shape[0]
        n = H.shape[1]
        device = H.device
        dtype = H.dtype

        H_new = H.clone()

        for b in range(batch_size):
            s_b = s[b]
            y_b = y[b]

            # ====== 前置检查：输入有效性 ======
            if torch.isnan(s_b).any() or torch.isinf(s_b).any():
                continue
            if torch.isnan(y_b).any() or torch.isinf(y_b).any():
                continue
            if torch.isnan(H[b]).any() or torch.isinf(H[b]).any():
                H_new[b] = torch.eye(n, device=device, dtype=dtype)
                continue

            # 检查步长是否太小（说明已收敛）或太大（不稳定）
            s_norm = s_b.norm()
            y_norm = y_b.norm()

            if s_norm < 1e-12 or s_norm > 1e6:
                continue
            if y_norm < 1e-12 or y_norm > 1e6:
                continue

            # ====== 计算关键量 ======
            sy = torch.dot(s_b, y_b)
            Hs = H[b] @ s_b
            sHs = torch.dot(s_b, Hs)

            # 检查 sHs 有效性
            if torch.isnan(sHs) or torch.isinf(sHs) or sHs < 1e-12:
                continue

            # ====== Powell's damping ======
            if sy < 0.2 * sHs:
                theta = 0.8 * sHs / (sHs - sy + 1e-10)
                theta = torch.clamp(theta, 0.0, 1.0)
                y_b = theta * y_b + (1 - theta) * Hs
                sy = torch.dot(s_b, y_b)

            # ====== 关键修复：sy 的下界要足够大 ======
            # 确保 rho = 1/sy 不会太大
            min_sy = 1e-6  # 对应 rho_max = 1e6
            if sy < min_sy:
                continue

            # 额外检查：rho 上界
            rho = 1.0 / sy
            rho_max = 1e6  # 限制 rho 的最大值
            if rho > rho_max:
                continue

            # ====== BFGS 更新 ======
            I = torch.eye(n, device=device, dtype=dtype)
            s_col = s_b.unsqueeze(1)
            y_col = y_b.unsqueeze(1)

            # 计算更新项
            # H_new = (I - rho*s*y') * H * (I - rho*y*s') + rho*s*s'

            left_factor = I - rho * s_col @ y_col.T
            right_factor = I - rho * y_col @ s_col.T
            rank_one_term = rho * s_col @ s_col.T

            H_candidate = left_factor @ H[b] @ right_factor + rank_one_term

            # ====== 后置检查：更新结果有效性 ======
            if torch.isnan(H_candidate).any() or torch.isinf(H_candidate).any():
                # print(f"Warning: Invalid Hessian approximation in BFGS update, resetting to identity in batch {b}.")
                # print(f"Number of nans: {torch.isnan(H_candidate).sum().item()}")
                # print(f"Number of infs: {torch.isinf(H_candidate).sum().item()}")
                continue

            # 检查对角线是否为正（Hessian应该正定）
            diag = torch.diag(H_candidate)
            if (diag <= 0).any():
                # 更新后不正定，保持原值
                continue

            # 检查数值范围是否合理
            if H_candidate.abs().max() > 1e8:
                # 数值太大，保持原值
                continue

            H_new[b] = H_candidate

        return H_new

    def _solve_qp_subproblem(
            self,
            H: Tensor,  # [batch, n, n]
            grad: Tensor,  # [batch, n]
            G: Tensor,  # [batch, m, n]
            h: Tensor,  # [batch, m]
            A: Tensor,  # [batch, k, n]
            b: Tensor,  # [batch, k]
            var_mask: Tensor,
    ) -> Tensor:
        """
        求解QP子问题 (SQP中间迭代用)

        注意：这里不需要梯度传播，使用纯数值方法即可
        OptNet只在最后一步使用
        """
        batch_size = H.shape[0]
        n = H.shape[1]
        device = H.device
        dtype = H.dtype

        # 正则化确保正定
        H_reg = H + 1e-3 * torch.eye(n, device=device, dtype=dtype).unsqueeze(0)

        # 方案1：无约束牛顿步 + 约束投影（简单高效）
        try:
            d = -torch.linalg.solve(H_reg, grad.unsqueeze(-1)).squeeze(-1)
        except:
            # 退化为对角近似
            d = -grad / (torch.diagonal(H_reg, dim1=-2, dim2=-1) + 1e-6)

        # 约束投影
        d = self._project_direction_to_constraints(d, G, h, A, b)

        return d

    def _project_direction_to_constraints(
            self,
            d: Tensor,  # [batch, n]
            G: Tensor,  # [batch, m, n]
            h: Tensor,  # [batch, m]
            A: Tensor,  # [batch, k, n]
            b: Tensor,  # [batch, k]
    ) -> Tensor:
        """将搜索方向投影到约束可行域"""
        batch_size = d.shape[0]
        device = d.device

        # 处理等式约束 Ad = b
        if A.shape[1] > 0:
            Ad = (A @ d.unsqueeze(-1)).squeeze(-1)
            violation = Ad - b
            try:
                AAT = A @ A.transpose(-1, -2)
                reg = 1e-6 * torch.eye(A.shape[1], device=device)
                AAT_inv_viol = torch.linalg.solve(AAT + reg, violation.unsqueeze(-1))
                correction = (A.transpose(-1, -2) @ AAT_inv_viol).squeeze(-1)
                d = d - correction
            except:
                pass

        # 处理不等式约束 Gd <= h
        Gd = (G @ d.unsqueeze(-1)).squeeze(-1)  # [batch, m]

        # 只关心 Gd > 0 且 Gd > h 的约束（即会违反的约束）
        # 对于 Gd <= 0 的约束，无论 alpha 多大都满足（因为 alpha >= 0）
        # 对于 Gd > 0 的约束，需要 alpha <= h / Gd

        # 计算每个约束的最大允许步长
        # alpha_i = h_i / Gd_i (当 Gd_i > 0 时)
        # alpha_i = inf        (当 Gd_i <= 0 时，约束不限制步长)

        eps = 1e-8

        # 标记哪些约束是"活跃"的（Gd > eps）
        active = Gd > eps  # [batch, m]

        # 对于活跃约束，计算 alpha = h / Gd
        # 对于非活跃约束，设为一个大值（不限制步长）
        alpha_per_constraint = torch.where(
            active,
            h / Gd.clamp(min=eps),  # 活跃约束
            torch.full_like(h, 1e6)  # 非活跃约束，设为大值
        )

        # 只考虑正的 alpha（负的 alpha 意味着约束在反方向，不相关）
        alpha_per_constraint = torch.where(
            alpha_per_constraint > 0,
            alpha_per_constraint,
            torch.full_like(alpha_per_constraint, 1e6)
        )

        # 取最小的 alpha（最严格的约束）
        alpha_max, _ = alpha_per_constraint.min(dim=-1, keepdim=True)  # [batch, 1]

        # 限制 alpha 在合理范围内
        alpha_max = alpha_max.clamp(min=0.01, max=1.0)

        # 检查NaN并处理
        alpha_max = torch.where(
            torch.isnan(alpha_max) | torch.isinf(alpha_max),
            torch.ones_like(alpha_max) * 0.1,  # 出问题时用保守步长
            alpha_max
        )

        # 只有当存在违反时才缩放
        has_violation = (Gd > h + eps).any(dim=-1, keepdim=True)  # [batch, 1]
        d = torch.where(has_violation, d * alpha_max, d)

        return d

    def _simple_constraint_projection(
        self,
        d: Tensor,
        G: Tensor,
        h: Tensor,
        A: Tensor,
        b: Tensor,
    ) -> Tensor:
        """简单的约束投影"""
        # 检查不等式约束违反
        Gd = (G @ d.unsqueeze(-1)).squeeze(-1)
        violations = (Gd - h).clamp(min=0)
        max_violation = violations.max(dim=-1, keepdim=True)[0]

        # 如果违反，缩放步长
        if max_violation.max() > 1e-6:
            scale = 1.0 / (1.0 + max_violation)
            d = d * scale

        return d

    def _line_search(
        self,
        x: Tensor,
        d: Tensor,
        var_mask: Tensor,
        member_mask: Tensor,
        tour_mask: Tensor,
        member_is_adult: Tensor,
        destination_logits: Tensor,
        home_zone: Tensor,
        expected_tt: Tensor,
        pref_departure: Tensor,
        pref_duration: Tensor,
        flexibility: Tensor,
    ) -> Tensor:
        """简单的回溯线搜索"""
        batch_size = x.shape[0]
        device = x.device

        # 计算当前目标值
        utilities = self._compute_household_utilities(
            x, member_mask, tour_mask, member_is_adult,
            destination_logits, home_zone, expected_tt,
            pref_departure, pref_duration, flexibility
        )
        log_U = torch.log(utilities.clamp(min=1e-6))
        f0 = -(log_U * member_mask).sum(dim=-1)

        alpha = torch.ones(batch_size, device=device)

        for _ in range(10):
            x_new = x + alpha.unsqueeze(-1) * d * var_mask
            x_new = self._project_to_feasible(x_new)

            utilities_new = self._compute_household_utilities(
                x_new, member_mask, tour_mask, member_is_adult,
                destination_logits, home_zone, expected_tt,
                pref_departure, pref_duration, flexibility
            )
            log_U_new = torch.log(utilities_new.clamp(min=1e-6))
            f_new = -(log_U_new * member_mask).sum(dim=-1)

            # Armijo条件
            improved = f_new < f0 - 1e-4 * alpha * (d * var_mask).pow(2).sum(dim=-1)

            if improved.all():
                break

            alpha = torch.where(improved, alpha, alpha * 0.5)

        return alpha.unsqueeze(-1)

    def _project_to_feasible(self, x: Tensor) -> Tensor:
        """投影到可行域"""
        batch_size = x.shape[0]

        x_reshaped = x.view(batch_size, self.max_members, self.max_tours, self.vars_per_tour)

        # 边界投影
        departure = x_reshaped[..., 0].clamp(0, 1)
        duration = x_reshaped[..., 1].clamp(0, 0.5)

        # mode概率归一化
        mode_prob = x_reshaped[..., 2:2+self.num_modes]
        mode_prob = mode_prob.clamp(min=1e-6)
        mode_prob = mode_prob / mode_prob.sum(dim=-1, keepdim=True)

        # joint/driver概率边界
        joint_prob = x_reshaped[..., 2+self.num_modes].clamp(1e-6, 1-1e-6)
        driver_prob = x_reshaped[..., 2+self.num_modes+1].clamp(1e-6, 1-1e-6)

        # 重新组装
        x_projected = torch.cat([
            departure.unsqueeze(-1),
            duration.unsqueeze(-1),
            mode_prob,
            joint_prob.unsqueeze(-1),
            driver_prob.unsqueeze(-1),
        ], dim=-1)

        return x_projected.view(batch_size, -1)

    # =========================================================================
    #                          最终OptNet步 (阶段2)
    # =========================================================================

    def _final_optnet_step(
        self,
        x_converged: Tensor,            # SQP收敛点 (detached)
        theta_departure: Tensor,        # 原始神经网络输出 (保持计算图)
        theta_duration: Tensor,
        theta_mode_logits: Tensor,
        theta_joint_logit: Tensor,
        theta_driver_logit: Tensor,
        theta_destination_logits: Tensor,
        var_mask: Tensor,
        member_mask: Tensor,
        tour_mask: Tensor,
        num_vehicles: Tensor,
        member_is_adult: Tensor,
        expected_tt: Tensor,
        home_zone: Tensor,
        pref_departure: Tensor,
        pref_duration: Tensor,
        flexibility: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        最终OptNet步：基于收敛点和原始θ构建QP，实现梯度传播

        关键设计：
        - Q, p 显式依赖 θ (神经网络输出)
        - G, h, A, b 基于收敛点构建约束
        - OptNet通过KKT条件计算 dx*/dθ
        """
        batch_size = x_converged.shape[0]
        n = x_converged.shape[1]
        device = x_converged.device
        dtype = x_converged.dtype

        # -----------------------------------------------------------------
        # 构建QP参数，使其显式依赖θ
        #
        # 目标函数: min -Σ log(U_n)
        # 在收敛点x*做二阶展开:
        #   f(x) ≈ f(x*) + g^T(x-x*) + 0.5*(x-x*)^T H (x-x*)
        #
        # 令 d = x - x*，QP子问题:
        #   min 0.5 * d^T Q d + p^T d
        # 其中:
        #   Q = ∂²f/∂x² (Hessian)
        #   p = ∂f/∂x   (梯度)
        #
        # 关键：p 和 Q 需要依赖 θ
        # -----------------------------------------------------------------

        # 打包θ为向量 (保持计算图)
        theta_x = self._pack_variables(
            theta_departure, theta_duration,
            theta_mode_logits, theta_joint_logit, theta_driver_logit
        )

        # 计算在收敛点的梯度和Hessian，但表达式中包含θ
        #
        # 设计：效用函数 U_n(x; θ) 包含偏好项 -β||x - θ||²
        # 这使得 ∂U/∂θ ≠ 0，从而 p 依赖 θ

        Q, p = self._build_qp_objective_with_theta(
            x_converged=x_converged,
            theta_x=theta_x,
            theta_destination_logits=theta_destination_logits,
            member_mask=member_mask,
            tour_mask=tour_mask,
            member_is_adult=member_is_adult,
            expected_tt=expected_tt,
            home_zone=home_zone,
            flexibility=flexibility,
        )

        # 构建约束 (基于收敛点)
        G, h, A, b = self._build_constraints(
            x_converged, var_mask, member_mask, tour_mask, num_vehicles, member_is_adult
        )

        # 正则化Q确保正定
        Q = Q + 1e-3 * torch.eye(n, device=device, dtype=dtype).unsqueeze(0)

        assert not torch.isnan(Q).any(), "Q contains NaN"
        assert not torch.isnan(p).any(), "p contains NaN"

        # OptNet求解
        converged = torch.ones(batch_size, dtype=torch.bool, device=device)

        if QPTH_AVAILABLE:
            try:
                d = QPFunction(verbose=False, maxIter=100, eps=1e-6)(
                    Q.double(), p.double(),
                    G.double(), h.double(),
                    A.double(), b.double()
                ).float()
            except Exception as e:
                print(f"OptNet failed: {e}, using fallback")
                d = self._fallback_qp_solve(Q, p, G, h, A, b)
                converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            d = self._fallback_qp_solve(Q, p, G, h, A, b)

        # x_final = x_converged + d
        # 但我们需要保持与θ的连接
        # 技巧：x_final = theta_x + (x_converged - theta_x).detach() + d
        # 这样 x_final 依赖 theta_x 和 d (后者通过OptNet依赖θ)

        x_final = theta_x + (x_converged - theta_x.detach()) + d
        x_final = self._project_to_feasible(x_final)

        return x_final, converged

    def _build_qp_objective_with_theta(
        self,
        x_converged: Tensor,            # [batch, n]
        theta_x: Tensor,                # [batch, n] 打包后的神经网络输出
        theta_destination_logits: Tensor,
        member_mask: Tensor,
        tour_mask: Tensor,
        member_is_adult: Tensor,
        expected_tt: Tensor,
        home_zone: Tensor,
        flexibility: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        构建依赖θ的QP目标函数参数

        核心思路：
        效用函数 U_n = U_n^base(x) - β||x - θ||² (偏好接近神经网络输出)

        Nash目标: f(x) = -Σ log(U_n)

        在x*处展开:
        f(x) ≈ f(x*) + p^T(x-x*) + 0.5*(x-x*)^T Q (x-x*)

        其中:
        p = ∂f/∂x|_{x*} = -Σ (1/U_n) * ∂U_n/∂x|_{x*}
        Q = ∂²f/∂x²|_{x*}  (近似为对角阵)

        关键：∂U_n/∂x 包含 -2β(x - θ) 项，使得 p 依赖 θ
        """
        batch_size = x_converged.shape[0]
        n = x_converged.shape[1]
        device = x_converged.device
        dtype = x_converged.dtype

        # 计算收敛点的效用
        utilities = self._compute_household_utilities(
            x_converged, member_mask, tour_mask, member_is_adult,
            theta_destination_logits.detach(), home_zone, expected_tt,
            theta_x[:, :self.max_members * self.max_tours].view(batch_size, self.max_members, self.max_tours),  # departure作为偏好
            theta_x[:, self.max_members * self.max_tours:2*self.max_members * self.max_tours].view(batch_size, self.max_members, self.max_tours),  # duration作为偏好
            flexibility
        )

        # 效用的倒数 [batch, M]
        inv_U = 1.0 / utilities.clamp(min=1e-6)  # [batch, M]

        # -----------------------------------------------------------------
        # 构建p (梯度)
        #
        # p_i = -Σ_n (1/U_n) * ∂U_n/∂x_i
        #
        # 偏好惩罚项贡献: -2β(x_i - θ_i)
        # 所以 ∂U_n/∂x_i 中包含 -2β (针对与成员n相关的变量)
        #
        # 简化：p ≈ -2β * Σ_n (1/U_n) * (x - θ) 对成员n的变量
        # -----------------------------------------------------------------

        beta = 2.0  # 偏好权重

        # 将inv_U扩展到变量维度
        # [batch, M] -> [batch, M, T, vars_per_tour] -> [batch, n]
        inv_U_expanded = inv_U.unsqueeze(-1).unsqueeze(-1)
        inv_U_expanded = inv_U_expanded.expand(-1, -1, self.max_tours, self.vars_per_tour)
        inv_U_flat = inv_U_expanded.reshape(batch_size, n)

        # p = -2β * inv_U * (x_converged - theta_x)
        # 注意：这里theta_x保持计算图
        p = -2 * beta * inv_U_flat * (x_converged.detach() - theta_x)

        # -----------------------------------------------------------------
        # 构建Q (Hessian近似)
        #
        # 简化为对角阵：Q_ii = 2β * Σ_n (1/U_n²) (如果变量i属于成员n)
        # -----------------------------------------------------------------

        inv_U_sq = 1.0 / (utilities.clamp(min=1e-6) ** 2)
        inv_U_sq_expanded = inv_U_sq.unsqueeze(-1).unsqueeze(-1)
        inv_U_sq_expanded = inv_U_sq_expanded.expand(-1, -1, self.max_tours, self.vars_per_tour)
        inv_U_sq_flat = inv_U_sq_expanded.reshape(batch_size, n)

        # Q为对角阵
        Q_diag = 2 * beta * inv_U_sq_flat
        Q = torch.diag_embed(Q_diag)

        return Q, p

    def _fallback_qp_solve(
        self,
        Q: Tensor,
        p: Tensor,
        G: Tensor,
        h: Tensor,
        A: Tensor,
        b: Tensor,
    ) -> Tensor:
        """备选QP求解"""
        try:
            d = -torch.linalg.solve(Q, p.unsqueeze(-1)).squeeze(-1)
        except:
            d = -p / (torch.diagonal(Q, dim1=-2, dim2=-1) + 1e-6)

        d = self._simple_constraint_projection(d, G, h, A, b)
        return d


# =============================================================================
#                    包装类：便于集成到现有模型
# =============================================================================

class HouseholdNashBargainingLayer(nn.Module):
    """
    家庭Nash Bargaining协调层 - 便于集成的包装类
    """

    def __init__(
        self,
        num_zones: int,
        num_modes: int = 4,
        max_members: int = 6,
        max_tours: int = 8,
        utility_params: Optional[UtilityParams] = None,
        sqp_max_iter: int = 10,
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
        """前向传播"""
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
        sqp_max_iter=config.get('sqp_max_iter', 10),
        sqp_tol=config.get('sqp_tol', 1e-4),
        enabled=config.get('enabled', True),
    )