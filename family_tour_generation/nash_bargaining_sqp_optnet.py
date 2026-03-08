"""
家庭层面 Nash Bargaining 协调层 (重构版 V4)
基于 SQP + OptNet 的可微优化实现

重构要点:
1. 直接使用 decoder 原生输出（predictions dict）
2. 去掉偏好惩罚项，改用 soft anchor λ·‖x-θ‖²
3. 加入联合出行一致性硬约束
4. 不优化 purpose, destination, end token

优化变量（每成员每活动15个）:
  [start_time(1), end_time(1), mode_prob(11), joint_prob(1), driver_prob(1)]

效用函数（无偏好项）:
  U_n = U_baseline(10.0)
      + α_social × Σ_t out_of_home_prob(t) × mask(t)
      + α_joint × Σ_t joint_prob(t) × mask(t)
      + θ_escort × Σ_t driver_prob(t) × (end-start)(t) × is_adult(n) × mask(t)
      + Σ_t Σ_k mode_prob(t,k) × θ_travel(k) × expected_tt(t,k) × out_of_home(t) × mask(t)

两阶段求解:
  Phase 1: SQP (BFGS + Armijo) → x_converged (无梯度)
  Phase 2: OptNet QP → d* (有梯度，通过 θ_packed)
  x_final = θ_packed + (x_converged - θ_packed).detach() + d*

Author: 郝赫
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional
import math

try:
    from qpth.qp import QPFunction
    QPTH_AVAILABLE = True
except ImportError:
    QPTH_AVAILABLE = False
    print("Warning: qpth not available. Install with: pip install qpth")




class NashBargainingSQPOptNet(nn.Module):
    """
    Nash Bargaining SQP + OptNet layer.

    Directly accepts and returns predictions dict from decoder.
    Only modifies: continuous (start/end time), mode, joint, driver.
    Does NOT modify: purpose, destination, end.
    """

    def __init__(
        self,
        num_modes: int = 11,
        max_members: int = 8,
        max_activities: int = 6,
        alpha_joint: float = 0.1,
        alpha_social: float = 0.3,
        theta_escort: float = -0.58,
        utility_baseline: float = 10.0,
        lambda_anchor: float = 0.1,
        joint_consistency_threshold: float = 0.5,
        car_mode_indices: list = None,
        sqp_max_iter: int = 15,
        sqp_tol: float = 1e-4,
    ):
        super().__init__()

        self.num_modes = num_modes
        self.max_members = max_members
        self.max_activities = max_activities
        self.lambda_anchor = lambda_anchor
        self.joint_consistency_threshold = joint_consistency_threshold
        self.car_mode_indices = car_mode_indices or [7]
        self.sqp_max_iter = sqp_max_iter
        self.sqp_tol = sqp_tol

        # Variables per activity: start(1) + end(1) + mode_prob(num_modes) + joint_prob(1) + driver_prob(1)
        self.vars_per_activity = 2 + num_modes + 2
        self.vars_per_member = max_activities * self.vars_per_activity
        self.total_vars = max_members * self.vars_per_member

        # Learnable utility parameters — updated via KKT implicit function theorem
        # QPFunction backward: ∂d*/∂Q (depends on U → depends on these params)
        self.alpha_joint = nn.Parameter(torch.tensor(alpha_joint))
        self.alpha_social = nn.Parameter(torch.tensor(alpha_social))
        self.theta_escort = nn.Parameter(torch.tensor(theta_escort))
        self.utility_baseline = nn.Parameter(torch.tensor(utility_baseline))

        theta_travel_init = torch.tensor([
            -1.5, -0.4, -0.4, -0.8, -0.8,
            -1.0, -1.0, -0.4, -1.0, -0.8, -0.5
        ][:num_modes])
        self.theta_travel = nn.Parameter(theta_travel_init)

        # Time z-score bounds: mean=12.946, std=4.691
        self.time_mean = 12.946
        self.time_std = 4.691
        self.z_min = (1 - self.time_mean) / self.time_std     # ≈ -2.55
        self.z_max = (24 - self.time_mean) / self.time_std    # ≈ 2.36

        self.eps = 1e-3  # Must be large enough that log(eps) is safe in float16
        self.enabled = True

        # Precompute constraint index templates (run once, no Python loops at forward time)
        self._precompute_constraint_indices()

    def set_enabled(self, enabled: bool):
        """Dynamically enable/disable the Nash layer."""
        self.enabled = enabled

    def _precompute_constraint_indices(self):
        """Precompute all constraint templates and index arrays.

        Called once at __init__. All Python for-loops run here only.
        At forward time, constraints are built via expand/gather — zero loops.
        """
        M = self.max_members
        T = self.max_activities
        K = self.num_modes
        V = self.vars_per_activity
        n = self.total_vars

        # ---- Bound templates [n] ----
        lb = torch.full((n,), self.eps)
        ub = torch.full((n,), 1 - self.eps)
        for m in range(M):
            for t in range(T):
                base = m * self.vars_per_member + t * V
                lb[base]     = self.z_min
                ub[base]     = self.z_max
                lb[base + 1] = self.z_min
                ub[base + 1] = self.z_max
        self.register_buffer('lb_template', lb)
        self.register_buffer('ub_template', ub)

        # ---- Time ordering: index pairs for G rows ----
        # Each constraint: d[pos_idx] - d[neg_idx] <= x[neg_idx] - x[pos_idx]
        time_pos_idx = []  # index where G row has +1
        time_neg_idx = []  # index where G row has -1
        for m in range(M):
            for t in range(T):
                base = m * self.vars_per_member + t * V
                si, ei = base, base + 1
                # end >= start
                time_pos_idx.append(si)
                time_neg_idx.append(ei)
                # start(t+1) >= end(t)
                if t < T - 1:
                    next_si = m * self.vars_per_member + (t + 1) * V
                    time_pos_idx.append(ei)
                    time_neg_idx.append(next_si)

        self.register_buffer('time_pos_idx', torch.tensor(time_pos_idx, dtype=torch.long))
        self.register_buffer('time_neg_idx', torch.tensor(time_neg_idx, dtype=torch.long))
        num_time_rows = len(time_pos_idx)

        # Build sparse time_G_template [num_time_rows, n]
        time_G = torch.zeros(num_time_rows, n)
        for i in range(num_time_rows):
            time_G[i, time_pos_idx[i]] = 1.0
            time_G[i, time_neg_idx[i]] = -1.0
        self.register_buffer('time_G_template', time_G)

        # ---- Vehicle constraint indices ----
        veh_drv_idx = []
        veh_car_idx = []
        veh_member_idx = []
        veh_activity_idx = []
        car_idx = self.car_mode_indices[0]
        for m in range(M):
            for t in range(T):
                base = m * self.vars_per_member + t * V
                veh_drv_idx.append(base + 2 + K + 1)
                veh_car_idx.append(base + 2 + car_idx)
                veh_member_idx.append(m)
                veh_activity_idx.append(t)
        self.register_buffer('veh_drv_idx', torch.tensor(veh_drv_idx, dtype=torch.long))
        self.register_buffer('veh_car_idx', torch.tensor(veh_car_idx, dtype=torch.long))
        self.register_buffer('veh_member_idx', torch.tensor(veh_member_idx, dtype=torch.long))
        self.register_buffer('veh_activity_idx', torch.tensor(veh_activity_idx, dtype=torch.long))

        # ---- Mode normalisation: A_mode_template [M*T, n] ----
        A_mode = torch.zeros(M * T, n)
        for m in range(M):
            for t in range(T):
                eq_idx = m * T + t
                base = m * self.vars_per_member + t * V
                for k in range(K):
                    A_mode[eq_idx, base + 2 + k] = 1.0
        self.register_buffer('A_mode_template', A_mode)

        # ---- Joint travel consistency: member pair indices ----
        pair_t = []
        pair_m1 = []
        pair_m2 = []
        pair_base_m1 = []
        pair_base_m2 = []
        for t in range(T):
            for m1 in range(M):
                for m2 in range(m1 + 1, M):
                    pair_t.append(t)
                    pair_m1.append(m1)
                    pair_m2.append(m2)
                    pair_base_m1.append(m1 * self.vars_per_member + t * V)
                    pair_base_m2.append(m2 * self.vars_per_member + t * V)

        num_pairs = len(pair_t)
        self.register_buffer('pair_t', torch.tensor(pair_t, dtype=torch.long))
        self.register_buffer('pair_m1', torch.tensor(pair_m1, dtype=torch.long))
        self.register_buffer('pair_m2', torch.tensor(pair_m2, dtype=torch.long))

        # Per pair: 2 (start, end) + K (mode) equality constraints
        rows_per_pair = 2 + K
        # Build joint_A_template [num_pairs * rows_per_pair, n]
        # and index arrays for b computation
        joint_A = torch.zeros(num_pairs * rows_per_pair, n)
        joint_col_m1 = []  # column index in x for m1 variable
        joint_col_m2 = []  # column index in x for m2 variable
        for p in range(num_pairs):
            bm1 = pair_base_m1[p]
            bm2 = pair_base_m2[p]
            row_base = p * rows_per_pair
            # start constraint
            joint_A[row_base, bm1] = 1.0
            joint_A[row_base, bm2] = -1.0
            joint_col_m1.append(bm1)
            joint_col_m2.append(bm2)
            # end constraint
            joint_A[row_base + 1, bm1 + 1] = 1.0
            joint_A[row_base + 1, bm2 + 1] = -1.0
            joint_col_m1.append(bm1 + 1)
            joint_col_m2.append(bm2 + 1)
            # mode constraints
            for k in range(K):
                joint_A[row_base + 2 + k, bm1 + 2 + k] = 1.0
                joint_A[row_base + 2 + k, bm2 + 2 + k] = -1.0
                joint_col_m1.append(bm1 + 2 + k)
                joint_col_m2.append(bm2 + 2 + k)

        self.register_buffer('joint_A_template', joint_A)
        self.register_buffer('joint_col_m1', torch.tensor(joint_col_m1, dtype=torch.long))
        self.register_buffer('joint_col_m2', torch.tensor(joint_col_m2, dtype=torch.long))
        self.num_pairs = num_pairs
        self.rows_per_pair = rows_per_pair

    # =====================================================================
    #                         Forward Pass
    # =====================================================================

    def forward(
        self,
        predictions: Dict[str, Tensor],  # decoder output dict
        member_mask: Tensor,             # [B, M]
        activity_mask: Tensor,           # [B, M, T]
        num_vehicles: Tensor,            # [B]
        home_zone: Tensor,               # [B]
        member_is_adult: Tensor,         # [B, M]
        travel_time_matrix: Tensor,      # [Z, Z] or [Z, Z, K]
    ) -> Dict[str, Tensor]:
        """Forward pass. Returns modified predictions dict."""
        if not self.enabled:
            return predictions

        # Work in float32 for numerical stability (disable AMP)
        with torch.amp.autocast('cuda', enabled=False):
            preds_f32 = {
                k: v.float() if torch.is_tensor(v) else v
                for k, v in predictions.items()
            }
            result = self._forward_impl(
                preds_f32,
                member_mask.float(),
                activity_mask.float(),
                num_vehicles.float(),
                home_zone,
                member_is_adult.float(),
                travel_time_matrix.float(),
            )

        return result

    def _forward_impl(
        self,
        predictions: Dict[str, Tensor],
        member_mask: Tensor,
        activity_mask: Tensor,
        num_vehicles: Tensor,
        home_zone: Tensor,
        member_is_adult: Tensor,
        travel_time_matrix: Tensor,
    ) -> Dict[str, Tensor]:
        """Core implementation in float32."""
        batch_size = member_mask.shape[0]
        device = member_mask.device

        # ---- Fixed quantities (detached, not optimized) ----
        dest_logits = predictions['destination'].detach()
        dest_prob = F.softmax(dest_logits, dim=-1)  # [B, M, T, Z]

        home_idx = home_zone.long().view(-1, 1, 1, 1).expand(
            batch_size, self.max_members, self.max_activities, 1
        )
        at_home_prob = dest_prob.gather(-1, home_idx).squeeze(-1)
        out_of_home_prob = (1 - at_home_prob).detach()

        expected_tt = self._compute_expected_tt(
            dest_prob, home_zone, travel_time_matrix
        ).detach()

        # ---- Pack decoder outputs ----
        theta_packed = self._pack_from_predictions(predictions)  # [B, n] keeps grad
        x0 = theta_packed.detach().clone()
        var_mask = self._build_var_mask(activity_mask)  # [B, n]

        # ---- Phase 1: SQP (no gradient) ----
        x_converged = self._sqp_phase(
            x0, var_mask, member_mask, activity_mask,
            num_vehicles, member_is_adult,
            out_of_home_prob, expected_tt
        )

        # ---- Phase 2: OptNet (gradient through theta_packed via KKT) ----
        x_final = self._optnet_phase(
            x_converged, theta_packed, var_mask,
            member_mask, activity_mask,
            num_vehicles, member_is_adult,
            out_of_home_prob, expected_tt
        )

        # ---- Unpack ----
        return self._unpack_to_predictions(x_final, predictions, activity_mask)


    # =====================================================================
    #                     Variable Packing / Unpacking
    # =====================================================================

    def _pack_from_predictions(self, predictions: Dict[str, Tensor]) -> Tensor:
        """Pack decoder outputs into flat vector [B, n].

        Per activity: [start(1), end(1), mode_prob(K), joint_prob(1), driver_prob(1)]
        """
        start_time = predictions['continuous'][..., 0]  # [B, M, T]
        end_time = predictions['continuous'][..., 1]    # [B, M, T]

        mode_prob = F.softmax(predictions['mode'], dim=-1)              # [B, M, T, K]
        joint_prob = F.softmax(predictions['joint'], dim=-1)[..., 1:2]  # [B, M, T, 1]
        driver_prob = F.softmax(predictions['driver'], dim=-1)[..., 1:2]  # [B, M, T, 1]

        packed = torch.cat([
            start_time.unsqueeze(-1),
            end_time.unsqueeze(-1),
            mode_prob,
            joint_prob,
            driver_prob,
        ], dim=-1)  # [B, M, T, vars_per_activity]

        return packed.reshape(packed.shape[0], -1)  # [B, n]

    def _unpack_flat(self, x: Tensor) -> dict:
        """Unpack flat vector to named tensors."""
        B = x.shape[0]
        K = self.num_modes
        x_4d = x.reshape(B, self.max_members, self.max_activities, self.vars_per_activity)

        return {
            'start_time':  x_4d[..., 0],            # [B, M, T]
            'end_time':    x_4d[..., 1],            # [B, M, T]
            'mode_prob':   x_4d[..., 2:2+K],        # [B, M, T, K]
            'joint_prob':  x_4d[..., 2+K],          # [B, M, T]
            'driver_prob': x_4d[..., 2+K+1],        # [B, M, T]
        }

    def _unpack_to_predictions(
        self,
        x_final: Tensor,
        original_predictions: Dict[str, Tensor],
        activity_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """Convert optimized vector back to predictions dict format."""
        unpacked = self._unpack_flat(x_final)
        result = {k: v for k, v in original_predictions.items()}  # shallow copy

        mask_3d = activity_mask.bool()   # [B, M, T]
        mask_4d = mask_3d.unsqueeze(-1)  # [B, M, T, 1]

        # --- continuous ---
        new_continuous = torch.stack([
            unpacked['start_time'], unpacked['end_time']
        ], dim=-1)
        result['continuous'] = torch.where(
            mask_4d.expand_as(new_continuous),
            new_continuous,
            original_predictions['continuous'],
        )

        # --- mode: prob → log-prob with straight-through gradient ---
        mode_prob_safe = unpacked['mode_prob'].clamp(min=self.eps)
        mode_prob_safe = mode_prob_safe / mode_prob_safe.sum(dim=-1, keepdim=True)  # re-normalize
        log_mode = torch.log(mode_prob_safe).clamp(min=-7.0)
        # Straight-through: forward = log(p), backward = identity (no 1/p distortion)
        new_mode_logits = mode_prob_safe + (log_mode - mode_prob_safe).detach()
        result['mode'] = torch.where(
            mask_4d.expand_as(new_mode_logits),
            new_mode_logits,
            original_predictions['mode'],
        )

        # --- joint: scalar prob → 2-class log-prob with straight-through ---
        jp = unpacked['joint_prob'].clamp(self.eps, 1 - self.eps)
        raw_joint = torch.stack([1 - jp, jp], dim=-1)
        log_joint = torch.log(raw_joint).clamp(min=-7.0)
        new_joint_logits = raw_joint + (log_joint - raw_joint).detach()
        result['joint'] = torch.where(
            mask_4d.expand_as(new_joint_logits),
            new_joint_logits,
            original_predictions['joint'],
        )

        # --- driver: scalar prob → 2-class log-prob with straight-through ---
        dp = unpacked['driver_prob'].clamp(self.eps, 1 - self.eps)
        raw_driver = torch.stack([1 - dp, dp], dim=-1)
        log_driver = torch.log(raw_driver).clamp(min=-7.0)
        new_driver_logits = raw_driver + (log_driver - raw_driver).detach()
        result['driver'] = torch.where(
            mask_4d.expand_as(new_driver_logits),
            new_driver_logits,
            original_predictions['driver'],
        )

        return result

    def _build_var_mask(self, activity_mask: Tensor) -> Tensor:
        """Build per-variable mask [B, n]."""
        mask_expanded = activity_mask.unsqueeze(-1).expand(
            -1, -1, -1, self.vars_per_activity
        )
        return mask_expanded.reshape(activity_mask.shape[0], -1)

    # =====================================================================
    #                     Fixed Quantity Computation
    # =====================================================================

    def _compute_expected_tt(
        self,
        dest_prob: Tensor,
        home_zone: Tensor,
        travel_time_matrix: Tensor,
    ) -> Tensor:
        """Compute expected travel time [B, M, T, K].

        dest_prob:            [B, M, T, Z]
        travel_time_matrix:   [Z, Z] or [Z, Z, K]
        """
        if travel_time_matrix.dim() == 2:
            # [Z, Z] – mode-independent travel time
            tt_from_home = travel_time_matrix[home_zone.long()]  # [B, Z]
            expected_tt = torch.einsum('bz,bmtz->bmt', tt_from_home, dest_prob)
            return expected_tt.unsqueeze(-1).expand(-1, -1, -1, self.num_modes)

        elif travel_time_matrix.dim() == 3:
            # [Z, Z, K] – mode-specific travel time
            tt_from_home = travel_time_matrix[home_zone.long()]  # [B, Z, K]
            return torch.einsum('bmtz,bzk->bmtk', dest_prob, tt_from_home)

        else:
            raise ValueError(
                f"Unexpected travel_time_matrix dim: {travel_time_matrix.dim()}"
            )

    # =====================================================================
    #                     Utility Function
    # =====================================================================

    def _compute_member_utilities(
        self,
        x: Tensor,               # [B, n]
        member_mask: Tensor,      # [B, M]
        activity_mask: Tensor,    # [B, M, T]
        member_is_adult: Tensor,  # [B, M]
        out_of_home_prob: Tensor, # [B, M, T]
        expected_tt: Tensor,      # [B, M, T, K]
    ) -> Tensor:
        """Compute utility for each member.  Returns [B, M]."""
        unpacked = self._unpack_flat(x)

        start      = unpacked['start_time']   # [B, M, T]
        end        = unpacked['end_time']      # [B, M, T]
        mode_prob  = unpacked['mode_prob']     # [B, M, T, K]
        joint_prob = unpacked['joint_prob']    # [B, M, T]
        driver_prob = unpacked['driver_prob']  # [B, M, T]

        mask = activity_mask  # [B, M, T]
        duration = end - start  # [B, M, T]  (z-score space)

        # Baseline (expand learnable scalar to [B, M])
        U = self.utility_baseline.expand(x.shape[0], self.max_members).clone()

        # Social utility
        U_social = self.alpha_social * (out_of_home_prob * mask).sum(dim=-1)

        # Joint utility
        U_joint = self.alpha_joint * (joint_prob * mask).sum(dim=-1)

        # Escort utility (adults only)
        escort_time = (driver_prob * duration * mask).sum(dim=-1)
        U_escort = self.theta_escort * escort_time * member_is_adult

        # Travel utility
        # per-trip: Σ_k mode_prob(k) * θ_travel(k) * expected_tt(k)
        travel_per_trip = (mode_prob * expected_tt * self.theta_travel).sum(dim=-1)  # [B,M,T]
        U_travel = (out_of_home_prob * travel_per_trip * mask).sum(dim=-1)

        U_total = U + U_social + U_joint + U_escort + U_travel

        # ---- Smooth lower bound (softplus): ensure U ≥ 1.0 ----
        U_min = 1.0
        U_total = U_total + F.softplus(U_min - U_total)

        # Masked members → baseline (avoid log(0) downstream)
        U_total = torch.where(
            member_mask > 0,
            U_total,
            self.utility_baseline.detach().expand_as(U_total),
        )

        return U_total  # [B, M]

    def _compute_member_utilities_detached(
        self,
        x: Tensor, member_mask: Tensor, activity_mask: Tensor,
        member_is_adult: Tensor, out_of_home_prob: Tensor, expected_tt: Tensor,
        alpha_joint, alpha_social, theta_escort, utility_baseline, theta_travel,
    ) -> Tensor:
        """Same as _compute_member_utilities but takes params as args (detached).

        Used in SQP inner loop to avoid accumulating gradient graph through nn.Parameters.
        """
        unpacked = self._unpack_flat(x)
        start = unpacked['start_time']
        end = unpacked['end_time']
        mode_prob = unpacked['mode_prob']
        joint_prob = unpacked['joint_prob']
        driver_prob = unpacked['driver_prob']

        mask = activity_mask
        duration = end - start

        U = torch.full((x.shape[0], self.max_members), utility_baseline.item(),
                        device=x.device, dtype=x.dtype)
        U_social = alpha_social * (out_of_home_prob * mask).sum(dim=-1)
        U_joint = alpha_joint * (joint_prob * mask).sum(dim=-1)
        escort_time = (driver_prob * duration * mask).sum(dim=-1)
        U_escort = theta_escort * escort_time * member_is_adult
        travel_per_trip = (mode_prob * expected_tt * theta_travel).sum(dim=-1)
        U_travel = (out_of_home_prob * travel_per_trip * mask).sum(dim=-1)

        U_total = U + U_social + U_joint + U_escort + U_travel
        U_total = U_total + F.softplus(1.0 - U_total)
        U_total = torch.where(member_mask > 0, U_total,
                              torch.full_like(U_total, utility_baseline.item()))
        return U_total

    def _compute_nash_welfare(
        self, utilities: Tensor, member_mask: Tensor
    ) -> Tensor:
        """Nash welfare = Σ_n log(U_n) for valid members."""
        log_U = torch.log(utilities.clamp(min=self.eps))
        return (log_U * member_mask).sum(dim=-1)  # [B]

    # =====================================================================
    #                     Phase 1: SQP (no gradient)
    # =====================================================================

    def _sqp_phase(
        self,
        x0: Tensor,
        var_mask: Tensor,
        member_mask: Tensor,
        activity_mask: Tensor,
        num_vehicles: Tensor,
        member_is_adult: Tensor,
        out_of_home_prob: Tensor,
        expected_tt: Tensor,
    ) -> Tensor:
        """SQP iterations to find a converged point.  No gradient needed.

        Solves a reduced-dimension system using only valid variables (var_mask==1).
        """
        B, n = x0.shape
        device = x0.device

        # ---- Compute valid variable indices (same across batch) ----
        valid_idx = var_mask[0].bool()  # [n]
        n_valid = valid_idx.sum().item()

        # If no valid variables, return immediately
        if n_valid == 0:
            return x0.clone()

        x = x0.clone()
        H_v = torch.eye(n_valid, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        prev_x_v = None
        prev_grad_v = None

        with torch.no_grad():
            # Detach learnable params for SQP inner loop to prevent graph accumulation
            alpha_joint_val = self.alpha_joint.detach()
            alpha_social_val = self.alpha_social.detach()
            theta_escort_val = self.theta_escort.detach()
            utility_baseline_val = self.utility_baseline.detach()
            theta_travel_val = self.theta_travel.detach()

            for _ in range(self.sqp_max_iter):
                # ---- gradient of Nash objective (w.r.t. x only, not params) ----
                x_var = x.detach().clone().requires_grad_(True)

                with torch.enable_grad():
                    utilities = self._compute_member_utilities_detached(
                        x_var, member_mask, activity_mask,
                        member_is_adult, out_of_home_prob, expected_tt,
                        alpha_joint_val, alpha_social_val, theta_escort_val,
                        utility_baseline_val, theta_travel_val,
                    )
                    nash_obj = -self._compute_nash_welfare(utilities, member_mask)
                    grad = torch.autograd.grad(nash_obj.sum(), x_var)[0]

                grad = grad.detach()

                # NaN guard
                if torch.isnan(grad).any():
                    break

                # Extract valid subsets
                grad_v = grad[:, valid_idx]  # [B, n_valid]

                # Convergence check
                grad_norm = grad_v.norm(dim=-1)
                if (grad_norm < self.sqp_tol).all():
                    break

                # BFGS update (reduced dimension)
                x_v = x[:, valid_idx]  # [B, n_valid]
                if prev_x_v is not None:
                    s_v = x_v - prev_x_v
                    y_v = grad_v - prev_grad_v
                    H_v = self._bfgs_update(H_v, s_v, y_v)

                prev_x_v = x_v.clone()
                prev_grad_v = grad_v.clone()

                # Newton step on reduced system
                H_reg_v = H_v + 1e-3 * torch.eye(n_valid, device=device).unsqueeze(0)
                try:
                    d_v = -torch.linalg.solve(H_reg_v, grad_v.unsqueeze(-1)).squeeze(-1)
                except Exception:
                    d_v = -grad_v / (torch.diagonal(H_reg_v, dim1=-2, dim2=-1) + 1e-6)

                # Expand back to full dimension
                d = torch.zeros(B, n, device=device)
                d[:, valid_idx] = d_v

                # Armijo line search
                alpha = self._line_search(
                    x, d, var_mask, member_mask, activity_mask,
                    member_is_adult, out_of_home_prob, expected_tt,
                )

                # Update & project
                x = x + alpha * d * var_mask
                x = self._project_to_feasible(x)

                # NaN fallback → reset to x0 (SQP did 0 steps, OptNet still valid)
                if torch.isnan(x).any():
                    x = x0.clone()
                    break

        return x

    def _bfgs_update(self, H: Tensor, s: Tensor, y: Tensor) -> Tensor:
        """Vectorized BFGS Hessian approximation update with Powell's damping.

        All operations are batched — no Python for-loop over B.
        H: [B, n, n], s: [B, n], y: [B, n]
        """
        # 1. Batch dot products
        sy = (s * y).sum(dim=-1)                              # [B]
        Hs = torch.bmm(H, s.unsqueeze(-1)).squeeze(-1)       # [B, n]
        sHs = (s * Hs).sum(dim=-1)                           # [B]

        # 2. Combined validity mask (replaces all if/continue)
        s_norm = s.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        valid = (
            ~torch.isnan(s).any(-1) & ~torch.isnan(y).any(-1)
            & ~torch.isinf(s).any(-1) & ~torch.isinf(y).any(-1)
            & (s_norm > 1e-12) & (s_norm < 1e6)
            & (y_norm > 1e-12) & (y_norm < 1e6)
            & ~torch.isnan(sHs) & ~torch.isinf(sHs)
            & (sHs > 1e-12)
        )  # [B]

        # 3. Powell's damping (torch.where replaces if)
        theta = torch.where(
            sy < 0.2 * sHs,
            (0.8 * sHs / (sHs - sy + 1e-10)).clamp(0.0, 1.0),
            torch.ones_like(sy),
        )
        y_d = theta.unsqueeze(-1) * y + (1 - theta).unsqueeze(-1) * Hs
        sy_d = (s * y_d).sum(-1)
        valid = valid & (sy_d > 1e-6)

        rho = 1.0 / (sy_d + 1e-10)
        valid = valid & (rho < 1e6)

        # 4. Batched rank-2 BFGS update
        s_c = s.unsqueeze(-1)       # [B, n, 1]
        y_c = y_d.unsqueeze(-1)     # [B, n, 1]
        rho3 = rho[:, None, None]   # [B, 1, 1]

        term1 = rho3 * torch.bmm(s_c, torch.bmm(y_c.transpose(-2, -1), H))
        term2 = rho3 * torch.bmm(torch.bmm(H, y_c), s_c.transpose(-2, -1))
        yHy = (y_d * torch.bmm(H, y_c).squeeze(-1)).sum(-1)  # [B]
        term3 = (rho3 ** 2 * yHy[:, None, None]) * torch.bmm(s_c, s_c.transpose(-2, -1))
        term4 = rho3 * torch.bmm(s_c, s_c.transpose(-2, -1))
        H_cand = H - term1 - term2 + term3 + term4

        # 5. Post-validation + selective application
        # Flatten last two dims for compatibility with older PyTorch
        H_flat = H_cand.reshape(H_cand.shape[0], -1)  # [B, n*n]
        valid = valid & (
            ~torch.isnan(H_flat).any(dim=-1)
            & ~torch.isinf(H_flat).any(dim=-1)
            & (torch.diagonal(H_cand, dim1=-2, dim2=-1) > 0).all(-1)
            & (H_flat.abs().max(dim=-1).values < 1e8)
        )

        return torch.where(valid[:, None, None], H_cand, H)

    def _line_search(
        self, x, d, var_mask, member_mask, activity_mask,
        member_is_adult, out_of_home_prob, expected_tt,
    ) -> Tensor:
        """Armijo backtracking line search."""
        B = x.shape[0]
        device = x.device

        utilities = self._compute_member_utilities(
            x, member_mask, activity_mask, member_is_adult,
            out_of_home_prob, expected_tt,
        )
        f0 = -self._compute_nash_welfare(utilities, member_mask)

        alpha = torch.ones(B, 1, device=device)

        for _ in range(10):
            x_new = self._project_to_feasible(x + alpha * d * var_mask)
            utils_new = self._compute_member_utilities(
                x_new, member_mask, activity_mask, member_is_adult,
                out_of_home_prob, expected_tt,
            )
            f_new = -self._compute_nash_welfare(utils_new, member_mask)

            sufficient_decrease = (
                f_new < f0 - 1e-4 * alpha.squeeze(-1) * (d * var_mask).pow(2).sum(dim=-1)
            )
            if sufficient_decrease.all():
                break
            alpha = torch.where(sufficient_decrease.unsqueeze(-1), alpha, alpha * 0.5)

        return alpha

    def _project_to_feasible(self, x: Tensor) -> Tensor:
        """Project to feasible region (bounds + simplex + ordering)."""
        B = x.shape[0]
        K = self.num_modes
        x_4d = x.reshape(B, self.max_members, self.max_activities, self.vars_per_activity)

        # Time bounds
        start = x_4d[..., 0].clamp(self.z_min, self.z_max)
        end   = x_4d[..., 1].clamp(self.z_min, self.z_max)
        # Ensure end >= start
        end = torch.maximum(end, start + 0.01)
        end = end.clamp(max=self.z_max)

        # Mode probability simplex
        mode_prob = x_4d[..., 2:2+K].clamp(min=self.eps)
        mode_prob = mode_prob / mode_prob.sum(dim=-1, keepdim=True)

        # Probability bounds
        joint_prob  = x_4d[..., 2+K].clamp(self.eps, 1 - self.eps)
        driver_prob = x_4d[..., 2+K+1].clamp(self.eps, 1 - self.eps)

        x_proj = torch.cat([
            start.unsqueeze(-1),
            end.unsqueeze(-1),
            mode_prob,
            joint_prob.unsqueeze(-1),
            driver_prob.unsqueeze(-1),
        ], dim=-1)

        return x_proj.reshape(B, -1)

    # =====================================================================
    #                     Phase 2: OptNet (with gradient)
    # =====================================================================

    def _optnet_phase(
        self,
        x_converged: Tensor,     # [B, n]  detached
        theta_packed: Tensor,     # [B, n]  with grad
        var_mask: Tensor,
        member_mask: Tensor,
        activity_mask: Tensor,
        num_vehicles: Tensor,
        member_is_adult: Tensor,
        out_of_home_prob: Tensor,
        expected_tt: Tensor,
    ) -> Tensor:
        """OptNet phase: solve reduced-dimension QP with gradient through θ_packed.

        Only optimizes valid variables (var_mask==1), reducing QP from n=720 to n_valid≈135.
        """
        B, n = x_converged.shape
        device = x_converged.device

        # ---- Valid variable indices ----
        valid_idx = var_mask[0].bool()  # [n]
        n_valid = valid_idx.sum().item()

        if n_valid == 0:
            return theta_packed.clone()

        # ---- Utility values at converged point (for Q_nash_diag) ----
        # x_converged is detached, but utility params (alpha_joint, etc.) are NOT detached.
        # This allows KKT backward: ∂d*/∂Q → ∂Q/∂(1/U²) → ∂U/∂params
        utilities = self._compute_member_utilities(
            x_converged.detach(), member_mask, activity_mask,
            member_is_adult, out_of_home_prob, expected_tt,
        )  # [B, M] — carries grad through params

        inv_U    = 1.0 / utilities.clamp(min=1.0)              # [B, M]
        inv_U_sq = inv_U ** 2                                    # [B, M]

        # Expand to per-variable dimension
        inv_U_sq_expanded = inv_U_sq.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.max_activities, self.vars_per_activity
        ).reshape(B, n)

        # ---- Full-dimensional Q_diag and p (p depends on θ!) ----
        Q_diag_full = inv_U_sq_expanded + self.lambda_anchor    # [B, n]
        p_full = self.lambda_anchor * (x_converged.detach() - theta_packed)  # depends on θ!

        # ---- Full-dimensional constraints ----
        G_full, h, A_full, b = self._build_optnet_constraints(
            x_converged, var_mask, member_mask, activity_mask,
            num_vehicles, member_is_adult,
        )

        # ---- Reduce to valid variables only ----
        Q_v_diag = Q_diag_full[:, valid_idx]                    # [B, n_valid]
        Q_v = torch.diag_embed(Q_v_diag)                       # [B, n_valid, n_valid]
        p_v = p_full[:, valid_idx]                              # [B, n_valid]
        G_v = G_full[:, :, valid_idx]                           # [B, num_ineq, n_valid]
        A_v = A_full[:, :, valid_idx]                           # [B, num_eq, n_valid]

        # Remove all-zero equality rows (inactive constraints for masked variables)
        if A_v.shape[1] > 0:
            active_eq = A_v.abs().sum(dim=(0, 2)) > 1e-12      # [num_eq]
            A_v = A_v[:, active_eq]
            b = b[:, active_eq]

        # Remove linearly dependent equality rows via QR rank detection
        # This prevents qpth from encountering singular KKT systems
        if A_v.shape[1] > 0:
            # Use the first batch element (constraint structure is similar across batch)
            try:
                _, R = torch.linalg.qr(A_v[0].double().T)  # R: [num_eq, num_eq]
                diag_R = torch.abs(torch.diag(R))
                indep_mask = diag_R > 1e-8
                if indep_mask.sum() < A_v.shape[1]:
                    A_v = A_v[:, indep_mask]
                    b = b[:, indep_mask]
            except Exception:
                pass  # keep all rows if QR fails

        # Remove all-zero inequality rows similarly
        if G_v.shape[1] > 0:
            active_ineq = G_v.abs().sum(dim=(0, 2)) > 1e-12
            G_v = G_v[:, active_ineq]
            h = h[:, active_ineq]

        # Slightly relax inequality bounds to improve feasibility
        if h.shape[1] > 0:
            h = h + 1e-4

        # ---- Solve reduced QP with gradient (KKT implicit function theorem) ----
        try:
            if QPTH_AVAILABLE:
                d_v_star = QPFunction(verbose=False, maxIter=200, eps=1e-8)(
                    Q_v.double(), p_v.double(),
                    G_v.double(), h.double(),
                    A_v.double(), b.double(),
                ).float()
            else:
                d_v_star = self._fallback_qp_solve(Q_v, p_v, G_v, h, A_v, b)
        except Exception:
            d_v_star = torch.zeros(B, n_valid, device=device)

        # NaN/Inf fallback
        if torch.isnan(d_v_star).any() or torch.isinf(d_v_star).any():
            d_v_star = torch.zeros(B, n_valid, device=device)

        # Clamp d_star to prevent extreme values
        d_v_star = d_v_star.clamp(-10.0, 10.0)

        # ---- Expand back to full dimension (preserves gradient from QPFunction) ----
        valid_positions = valid_idx.nonzero(as_tuple=True)[0]  # [n_valid]
        d_star = theta_packed.new_zeros(B, n)
        d_star = d_star.scatter(1, valid_positions.unsqueeze(0).expand(B, -1), d_v_star)

        # ---- x_final = θ + (x_conv − θ).detach() + d*(θ) ----
        # d* carries gradient through QPFunction's KKT backward (implicit function theorem)
        # p = λ(x_conv - θ) → ∂d*/∂θ via KKT → decoder learns to align with Nash equilibrium
        x_final = theta_packed + (x_converged - theta_packed).detach() + d_star

        if torch.isnan(x_final).any():
            x_final = theta_packed  # Fallback: skip Nash adjustment

        x_final = self._project_to_feasible(x_final)

        return x_final

    # =====================================================================
    #                     Constraint Building (Phase 2)
    # =====================================================================

    def _build_optnet_constraints(
        self,
        x_conv: Tensor,          # [B, n]
        var_mask: Tensor,         # [B, n]
        member_mask: Tensor,      # [B, M]
        activity_mask: Tensor,    # [B, M, T]
        num_vehicles: Tensor,     # [B]
        member_is_adult: Tensor,  # [B, M]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build QP constraints using precomputed templates — zero Python loops.

        Inequality (G d ≤ h):
            1. Bound constraints
            2. Time ordering constraints
            3. Vehicle constraint (linearised bilinear)

        Equality (A d = b):
            4. Mode probability normalisation  Σ Δmode_prob = 0
            5. Joint travel consistency  (conditional)
        """
        B, n = x_conv.shape
        device = x_conv.device
        M = self.max_members
        T = self.max_activities
        K = self.num_modes
        V = self.vars_per_activity

        G_list: list[Tensor] = []
        h_list: list[Tensor] = []
        A_list: list[Tensor] = []
        b_list: list[Tensor] = []

        I_n = torch.eye(n, device=device).unsqueeze(0).expand(B, -1, -1)

        # ---- 1. Bound constraints (expand precomputed templates) ----
        lb = self.lb_template.unsqueeze(0).expand(B, -1)
        ub = self.ub_template.unsqueeze(0).expand(B, -1)

        G_list.append(-I_n)
        h_list.append(x_conv - lb)
        G_list.append(I_n)
        h_list.append(ub - x_conv)

        # ---- 2. Time ordering (expand template + gather h) ----
        time_G = self.time_G_template.unsqueeze(0).expand(B, -1, -1)
        time_h = x_conv[:, self.time_neg_idx] - x_conv[:, self.time_pos_idx]  # [B, num_time_rows]
        G_list.append(time_G)
        h_list.append(time_h)

        # ---- 3. Vehicle constraint (vectorized gather) ----
        x_4d = x_conv.reshape(B, M, T, V)
        driver_p0 = x_4d[..., 2 + K + 1]     # [B, M, T]
        car_p0 = x_4d[..., 2 + self.car_mode_indices[0]]  # [B, M, T]

        # Flatten to [B, M*T] using precomputed indices
        c0_flat = car_p0[:, self.veh_member_idx, self.veh_activity_idx]     # [B, M*T]
        d0_flat = driver_p0[:, self.veh_member_idx, self.veh_activity_idx]  # [B, M*T]
        adult_flat = member_is_adult[:, self.veh_member_idx]                # [B, M*T]
        valid_flat = activity_mask[:, self.veh_member_idx, self.veh_activity_idx]  # [B, M*T]
        weight = adult_flat * valid_flat  # [B, M*T]

        G_veh = torch.zeros(B, 1, n, device=device)
        # scatter driver and car coefficients
        G_veh[:, 0, :].scatter_add_(
            1, self.veh_drv_idx.unsqueeze(0).expand(B, -1),
            c0_flat * weight,
        )
        G_veh[:, 0, :].scatter_add_(
            1, self.veh_car_idx.unsqueeze(0).expand(B, -1),
            d0_flat * weight,
        )

        current_usage = (
            driver_p0 * car_p0 * member_is_adult.unsqueeze(-1) * activity_mask
        ).sum(dim=(1, 2))
        h_veh = (num_vehicles - current_usage).unsqueeze(-1)
        G_list.append(G_veh)
        h_list.append(h_veh)

        # ---- 4. Mode normalisation (expand template) ----
        A_mode = self.A_mode_template.unsqueeze(0).expand(B, -1, -1)
        b_mode = torch.zeros(B, M * T, device=device)
        A_list.append(A_mode)
        b_list.append(b_mode)

        # ---- 5. Joint travel consistency (vectorized) ----
        if self.num_pairs > 0:
            joint_prob_conv = x_4d[..., 2 + K]  # [B, M, T]

            # Compute activation for all pairs at once
            jp_m1 = joint_prob_conv[:, self.pair_m1, self.pair_t]  # [B, num_pairs]
            jp_m2 = joint_prob_conv[:, self.pair_m2, self.pair_t]  # [B, num_pairs]
            active = (jp_m1 * jp_m2 > self.joint_consistency_threshold).float()  # [B, num_pairs]

            # Expand active to per-row: [B, num_pairs * rows_per_pair]
            active_exp = active.unsqueeze(-1).expand(
                -1, -1, self.rows_per_pair
            ).reshape(B, -1)  # [B, num_pairs * rows_per_pair]

            # Apply activation mask to template
            joint_A = self.joint_A_template.unsqueeze(0).expand(B, -1, -1) * active_exp.unsqueeze(-1)

            # Compute b: active * (x[col_m2] - x[col_m1])
            joint_b = active_exp * (
                x_conv[:, self.joint_col_m2] - x_conv[:, self.joint_col_m1]
            )

            A_list.append(joint_A)
            b_list.append(joint_b)

        # ---- Combine ----
        G = torch.cat(G_list, dim=1)
        h = torch.cat(h_list, dim=1)
        A = (
            torch.cat(A_list, dim=1)
            if A_list
            else torch.zeros(B, 0, n, device=device)
        )
        b = (
            torch.cat(b_list, dim=1)
            if b_list
            else torch.zeros(B, 0, device=device)
        )

        return G, h, A, b

    # =====================================================================
    #                     Fallback QP Solver
    # =====================================================================

    def _fallback_qp_solve(
        self, Q: Tensor, p: Tensor, G: Tensor, h: Tensor, A: Tensor, b: Tensor,
    ) -> Tensor:
        """Fallback QP solver when qpth is unavailable."""
        try:
            d = -torch.linalg.solve(Q, p.unsqueeze(-1)).squeeze(-1)
        except Exception:
            d = -p / (torch.diagonal(Q, dim1=-2, dim2=-1) + 1e-6)
        return d


# =============================================================================
#                     Factory Function
# =============================================================================

def create_nash_layer_from_config(config) -> NashBargainingSQPOptNet:
    """Create Nash layer from model config."""
    nc = config.nash_config
    return NashBargainingSQPOptNet(
        num_modes=nc.get('num_modes', 11),
        max_members=config.max_members,
        max_activities=config.max_activities,
        alpha_joint=nc.get('alpha_joint', 0.1),
        alpha_social=nc.get('alpha_social', 0.3),
        theta_escort=nc.get('theta_escort', -0.58),
        utility_baseline=nc.get('utility_baseline', 10.0),
        lambda_anchor=nc.get('lambda_anchor', 0.1),
        joint_consistency_threshold=nc.get('joint_consistency_threshold', 0.5),
        car_mode_indices=nc.get('car_mode_indices', [7]),
        sqp_max_iter=nc.get('sqp_max_iter', 15),
        sqp_tol=nc.get('sqp_tol', 1e-4),
    )
