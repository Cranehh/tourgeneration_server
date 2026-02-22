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
        self.alpha_joint = alpha_joint
        self.alpha_social = alpha_social
        self.theta_escort = theta_escort
        self.utility_baseline = utility_baseline
        self.lambda_anchor = lambda_anchor
        self.joint_consistency_threshold = joint_consistency_threshold
        self.car_mode_indices = car_mode_indices or [7]
        self.sqp_max_iter = sqp_max_iter
        self.sqp_tol = sqp_tol

        # Variables per activity: start(1) + end(1) + mode_prob(num_modes) + joint_prob(1) + driver_prob(1)
        self.vars_per_activity = 2 + num_modes + 2
        self.vars_per_member = max_activities * self.vars_per_activity
        self.total_vars = max_members * self.vars_per_member

        # Travel time coefficients per mode (11 modes)
        # 0:walk(-1.5) 1:bus(-0.4) 2:metro(-0.4) 3:bike(-0.8) 4:ebike(-0.8)
        # 5:car(-1.0) 6:other_motor(-1.0) 7:shuttle(-0.4) 8:taxi(-1.0) 9:motorcycle(-0.8) 10:other(-0.5)
        theta_travel = torch.tensor([
            -1.5, -0.4, -0.4, -0.8, -0.8,
            -1.0, -1.0, -0.4, -1.0, -0.8, -0.5
        ][:num_modes])
        self.register_buffer('theta_travel', theta_travel)

        # Time z-score bounds: mean=12.946, std=4.691
        self.time_mean = 12.946
        self.time_std = 4.691
        self.z_min = (1 - self.time_mean) / self.time_std     # ≈ -2.55
        self.z_max = (24 - self.time_mean) / self.time_std    # ≈ 2.36

        self.eps = 1e-6
        self.enabled = True

    def set_enabled(self, enabled: bool):
        """Dynamically enable/disable the Nash layer."""
        self.enabled = enabled

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

        # out_of_home_prob = 1 - P(dest == home)
        home_idx = home_zone.long().view(-1, 1, 1, 1).expand(
            batch_size, self.max_members, self.max_activities, 1
        )
        at_home_prob = dest_prob.gather(-1, home_idx).squeeze(-1)  # [B, M, T]
        out_of_home_prob = (1 - at_home_prob).detach()

        # expected travel time [B, M, T, K]
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

        # ---- Phase 2: OptNet (gradient through theta_packed) ----
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

        # --- mode: prob → log-prob (logits) ---
        mode_prob_safe = unpacked['mode_prob'].clamp(min=self.eps)
        new_mode_logits = torch.log(mode_prob_safe)
        result['mode'] = torch.where(
            mask_4d.expand_as(new_mode_logits),
            new_mode_logits,
            original_predictions['mode'],
        )

        # --- joint: scalar prob → 2-class log-prob ---
        jp = unpacked['joint_prob'].clamp(self.eps, 1 - self.eps)
        new_joint_logits = torch.stack([torch.log(1 - jp), torch.log(jp)], dim=-1)
        result['joint'] = torch.where(
            mask_4d.expand_as(new_joint_logits),
            new_joint_logits,
            original_predictions['joint'],
        )

        # --- driver: scalar prob → 2-class log-prob ---
        dp = unpacked['driver_prob'].clamp(self.eps, 1 - self.eps)
        new_driver_logits = torch.stack([torch.log(1 - dp), torch.log(dp)], dim=-1)
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

        # Baseline
        U = torch.full(
            (x.shape[0], self.max_members),
            self.utility_baseline,
            device=x.device, dtype=x.dtype,
        )

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
            torch.full_like(U_total, self.utility_baseline),
        )

        return U_total  # [B, M]

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
        """SQP iterations to find a converged point.  No gradient needed."""
        B, n = x0.shape
        device = x0.device

        x = x0.clone()
        H = torch.eye(n, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        prev_x = None
        prev_grad = None

        with torch.no_grad():
            for _ in range(self.sqp_max_iter):
                # ---- gradient of Nash objective ----
                x_var = x.clone().requires_grad_(True)

                with torch.enable_grad():
                    utilities = self._compute_member_utilities(
                        x_var, member_mask, activity_mask,
                        member_is_adult, out_of_home_prob, expected_tt,
                    )
                    nash_obj = -self._compute_nash_welfare(utilities, member_mask)
                    grad = torch.autograd.grad(nash_obj.sum(), x_var)[0]

                grad = grad.detach()

                # NaN guard
                if torch.isnan(grad).any():
                    break

                # Convergence check
                grad_norm = (grad * var_mask).norm(dim=-1)
                if (grad_norm < self.sqp_tol).all():
                    break

                # BFGS update
                if prev_x is not None:
                    s = x - prev_x
                    y = grad - prev_grad
                    H = self._bfgs_update(H, s, y)

                prev_x = x.clone()
                prev_grad = grad.clone()

                # Newton step (unconstrained)
                H_reg = H + 1e-3 * torch.eye(n, device=device).unsqueeze(0)
                try:
                    d = -torch.linalg.solve(H_reg, grad.unsqueeze(-1)).squeeze(-1)
                except Exception:
                    d = -grad / (torch.diagonal(H_reg, dim1=-2, dim2=-1) + 1e-6)

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
        """BFGS Hessian approximation update with Powell's damping."""
        B, n, _ = H.shape
        device = H.device
        H_new = H.clone()

        for b in range(B):
            sb, yb = s[b], y[b]

            # ---- Input validation ----
            if torch.isnan(sb).any() or torch.isnan(yb).any():
                continue
            if torch.isinf(sb).any() or torch.isinf(yb).any():
                continue
            if sb.norm() < 1e-12 or sb.norm() > 1e6:
                continue
            if yb.norm() < 1e-12 or yb.norm() > 1e6:
                continue

            sy = torch.dot(sb, yb)
            Hs = H[b] @ sb
            sHs = torch.dot(sb, Hs)

            if torch.isnan(sHs) or torch.isinf(sHs) or sHs < 1e-12:
                continue

            # Powell's damping
            if sy < 0.2 * sHs:
                theta = 0.8 * sHs / (sHs - sy + 1e-10)
                theta = theta.clamp(0.0, 1.0)
                yb = theta * yb + (1 - theta) * Hs
                sy = torch.dot(sb, yb)

            if sy < 1e-6:
                continue

            rho = 1.0 / sy
            if rho > 1e6:
                continue

            # BFGS rank-2 update
            I = torch.eye(n, device=device)
            s_col = sb.unsqueeze(1)
            y_col = yb.unsqueeze(1)

            left  = I - rho * s_col @ y_col.T
            right = I - rho * y_col @ s_col.T
            H_candidate = left @ H[b] @ right + rho * s_col @ s_col.T

            # ---- Output validation ----
            if torch.isnan(H_candidate).any() or torch.isinf(H_candidate).any():
                continue
            if (torch.diag(H_candidate) <= 0).any():
                continue
            if H_candidate.abs().max() > 1e8:
                continue

            H_new[b] = H_candidate

        return H_new

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
        """OptNet phase: solve QP with gradient through θ_packed.

        QP:  min  0.5 d^T Q d + p^T d   s.t.  G d ≤ h,  A d = b

        Q = Q_nash_diag(detached) + λ_anchor · I
        p = g_nash(detached, ≈0 at converged) + λ_anchor · (x_conv.detach() − θ_packed)

        θ_packed keeps computation graph → p depends on θ → OptNet 通过 KKT 传梯度.
        """
        B, n = x_converged.shape
        device = x_converged.device

        # ---- Utility values at converged point (for Q_nash_diag) ----
        utilities = self._compute_member_utilities(
            x_converged, member_mask, activity_mask,
            member_is_adult, out_of_home_prob, expected_tt,
        )  # [B, M]

        inv_U    = (1.0 / utilities.clamp(min=1.0)).detach()   # [B, M]
        inv_U_sq = (inv_U ** 2).detach()                        # [B, M]

        # Expand to per-variable dimension
        inv_U_sq_expanded = inv_U_sq.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.max_activities, self.vars_per_activity
        ).reshape(B, n)

        # ---- Q = diag(1/U_n²) + λ I ----
        Q_diag = inv_U_sq_expanded + self.lambda_anchor
        Q = torch.diag_embed(Q_diag)  # [B, n, n]

        # ---- p = λ (x_conv − θ)  (g_nash ≈ 0 at converged point) ----
        p = self.lambda_anchor * (x_converged.detach() - theta_packed)  # depends on θ!

        # ---- Constraints ----
        G, h, A, b = self._build_optnet_constraints(
            x_converged, var_mask, member_mask, activity_mask,
            num_vehicles, member_is_adult,
        )

        # ---- Solve QP ----
        try:
            if QPTH_AVAILABLE:
                d_star = QPFunction(verbose=False, maxIter=100, eps=1e-6)(
                    Q.double(), p.double(),
                    G.double(), h.double(),
                    A.double(), b.double(),
                ).float()
            else:
                d_star = self._fallback_qp_solve(Q, p, G, h, A, b)
        except Exception as e:
            # Fallback: d* = 0  →  Nash layer ≡ identity, gradient still flows
            d_star = torch.zeros_like(theta_packed)

        # NaN fallback
        if torch.isnan(d_star).any():
            d_star = torch.zeros_like(theta_packed)

        # ---- x_final = θ + (x_conv − θ).detach() + d* ----
        x_final = theta_packed + (x_converged - theta_packed).detach() + d_star
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
        """Build QP constraints:  G d ≤ h,  A d = b.

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

        # ---- 1. Bound constraints ----
        lb = torch.full((B, n), self.eps, device=device)
        ub = torch.full((B, n), 1 - self.eps, device=device)

        for m in range(M):
            for t in range(T):
                base = m * self.vars_per_member + t * V
                lb[:, base]     = self.z_min   # start lb
                ub[:, base]     = self.z_max   # start ub
                lb[:, base + 1] = self.z_min   # end lb
                ub[:, base + 1] = self.z_max   # end ub

        # -I d ≤ x_conv − lb   (lower bound)
        G_list.append(-I_n)
        h_list.append(x_conv - lb)
        #  I d ≤ ub − x_conv   (upper bound)
        G_list.append(I_n)
        h_list.append(ub - x_conv)

        # ---- 2. Time ordering constraints ----
        time_G_rows: list[Tensor] = []
        time_h_vals: list[Tensor] = []

        for m in range(M):
            for t in range(T):
                base = m * self.vars_per_member + t * V
                si = base       # start index
                ei = base + 1   # end index

                # end ≥ start  →  d[start] − d[end] ≤ x_conv[end] − x_conv[start]
                row = torch.zeros(B, n, device=device)
                row[:, si] =  1.0
                row[:, ei] = -1.0
                time_G_rows.append(row)
                time_h_vals.append(x_conv[:, ei] - x_conv[:, si])

                # start(t+1) ≥ end(t)
                if t < T - 1:
                    next_si = m * self.vars_per_member + (t + 1) * V
                    row2 = torch.zeros(B, n, device=device)
                    row2[:, ei]      =  1.0
                    row2[:, next_si] = -1.0
                    time_G_rows.append(row2)
                    time_h_vals.append(x_conv[:, next_si] - x_conv[:, ei])

        if time_G_rows:
            G_list.append(torch.stack(time_G_rows, dim=1))
            h_list.append(torch.stack(time_h_vals, dim=1))

        # ---- 3. Vehicle constraint (linearised bilinear) ----
        x_4d = x_conv.reshape(B, M, T, V)
        driver_p0 = x_4d[..., 2 + K + 1]                        # [B, M, T]
        car_idx   = self.car_mode_indices[0]
        car_p0    = x_4d[..., 2 + car_idx]                       # [B, M, T]

        G_veh = torch.zeros(B, 1, n, device=device)
        for m in range(M):
            for t in range(T):
                base = m * self.vars_per_member + t * V
                drv_vi = base + 2 + K + 1
                car_vi = base + 2 + car_idx

                c0    = car_p0[:, m, t]
                d0    = driver_p0[:, m, t]
                adult = member_is_adult[:, m]
                valid = activity_mask[:, m, t]

                G_veh[:, 0, drv_vi] = c0 * adult * valid
                G_veh[:, 0, car_vi] = d0 * adult * valid

        current_usage = (
            driver_p0 * car_p0 * member_is_adult.unsqueeze(-1) * activity_mask
        ).sum(dim=(1, 2))
        h_veh = (num_vehicles - current_usage).unsqueeze(-1)

        G_list.append(G_veh)
        h_list.append(h_veh)

        # ---- 4. Mode probability normalisation (equality) ----
        num_eq_mode = M * T
        A_mode = torch.zeros(B, num_eq_mode, n, device=device)
        b_mode = torch.zeros(B, num_eq_mode, device=device)

        eq_idx = 0
        for m in range(M):
            for t in range(T):
                base = m * self.vars_per_member + t * V
                for k in range(K):
                    A_mode[:, eq_idx, base + 2 + k] = 1.0
                b_mode[:, eq_idx] = 0.0   # Σ Δmode_prob = 0
                eq_idx += 1

        A_list.append(A_mode)
        b_list.append(b_mode)

        # ---- 5. Joint travel consistency (equality, conditional) ----
        joint_prob_conv = x_4d[..., 2 + K]   # [B, M, T]

        joint_A_rows: list[Tensor] = []
        joint_b_vals: list[Tensor] = []

        for t in range(T):
            for m1 in range(M):
                for m2 in range(m1 + 1, M):
                    jp_prod = joint_prob_conv[:, m1, t] * joint_prob_conv[:, m2, t]
                    active = (jp_prod > self.joint_consistency_threshold).float()  # [B]

                    if active.sum() == 0:
                        continue

                    base_m1 = m1 * self.vars_per_member + t * V
                    base_m2 = m2 * self.vars_per_member + t * V

                    # start(m1) = start(m2)
                    row_s = torch.zeros(B, n, device=device)
                    row_s[:, base_m1]     =  active
                    row_s[:, base_m2]     = -active
                    joint_A_rows.append(row_s)
                    joint_b_vals.append(active * (x_conv[:, base_m2] - x_conv[:, base_m1]))

                    # end(m1) = end(m2)
                    row_e = torch.zeros(B, n, device=device)
                    row_e[:, base_m1 + 1] =  active
                    row_e[:, base_m2 + 1] = -active
                    joint_A_rows.append(row_e)
                    joint_b_vals.append(
                        active * (x_conv[:, base_m2 + 1] - x_conv[:, base_m1 + 1])
                    )

                    # mode_prob(m1,t,k) = mode_prob(m2,t,k) for each k
                    for k in range(K):
                        row_mk = torch.zeros(B, n, device=device)
                        row_mk[:, base_m1 + 2 + k] =  active
                        row_mk[:, base_m2 + 2 + k] = -active
                        joint_A_rows.append(row_mk)
                        joint_b_vals.append(
                            active * (
                                x_conv[:, base_m2 + 2 + k]
                                - x_conv[:, base_m1 + 2 + k]
                            )
                        )

        if joint_A_rows:
            A_list.append(torch.stack(joint_A_rows, dim=1))
            b_list.append(torch.stack(joint_b_vals, dim=1))

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
