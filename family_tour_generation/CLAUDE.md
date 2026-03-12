# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhD dissertation project: family tour generation using deep learning. Two components:

1. **Family Tour Generation** (`family_tour_generation/`): Main model — Transformer-based neural network generating realistic family activity chains

**Active training script**: `train_with_ss_rollout.py` (scheduled sampling + autoregressive rollout). Training logs: `../train_500ep.log`. Checkpoints: `checkpoints/`.

## Development Environment

```bash
conda activate DiT   # active environment used by the running training process
python train_with_ss_rollout.py
```

## Architecture

### Data Flow
```
family_features [B, 32]  ─┐
member_features [B, 8, 48] ┼─▶ PLEEncoder ──▶ member_repr [B, 8, 256]
                           │        (CGC layers × 3)
                           └── pattern_predictor (frozen student)

member_repr ──▶ MTANDecoder (20 layers, autoregressive)
                    ├── task-specific attention (per head: continuous/purpose/mode/driver/joint)
                    └── cross-role attention (inter-member coordination)

MTANDecoder output (predictions dict)
    ──▶ NashBargainingSQPOptNet  ←── CURRENT DEVELOPMENT FOCUS
    ──▶ final predictions dict
```

### Prediction Heads (all per [B, M, T])
| Head | Output | Loss |
|------|--------|------|
| `continuous` | start/end time (z-score) | MAE |
| `purpose` | 10-class logits | Focal |
| `mode` | 11-class logits | Focal |
| `driver` | 2-class logits | Focal |
| `joint` | 2-class logits | Focal |
| `destination` | 2006-class logits | CE |

### Nash Bargaining Layer (`nash_bargaining_sqp_optnet.py`)

**Current development focus.** Implements family-level Nash Bargaining coordination as a differentiable last layer.

**Optimization variables per member per activity (15 total)**:
```
[start_time(1), end_time(1), mode_prob(11), joint_prob(1), driver_prob(1)]
```
Does NOT modify: `purpose`, `destination`, `end` token.

**Two-phase solve**:

1. **Phase 1 — SQP** (`_sqp_phase`): BFGS + Armijo line search, finds converged point `x_converged`. Runs under `torch.no_grad()`. Uses `_compute_member_utilities_detached()` (takes params as arguments to avoid graph accumulation). NaN guard: if gradient is NaN or `x` becomes NaN, resets to `x0` and breaks — OptNet still fires.

2. **Phase 2 — OptNet** (`_optnet_phase`): Solves reduced-dimension QP (valid variables only, ~135 of 720) via `QPFunction` (qpth) under `torch.no_grad()`. **Gradient flows through `_DiagonalQPBackward`** (custom autograd.Function that approximates KKT backward with diagonal QP structure, avoiding qpth's C++ reference cycle memory leak):
   - `p = λ(x_converged − θ_packed)` where `θ_packed` carries decoder grad
   - `Q_diag = 1/U² + λ` where `U` depends on learnable utility params
   - `∂d*/∂p ≈ -1/Q_diag` → gradient to `θ_packed` (decoder)
   - `∂d*/∂Q_diag ≈ -d*/Q_diag` → gradient to utility params
   - `x_final = θ_packed + (x_converged − θ_packed).detach() + d*(θ_packed)`
   - Learnable params updated: `alpha_joint`, `alpha_social`, `theta_escort`, `theta_joint_cost`, `utility_baseline`, `theta_travel`

**Constraint structure**:
- Inequality: bound constraints + time ordering (end≥start, start(t+1)≥end(t)) + vehicle constraint (linearised bilinear)
- Equality: mode probability normalisation (Σmode=1) + joint travel consistency including joint_prob symmetry (soft sigmoid activation on joint_prob product)

**Constraint templates** are precomputed once in `_precompute_constraint_indices()` at `__init__` — zero Python loops at forward time.

**Failure protection**:
- `set_enabled(False)` → pass-through (returns `predictions` unchanged)
- Forward runs under `torch.amp.autocast('cuda', enabled=False)` (float32 always)
- NaN/Inf in `d_v_star` → zeroed out (`d_star = 0`, i.e., `x_final = θ_packed + (x_conv−θ).detach()`)
- `x_final` NaN → fallback to `θ_packed` (skip Nash adjustment entirely)
- QP solve failure (exception) → `d_v_star = zeros`
- SQP grad NaN → break, hand off current `x` (possibly `x0`) to OptNet
- BFGS update NaN/invalid → `torch.where(valid, H_cand, H)` (keep old Hessian)
- Mode logits clamped to `[-7.0, ∞)` on unpack to prevent `log(0)` in AMP
- Rollout skipped during Nash warm-up (first 10 epochs after `nash_bargaining_start_epoch`) to avoid conflicting gradients

**`_project_to_feasible`**: Fully non-in-place implementation using list+stack for time ordering. Safe to call on tensors in the autograd graph. Now also includes:
- **Vehicle constraint projection**: scales down `driver_prob` when `Σ(driver×car×adult×mask) > num_vehicles`
- **Joint consistency projection**: for 2-person families, averages start/end/mode for active joint pairs (3+ person families handled by OptNet equality constraints)
- All new projections are optional (backward-compatible signature with `Optional` params)

**`_DiagonalQPBackward`**: Custom `torch.autograd.Function` replacing qpth's KKT backward. QPFunction runs under `no_grad()` (correct constrained solution), then `_DiagonalQPBackward.apply(p_v, Q_v_diag, d_detached)` re-attaches gradient via diagonal approximation. Eliminates qpth's C++ reference cycle memory leak (~1750 MB/batch).

**Straight-through log unpack** (`_unpack_to_predictions`): Mode/joint/driver heads use `p + (log(p) - p).detach()` — forward = `log(p)` (correct loss), backward = identity (no `1/p` gradient distortion). Without this, non-target gradient for 11-class mode is ~11x too large (constant 1.0 vs proportional `p_i`).

**Checkpoint compatibility**: `load_checkpoint()` in `train_with_ss_rollout.py` overrides `utility_baseline` from config after loading, since old checkpoints store the previous value (10.0) as `nn.Parameter`. Also initializes `theta_joint_cost` if missing from old checkpoints.

**QR rank filtering**: before passing to QPFunction, removes linearly dependent equality rows via `torch.linalg.qr` to prevent singular KKT systems.

**Fallback QP solver** (`_fallback_qp_solve`): activated when qpth unavailable — solves unconstrained diagonal approximation via `torch.linalg.solve`.

## Key Files

| File | Purpose |
|------|---------|
| `nash_bargaining_sqp_optnet.py` | Nash layer — SQP + OptNet, KKT backward |
| `train_with_ss_rollout.py` | Active training: scheduled sampling + rollout |
| `model.py` | `FamilyTourGenerator`: PLEEncoder + MTANDecoder + NashLayer |
| `config.py` | All hyperparams; `nash_config` dict controls Nash layer |
| `mtan_decoder.py` | Multi-task attention decoder + `autoregressive_rollout()` |
| `exposure_bias.py` | Scheduled sampling scheduler and handler |
| `losses.py` | `FamilyTourLoss`, `compute_rollout_loss`, focal loss, MAE |
| `data.py` | `FamilyTourDataset`, `FamilyTourBatch`, `collate_fn` |
| `ple_encoder.py` | PLE encoder with CGC (cross-gate connections) |

## Config: Nash Layer

```python
# config.py ModelConfig
use_nash_bargaining: bool = True
nash_config = {
    'num_modes': 11,
    'alpha_joint': 0.1,       # learnable, init value
    'alpha_social': 0.3,      # learnable, init value
    'theta_escort': -0.58,    # learnable, init value
    'theta_joint_cost': -0.5, # learnable, quadratic jp cost → interior optimum jp*=0.35
    'utility_baseline': 2.0,  # learnable, init value (was 10.0 — too inert)
    'lambda_anchor': 0.3,     # soft anchor weight, fixed (was 0.1)
    'sqp_max_iter': 15,
    'sqp_tol': 1e-4,
    'joint_consistency_threshold': 0.5,
    'car_mode_indices': [7],
}
```

**Nash correction weight** = `inv_U_sq / (inv_U_sq + lambda_anchor)`:
- Old (baseline=10, λ=0.1): **9.1%** — nearly inert
- Current (baseline=2, λ=0.3): **45.5%** — balanced Nash vs decoder

## Data Dimensions

- Family features: `[B, 32]`
- Member features: `[B, 8, 48]`
- Activity features: `[B, 8, 6, 27]` (2 continuous + 10 purpose + 11 mode + 2 driver + 2 joint)
- `member_mask`: `[B, 8]` — valid members
- `activity_mask`: `[B, 8, 6]` — valid activities

Time variables are z-score normalized: mean=12.946, std=4.691; range ≈ [−2.55, 2.36].

## Training Status Check

**Full monitoring documentation**: see `training_monitor.md` for complete scripts, anomaly thresholds, and historical records.

Run the full check script at the start of any session. The script is in `training_monitor.md`. It covers:
1. Process & GPU status
2. Train/Val Loss trends (last 10 epochs)
3. gen_acc trends (last 5 records, logged every 5 epochs)
4. Anomaly detection (NaN, backward failures, rate of increase)
5. Historical best values comparison
6. Overfitting check (Train-Val gap)
7. Learning rate

**Quick anomaly thresholds** (see `training_monitor.md` for full table):

| Check | Warning | Critical |
|-------|---------|----------|
| Process count | 0 = crashed | — |
| Val Loss | >1.2 or 5-epoch uptrend | >1.5 |
| Train-Val Gap | >0.8 (overfitting) | >1.0 |
| gen_acc_purpose | <0.78 | <0.75 |
| gen_acc_mode | <0.52 | <0.50 |
| backward failed (recent 10ep) | 10-20 | >20 |
| loss is NaN | any nonzero | — |
| GPU temp | 80-85°C | >85°C |

**Historical best** (updated Epoch 136, 2026-02-25):

| Metric | Best | Epoch | Current (Ep136) |
|--------|------|-------|-----------------|
| Val Loss | 0.9964 | 117 | ~1.05 |
| TF Val Loss | 0.9654 | 117 | ~1.01 |
| purpose | 0.8055 | 129 | 0.8041 |
| mode | 0.5640 | 134 | 0.5640 |
| driver | 0.8734 | 99 | 0.8687 |
| joint | 0.9440 | 114 | 0.9297 |
| destination | 0.8063 | 129 | 0.8060 |
| start_time | 0.3394 | 124 | 0.3515 |
| end_time | 0.3360 | 124 | 0.3501 |

