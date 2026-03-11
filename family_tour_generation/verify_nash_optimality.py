"""
验证原始家庭出行调查数据是否接近 Nash Bargaining 最优。

方法：用训练好的效用函数参数，对每个家庭从观测方案出发运行 SQP 优化器，
比较最优方案与观测方案的差距。如果数据已经最优，优化器应几乎不改变结果。

Usage:
    cd family_tour_generation
    python verify_nash_optimality.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path so we can import nash layer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nash_bargaining_sqp_optnet import NashBargainingSQPOptNet


# ========================================================================
#  Constants
# ========================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "数据"
CKPT_PATH = Path(__file__).resolve().parent.parent / "checkpoints_ss_with_condition_nash_without_end3" / "best_model.pt"
OUTPUT_DIR = Path(__file__).resolve().parent / "verification_results"
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Z-score normalization constants for time
TIME_MEAN = 12.946
TIME_STD = 4.691


# ========================================================================
#  Step 1: Load data & parameters
# ========================================================================
def load_data():
    """Load raw .npy data files."""
    family_data = np.load(DATA_DIR / "family_sample_improved_cluster_train.npy")
    member_data = np.load(DATA_DIR / "family_member_sample_improved_cluster_train.npy")
    activity_data = np.load(DATA_DIR / "family_activity_train.npy")
    travel_time_matrix = np.load(DATA_DIR / "travel_time_matrix.npy")

    print(f"Loaded data: families={family_data.shape[0]}", flush=True)
    print(f"  family_data:    {family_data.shape}")
    print(f"  member_data:    {member_data.shape}")
    print(f"  activity_data:  {activity_data.shape}")
    print(f"  travel_time_matrix: {travel_time_matrix.shape}", flush=True)

    return family_data, member_data, activity_data, travel_time_matrix


def load_nash_layer(device):
    """Create Nash layer and load learned parameters from checkpoint."""
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    sd = ckpt['model_state_dict']

    # Extract learned Nash parameters
    alpha_joint = sd['decoder.nash_layer.alpha_joint'].item()
    alpha_social = sd['decoder.nash_layer.alpha_social'].item()
    theta_escort = sd['decoder.nash_layer.theta_escort'].item()
    utility_baseline = sd['decoder.nash_layer.utility_baseline'].item()
    theta_travel = sd['decoder.nash_layer.theta_travel']

    print(f"\nLearned Nash parameters:")
    print(f"  alpha_joint:      {alpha_joint:.4f}")
    print(f"  alpha_social:     {alpha_social:.4f}")
    print(f"  theta_escort:     {theta_escort:.4f}")
    print(f"  utility_baseline: {utility_baseline:.4f}")
    print(f"  theta_travel:     {theta_travel.tolist()}")

    nash_layer = NashBargainingSQPOptNet(
        num_modes=11,
        max_members=8,
        max_activities=6,
        alpha_joint=alpha_joint,
        alpha_social=alpha_social,
        theta_escort=theta_escort,
        utility_baseline=utility_baseline,
        lambda_anchor=0.3,
        joint_consistency_threshold=0.5,
        car_mode_indices=[7],
        sqp_max_iter=30,  # More iterations for thorough convergence
        sqp_tol=1e-5,     # Tighter tolerance
    )

    # Override theta_travel with learned values
    nash_layer.theta_travel.data.copy_(theta_travel)

    nash_layer = nash_layer.to(device)
    nash_layer.eval()
    return nash_layer


# ========================================================================
#  Step 2: Data → Nash input format
# ========================================================================
def prepare_batch(family_data, member_data, activity_data, tt_tensor,
                  indices, device):
    """Convert raw numpy arrays to Nash layer input format for a batch.

    Args:
        tt_tensor: Pre-converted travel time tensor on device [Z, Z, K].
    """
    fam = torch.FloatTensor(family_data[indices]).to(device)       # [B, 81]
    mem = torch.FloatTensor(member_data[indices]).to(device)       # [B, 8, 52]
    act = torch.FloatTensor(activity_data[indices]).to(device)     # [B, 8, 6, 29]

    B = fam.shape[0]

    # --- Extract fields from activity_data ---
    # Channels: [start(1), end(1), purpose_onehot(10), mode_onehot(11),
    #            driver_onehot(2), joint_onehot(2), origin(1), destination(1)]
    start_time = act[..., 0]                    # [B, 8, 6]
    end_time = act[..., 1]                      # [B, 8, 6]
    mode_onehot = act[..., 12:23]               # [B, 8, 6, 11]
    driver_val = act[..., 24]                   # [B, 8, 6] (1=driver in one-hot pos)
    joint_val = act[..., 26]                    # [B, 8, 6] (1=joint in one-hot pos)
    destination = act[..., -1].long()           # [B, 8, 6] zone ID

    # --- Family/member features ---
    home_zone = fam[:, -1].long()                                   # [B]
    # member_data[:, :, 0] is age (standardized, same as member_attr[...,0])
    member_is_adult = ((mem[:, :, 0] * 3.4856215 + 8.42397611) > 3.6).float()  # [B, 8]
    num_vehicles = fam[:, 2] * 0.53169051 + 0.50155405              # [B]

    # --- Masks ---
    # member_mask: valid if any feature is nonzero (excluding work_position col)
    member_mask = (mem[:, :, :-1].abs().sum(-1) > 0).float()         # [B, 8]
    # activity_mask: valid if any of the 27 activity features is nonzero
    activity_mask = (act[..., :27].abs().sum(-1) > 0).float()        # [B, 8, 6]

    # --- Out-of-home (deterministic from known destination) ---
    home_expanded = home_zone.view(B, 1, 1).expand_as(destination)
    out_of_home = (destination != home_expanded).float()              # [B, 8, 6]

    # --- Expected travel time (deterministic: exact from OD pair) ---
    # tt_tensor shape: [Z, Z, K=11]
    Z = tt_tensor.shape[0]
    dest_clamped = destination.clamp(0, Z - 1)    # [B, 8, 6]
    home_clamped = home_zone.clamp(0, Z - 1)      # [B]

    # Gather travel times: tt[home, dest, :] for each activity
    home_exp = home_clamped.view(B, 1, 1).expand(B, 8, 6)
    expected_tt_flat = tt_tensor[home_exp.reshape(-1), dest_clamped.reshape(-1)]  # [B*8*6, K]
    expected_tt = expected_tt_flat.reshape(B, 8, 6, 11)               # [B, 8, 6, K]

    # --- Convert observed data to mode probabilities ---
    # Mode one-hot → treat as probability (already normalized)
    mode_prob = mode_onehot.clamp(min=1e-3)
    mode_prob = mode_prob / mode_prob.sum(dim=-1, keepdim=True)

    # Joint/driver: one-hot index → probability scalar
    joint_prob = joint_val.clamp(1e-3, 1 - 1e-3)
    driver_prob = driver_val.clamp(1e-3, 1 - 1e-3)

    # --- Pack to flat vector [B, n] ---
    # Per activity: [start(1), end(1), mode_prob(11), joint_prob(1), driver_prob(1)] = 15
    packed = torch.cat([
        start_time.unsqueeze(-1),
        end_time.unsqueeze(-1),
        mode_prob,
        joint_prob.unsqueeze(-1),
        driver_prob.unsqueeze(-1),
    ], dim=-1)  # [B, 8, 6, 15]
    x_obs = packed.reshape(B, -1)  # [B, 720]

    return {
        'x_obs': x_obs,
        'member_mask': member_mask,
        'activity_mask': activity_mask,
        'num_vehicles': num_vehicles,
        'home_zone': home_zone,
        'member_is_adult': member_is_adult,
        'out_of_home': out_of_home,
        'expected_tt': expected_tt,
        'start_time_obs': start_time,
        'end_time_obs': end_time,
        'mode_onehot_obs': mode_onehot,
        'joint_obs': joint_val,
        'driver_obs': driver_val,
    }


# ========================================================================
#  Step 3: Run SQP optimization
# ========================================================================
def run_verification(nash_layer, family_data, member_data, activity_data,
                     travel_time_matrix, device):
    """Run SQP optimization on all families, collect results."""
    N = family_data.shape[0]
    num_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE

    all_results = {
        'welfare_obs': [],
        'welfare_opt': [],
        'welfare_gap': [],
        'start_change': [],   # MAE in hours
        'end_change': [],     # MAE in hours
        'mode_match': [],     # fraction of activities where mode argmax matches
        'joint_change': [],   # mean |Δjoint_prob|
        'driver_change': [],  # mean |Δdriver_prob|
        'family_size': [],
        'has_vehicle': [],
        'has_child': [],
        'num_valid_activities': [],
    }

    print(f"\nRunning verification on {N} families ({num_batches} batches)...", flush=True)

    # Pre-convert travel time matrix to tensor once (avoids ~170MB copy per batch)
    tt_tensor = torch.FloatTensor(travel_time_matrix).to(device)

    with torch.no_grad():
        # Detach learnable params for utility computation
        alpha_joint_val = nash_layer.alpha_joint.detach()
        alpha_social_val = nash_layer.alpha_social.detach()
        theta_escort_val = nash_layer.theta_escort.detach()
        utility_baseline_val = nash_layer.utility_baseline.detach()
        theta_travel_val = nash_layer.theta_travel.detach()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, N)
            indices = list(range(start_idx, end_idx))

            batch = prepare_batch(
                family_data, member_data, activity_data,
                tt_tensor, indices, device
            )

            x_obs = batch['x_obs']
            member_mask = batch['member_mask']
            activity_mask = batch['activity_mask']
            var_mask = nash_layer._build_var_mask(activity_mask)

            # Project observed data to feasible region first
            x_obs = nash_layer._project_to_feasible(x_obs)

            # Compute observed welfare
            U_obs = nash_layer._compute_member_utilities_detached(
                x_obs, member_mask, activity_mask,
                batch['member_is_adult'], batch['out_of_home'], batch['expected_tt'],
                alpha_joint_val, alpha_social_val, theta_escort_val,
                utility_baseline_val, theta_travel_val,
            )
            W_obs = nash_layer._compute_nash_welfare(U_obs, member_mask)

            # Run SQP to find optimal
            x_opt = nash_layer._sqp_phase(
                x_obs, var_mask, member_mask, activity_mask,
                batch['num_vehicles'], batch['member_is_adult'],
                batch['out_of_home'], batch['expected_tt'],
            )

            # Compute optimal welfare
            U_opt = nash_layer._compute_member_utilities_detached(
                x_opt, member_mask, activity_mask,
                batch['member_is_adult'], batch['out_of_home'], batch['expected_tt'],
                alpha_joint_val, alpha_social_val, theta_escort_val,
                utility_baseline_val, theta_travel_val,
            )
            W_opt = nash_layer._compute_nash_welfare(U_opt, member_mask)

            # Compute per-family metrics
            gap = (W_opt - W_obs).cpu().numpy()
            all_results['welfare_obs'].extend(W_obs.cpu().numpy().tolist())
            all_results['welfare_opt'].extend(W_opt.cpu().numpy().tolist())
            all_results['welfare_gap'].extend(gap.tolist())

            # Unpack for variable-level comparison
            obs_unpacked = nash_layer._unpack_flat(x_obs)
            opt_unpacked = nash_layer._unpack_flat(x_opt)
            amask = activity_mask.bool()  # [B, 8, 6]

            B_cur = x_obs.shape[0]
            for b in range(B_cur):
                valid = amask[b]  # [8, 6]
                n_valid = valid.sum().item()
                if n_valid == 0:
                    continue

                # Time changes (convert z-score → hours)
                s_obs = obs_unpacked['start_time'][b][valid] * TIME_STD + TIME_MEAN
                s_opt = opt_unpacked['start_time'][b][valid] * TIME_STD + TIME_MEAN
                e_obs = obs_unpacked['end_time'][b][valid] * TIME_STD + TIME_MEAN
                e_opt = opt_unpacked['end_time'][b][valid] * TIME_STD + TIME_MEAN

                all_results['start_change'].append((s_opt - s_obs).abs().mean().item())
                all_results['end_change'].append((e_opt - e_obs).abs().mean().item())

                # Mode match
                mode_obs_argmax = obs_unpacked['mode_prob'][b][valid].argmax(dim=-1)
                mode_opt_argmax = opt_unpacked['mode_prob'][b][valid].argmax(dim=-1)
                all_results['mode_match'].append(
                    (mode_obs_argmax == mode_opt_argmax).float().mean().item()
                )

                # Joint/driver changes
                jp_obs = obs_unpacked['joint_prob'][b][valid]
                jp_opt = opt_unpacked['joint_prob'][b][valid]
                all_results['joint_change'].append((jp_opt - jp_obs).abs().mean().item())

                dp_obs = obs_unpacked['driver_prob'][b][valid]
                dp_opt = opt_unpacked['driver_prob'][b][valid]
                all_results['driver_change'].append((dp_opt - dp_obs).abs().mean().item())

                all_results['num_valid_activities'].append(n_valid)

            # Subgroup info (use already-loaded data from batch)
            family_size = member_mask.sum(dim=-1).cpu().numpy()
            has_vehicle = (batch['num_vehicles'] > 0.5).cpu().numpy()
            # Child: any member with age category <= 3.6 (and is valid member)
            # member_is_adult uses age > 3.6, so child = NOT adult AND valid member
            has_child = ((batch['member_is_adult'] < 0.5) & member_mask.bool()).any(dim=-1).cpu().numpy()

            all_results['family_size'].extend(family_size.tolist())
            all_results['has_vehicle'].extend(has_vehicle.tolist())
            all_results['has_child'].extend(has_child.tolist())

            if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
                print(f"  Batch {batch_idx+1}/{num_batches} done. "
                      f"Mean gap so far: {np.mean(all_results['welfare_gap']):.6f}",
                      flush=True)

    # Convert to numpy
    for k in all_results:
        all_results[k] = np.array(all_results[k])

    return all_results


# ========================================================================
#  Step 4: Analysis & reporting
# ========================================================================
def analyze_and_report(results):
    """Print statistics and generate plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gap = results['welfare_gap']
    N = len(gap)

    # ---- 1. Welfare gap distribution ----
    print("\n" + "=" * 70)
    print("NASH OPTIMALITY VERIFICATION RESULTS")
    print("=" * 70)

    print(f"\n1. Welfare Gap Distribution (W_optimal - W_observed)")
    print(f"   N = {N}")
    print(f"   Mean:   {gap.mean():.6f}")
    print(f"   Median: {np.median(gap):.6f}")
    print(f"   Std:    {gap.std():.6f}")
    print(f"   Min:    {gap.min():.6f}")
    print(f"   Max:    {gap.max():.6f}")
    print(f"   5th percentile:  {np.percentile(gap, 5):.6f}")
    print(f"   25th percentile: {np.percentile(gap, 25):.6f}")
    print(f"   75th percentile: {np.percentile(gap, 75):.6f}")
    print(f"   95th percentile: {np.percentile(gap, 95):.6f}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(gap, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(gap.mean(), color='red', linestyle='--', label=f'Mean={gap.mean():.4f}')
    ax.axvline(np.median(gap), color='orange', linestyle='--', label=f'Median={np.median(gap):.4f}')
    ax.set_xlabel('Welfare Gap (W_optimal - W_observed)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Nash Welfare Gap')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'welfare_gap_hist.png', dpi=150)
    plt.close()

    # ---- 2. Variable-level changes ----
    print(f"\n2. Variable-Level Changes")
    n_act = len(results['start_change'])
    print(f"   Families with valid activities: {n_act}")
    print(f"   Start time MAE (hours): {results['start_change'].mean():.4f} "
          f"(median: {np.median(results['start_change']):.4f})")
    print(f"   End time MAE (hours):   {results['end_change'].mean():.4f} "
          f"(median: {np.median(results['end_change']):.4f})")
    print(f"   Mode match rate:        {results['mode_match'].mean():.4f} "
          f"(median: {np.median(results['mode_match']):.4f})")
    print(f"   Joint prob MAE:         {results['joint_change'].mean():.6f} "
          f"(median: {np.median(results['joint_change']):.6f})")
    print(f"   Driver prob MAE:        {results['driver_change'].mean():.6f} "
          f"(median: {np.median(results['driver_change']):.6f})")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    var_data = [
        (results['start_change'], 'Start Time MAE (hours)', 'Start Time Change'),
        (results['end_change'], 'End Time MAE (hours)', 'End Time Change'),
        (results['mode_match'], 'Mode Match Rate', 'Mode Agreement'),
        (results['joint_change'], 'Joint Prob MAE', 'Joint Travel Change'),
        (results['driver_change'], 'Driver Prob MAE', 'Driver Status Change'),
    ]
    for idx, (data, xlabel, title) in enumerate(var_data):
        ax = axes[idx // 3, idx % 3]
        ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean={data.mean():.4f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'variable_changes.png', dpi=150)
    plt.close()

    # ---- 3. Optimality proportion ----
    print(f"\n3. Optimality Proportion (fraction of families with gap < threshold)")
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    print(f"   {'Threshold':<12} {'Proportion':<12} {'Count':<10}")
    proportions = []
    for th in thresholds:
        prop = (gap < th).mean()
        count = (gap < th).sum()
        proportions.append(prop)
        print(f"   {th:<12.3f} {prop:<12.4f} {count:<10d}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(thresholds, proportions, 'bo-', markersize=6)
    ax.set_xlabel('Gap Threshold')
    ax.set_ylabel('Proportion of Families Below Threshold')
    ax.set_title('Cumulative Optimality Distribution')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    for th, prop in zip(thresholds, proportions):
        ax.annotate(f'{prop:.2%}', (th, prop), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'optimality_proportion.png', dpi=150)
    plt.close()

    # ---- 4. Subgroup analysis ----
    print(f"\n4. Subgroup Analysis")

    # By family size
    sizes = results['family_size']
    unique_sizes = sorted(set(sizes.astype(int)))
    print(f"\n   By family size:")
    print(f"   {'Size':<8} {'N':<8} {'Mean Gap':<12} {'Median Gap':<12} {'Mode Match':<12}")
    size_gaps = {}
    for s in unique_sizes:
        mask = sizes.astype(int) == s
        if mask.sum() < 5:
            continue
        g = gap[mask]
        mm = results['mode_match'][mask[:len(results['mode_match'])]] if mask.sum() <= len(results['mode_match']) else np.array([])
        mm_str = f"{mm.mean():.4f}" if len(mm) > 0 else "N/A"
        print(f"   {s:<8d} {mask.sum():<8d} {g.mean():<12.6f} {np.median(g):<12.6f} {mm_str:<12}")
        size_gaps[s] = g

    # By vehicle ownership
    has_veh = results['has_vehicle'].astype(bool)
    print(f"\n   By vehicle ownership:")
    for label, mask in [("Has vehicle", has_veh), ("No vehicle", ~has_veh)]:
        if mask.sum() == 0:
            continue
        g = gap[mask]
        print(f"   {label:<16} N={mask.sum():<6d} Mean Gap={g.mean():.6f}  Median={np.median(g):.6f}")

    # By presence of children
    has_ch = results['has_child'].astype(bool)
    print(f"\n   By presence of children:")
    for label, mask in [("Has child", has_ch), ("No child", ~has_ch)]:
        if mask.sum() == 0:
            continue
        g = gap[mask]
        print(f"   {label:<16} N={mask.sum():<6d} Mean Gap={g.mean():.6f}  Median={np.median(g):.6f}")

    # Subgroup bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Family size
    if size_gaps:
        sizes_list = sorted(size_gaps.keys())
        means = [size_gaps[s].mean() for s in sizes_list]
        axes[0].bar([str(s) for s in sizes_list], means, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Family Size')
        axes[0].set_ylabel('Mean Welfare Gap')
        axes[0].set_title('Gap by Family Size')

    # Vehicle
    veh_means = []
    veh_labels = []
    for label, mask in [("Has vehicle", has_veh), ("No vehicle", ~has_veh)]:
        if mask.sum() > 0:
            veh_labels.append(label)
            veh_means.append(gap[mask].mean())
    if veh_means:
        axes[1].bar(veh_labels, veh_means, color='coral', edgecolor='black')
        axes[1].set_ylabel('Mean Welfare Gap')
        axes[1].set_title('Gap by Vehicle Ownership')

    # Children
    ch_means = []
    ch_labels = []
    for label, mask in [("Has child", has_ch), ("No child", ~has_ch)]:
        if mask.sum() > 0:
            ch_labels.append(label)
            ch_means.append(gap[mask].mean())
    if ch_means:
        axes[2].bar(ch_labels, ch_means, color='mediumpurple', edgecolor='black')
        axes[2].set_ylabel('Mean Welfare Gap')
        axes[2].set_title('Gap by Presence of Children')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'subgroup_analysis.png', dpi=150)
    plt.close()

    # ---- 5. Top-5 cases with largest gap ----
    print(f"\n5. Top-5 Families with Largest Welfare Gap")
    top_idx = np.argsort(gap)[-5:][::-1]
    for rank, idx in enumerate(top_idx, 1):
        print(f"\n   --- Family #{idx} (Rank {rank}) ---")
        print(f"   Welfare: obs={results['welfare_obs'][idx]:.4f} → "
              f"opt={results['welfare_opt'][idx]:.4f}  gap={gap[idx]:.6f}")
        print(f"   Family size: {int(results['family_size'][idx])}")
        print(f"   Has vehicle: {bool(results['has_vehicle'][idx])}")
        print(f"   Has child: {bool(results['has_child'][idx])}")
        if idx < len(results['start_change']):
            print(f"   Start time MAE: {results['start_change'][idx]:.4f} hours")
            print(f"   End time MAE:   {results['end_change'][idx]:.4f} hours")
            print(f"   Mode match:     {results['mode_match'][idx]:.4f}")
            print(f"   Joint change:   {results['joint_change'][idx]:.6f}")
            print(f"   Driver change:  {results['driver_change'][idx]:.6f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    near_optimal = (gap < 0.01).mean()
    print(f"  {near_optimal:.1%} of families have welfare gap < 0.01")
    print(f"  Mean welfare gap: {gap.mean():.6f}")
    print(f"  Mean start time change: {results['start_change'].mean():.4f} hours")
    print(f"  Mean mode match rate: {results['mode_match'].mean():.4f}")

    if gap.mean() < 0.05:
        print("\n  → Data is approximately Nash optimal — supports modeling assumption.")
    elif gap.mean() < 0.2:
        print("\n  → Data shows moderate deviation from Nash optimal.")
        print("    This suggests some non-optimal behavior or utility function limitations.")
    else:
        print("\n  → Data shows substantial deviation from Nash optimal.")
        print("    The utility function may need refinement, or real families")
        print("    deviate significantly from Nash Bargaining behavior.")

    print(f"\nPlots saved to: {OUTPUT_DIR}/")


# ========================================================================
#  Main
# ========================================================================
def main():
    print("Nash Optimality Verification")
    print(f"Device: {DEVICE}")

    family_data, member_data, activity_data, travel_time_matrix = load_data()
    nash_layer = load_nash_layer(DEVICE)

    results = run_verification(
        nash_layer, family_data, member_data, activity_data,
        travel_time_matrix, DEVICE
    )

    analyze_and_report(results)


if __name__ == '__main__':
    main()
