"""跑 TR-C 投稿 Results §2-§5 的全部补充实验：B / C / D / E / G

输出目录：baselines/results/extra_metrics/
"""
import os
import numpy as np
import pandas as pd

from common import load_split, pack_unconditional, unpack_unconditional, assemble_activity_29
from extra_metrics import (
    OUT_DIR, get_is_real_mask_from_real, to_dataframe_aligned,
    plot_marginals, family_coord_metrics, family_coord_metrics_extended,
    plot_family_coord_bars,
    compute_trip_lengths, plot_trip_length_kde, trip_length_stats,
    compute_durations, plot_duration_by_purpose,
    plot_mode_share_subgroup,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "数据")

# 模型显示名顺序（论文中的呈现顺序）
MODEL_ORDER = [
    "real",
    "main",
    "main_no_nash",
    "cgan",
    "cwgan",
    "cvae",
    "gan",
    "wgan",
    "vae",
]


def main():
    print("=" * 70)
    print("Loading test set + building ground truth ...")
    test_d = load_split("test")
    flat = pack_unconditional(test_d)
    real_unp = unpack_unconditional(flat)
    real_arr29 = assemble_activity_29(real_unp, test_d["home_zone"], test_d["member_mask"])

    is_real_mask = get_is_real_mask_from_real(real_arr29)
    real_mask_per_act = (real_arr29[..., :27].sum(axis=-1) != 0)  # [N, 8, 6]
    print(f"real_arr29: {real_arr29.shape}, is_real total: {is_real_mask.sum()}")

    # 加载所有模型的样本
    all_arr = {"real": real_arr29}
    for m in MODEL_ORDER[1:]:
        path = os.path.join(SAMPLES_DIR, f"{m}.npy")
        if os.path.exists(path):
            all_arr[m] = np.load(path)
            print(f"  loaded {m}: {all_arr[m].shape}")
        else:
            print(f"  WARN: {m}.npy not found, skipping")

    # ===== B: 1D 边际分布 =====
    print("\n[B] Plotting 1D marginal distributions ...")
    all_dfs = {name: to_dataframe_aligned(arr, is_real_mask) for name, arr in all_arr.items()}
    plot_marginals(all_dfs, os.path.join(OUT_DIR, "marginals_8attr.png"))
    print(f"  saved: {OUT_DIR}/marginals_8attr.png")

    # ===== C: 家庭级联合性 (4 旧 + 4 新) =====
    print("\n[C] Computing 8 family coordination metrics ...")
    coord_rows = []
    for name, arr in all_arr.items():
        m = family_coord_metrics_extended(
            arr, test_d["family_attr"], test_d["member_attr"],
            test_d["member_mask"], test_d["home_zone"],
            real_mask_per_act=real_mask_per_act if name != "real" else (arr[..., :27].sum(axis=-1) != 0),
        )
        m["model"] = name
        coord_rows.append(m)
        print(f"  {name}: P_joint={m['P_joint']:.3f}, V_vio={m['V_vio']:.3f}, H_vio={m['H_vio']:.3f}, "
              f"P_pc={m['P_pc']:.3f}, P_chap={m['P_chap']:.3f}, n_tour={m['n_tour']:.3f}, "
              f"size={m['mean_joint_size']:.2f}, P_overlap={m['P_overlap']:.3f}")
    coord_df = pd.DataFrame(coord_rows)[
        ["model", "P_joint", "V_vio", "H_vio", "P_pc",
         "P_chap", "n_tour", "mean_joint_size", "P_overlap"]
    ]
    coord_df.to_csv(os.path.join(OUT_DIR, "family_coord.csv"), index=False)
    plot_family_coord_bars(coord_df, os.path.join(OUT_DIR, "family_coord_bars.png"))
    print(f"  saved: {OUT_DIR}/family_coord.csv + family_coord_bars.png")

    # ===== D: Trip length 分布 =====
    print("\n[D] Computing trip length distributions ...")
    distance_matrix = np.load(f"{DATA_DIR}/position_distance.npy")
    print(f"  distance_matrix: shape={distance_matrix.shape}, range=[{distance_matrix.min():.2f}, {distance_matrix.max():.2f}]")
    model_dists = {}
    for name, arr in all_arr.items():
        # trip length 用各模型自身的 is_real（防止 real 的 mask 把模型预测的"位置错"也丢掉）
        # 但为了对齐 distribution shape，统一用 ground-truth mask
        d = compute_trip_lengths(arr, distance_matrix, is_real_mask)
        # 过滤自环（origin == dest，距离 = 对角值即 -1.55）
        d = d[d > distance_matrix.diagonal().mean() + 0.01]
        model_dists[name] = d
        print(f"  {name}: n={len(d)}, mean={d.mean():.3f}, p50={np.percentile(d, 50):.3f}")
    plot_trip_length_kde(model_dists, os.path.join(OUT_DIR, "trip_length_kde.png"))
    stats_df = trip_length_stats(model_dists)
    stats_df.to_csv(os.path.join(OUT_DIR, "trip_length_stats.csv"), index=False)
    print(f"  saved: {OUT_DIR}/trip_length_kde.png + trip_length_stats.csv")

    # ===== E: 活动持续时间 =====
    print("\n[E] Computing activity dwell duration distributions ...")
    model_durdata = {}
    for name, arr in all_arr.items():
        # dwell time 用 ground-truth real_mask_per_act（保证 next_real 一致对齐）
        dur, purp = compute_durations(arr, real_mask_per_act)
        model_durdata[name] = (dur, purp)
        print(f"  {name}: n_pairs={len(dur)}, mean_dwell={dur.mean():.2f}h, p50={np.percentile(dur, 50) if len(dur) > 0 else 0:.2f}h")
    plot_duration_by_purpose(model_durdata, os.path.join(OUT_DIR, "duration_by_purpose.png"))
    print(f"  saved: {OUT_DIR}/duration_by_purpose.png")

    # ===== G: Mode share by 子群 =====
    print("\n[G] Plotting mode share by income / family type ...")
    sub_df = plot_mode_share_subgroup(
        all_arr, test_d["family_attr"], real_mask_per_act, OUT_DIR,
    )
    sub_df.to_csv(os.path.join(OUT_DIR, "mode_share_subgroup.csv"), index=False)
    print(f"  saved: {OUT_DIR}/mode_share_by_*.png + mode_share_subgroup.csv")

    print("\n" + "=" * 70)
    print(f"All extra metrics computed. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
