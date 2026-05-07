"""加载 baselines/samples/*.npy，对每个 baseline 计算 4 套 SRMSE，输出汇总 CSV

与 notebook cell 67 一致：用 raw test 的 is_real mask 对齐 raw 和 synth 的行（每个模型只保留
ground-truth 标记为有效的活动位置，确保两端行数严格匹配）。
"""
import argparse
import os
import numpy as np
import pandas as pd

from common import load_split, pack_unconditional, unpack_unconditional, assemble_activity_29
from srmse_metric import activity_tensor_to_srmse_df, evaluate_4_srmse

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_raw_df_and_mask():
    """构建 ground truth DF（与 notebook results_activity_df 同构）和 is_real mask
    (mask shape: [N*8*6])，供所有模型共享对齐。
    """
    d = load_split("test")
    flat = pack_unconditional(d)
    unpacked = unpack_unconditional(flat)
    arr29 = assemble_activity_29(unpacked, d["home_zone"], d["member_mask"])
    is_real = (arr29[..., :27].sum(axis=-1) != 0).reshape(-1)
    raw_df = activity_tensor_to_srmse_df(arr29, is_real_mask=is_real)
    return raw_df, is_real, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["gan", "wgan", "vae", "cgan", "cwgan", "cvae"])
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "srmse_table.csv"))
    args = p.parse_args()

    print("Building ground truth + is_real mask from test set ...")
    raw_df, is_real, _ = build_raw_df_and_mask()
    print(f"raw_df shape: {raw_df.shape}, is_real total: {is_real.sum()}")

    rows = []
    for name in args.models:
        path = os.path.join(SAMPLES_DIR, f"{name}.npy")
        if not os.path.exists(path):
            print(f"  [{name}] not found, skip")
            continue
        arr29 = np.load(path)
        # 用 ground truth 的 is_real mask 过滤 → 与 notebook cell 67 一致
        synth_df = activity_tensor_to_srmse_df(arr29, is_real_mask=is_real)
        print(f"  [{name}] valid synthetic activities: {len(synth_df)} (mask-aligned)")
        if len(synth_df) == 0:
            print(f"  [{name}] WARNING: 0 valid activities, baseline may have failed")
            row = {"model": name, "n_synth_valid": 0,
                   "SRMSE_4D": np.nan, "SRMSE_5D_purpose": np.nan,
                   "SRMSE_5D_mode": np.nan, "SRMSE_8D_full": np.nan}
        else:
            res = evaluate_4_srmse(raw_df, synth_df)
            row = {"model": name, "n_synth_valid": len(synth_df), **res}
        rows.append(row)
        print(f"  [{name}] {row}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
