"""§1 家庭聚合 SRMSE 驱动：对 9 个模型计算 family-level 5D 联合 SRMSE。

输出 results/family_srmse_table.csv，与个人 trip SRMSE 表配套。
"""
import os
import numpy as np
import pandas as pd

from common import load_split, pack_unconditional, unpack_unconditional, assemble_activity_29
from family_srmse import aggregate_family_features, evaluate_family_srmse_all, FAMILY_FEATURES

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ORDER = [
    "real", "main", "main_no_nash",
    "cgan", "cwgan", "cvae",
    "gan", "wgan", "vae",
]


def main():
    print("=" * 70)
    print("Loading test set + building real arr29 ...")
    test_d = load_split("test")
    flat = pack_unconditional(test_d)
    real_unp = unpack_unconditional(flat)
    real_arr29 = assemble_activity_29(real_unp, test_d["home_zone"], test_d["member_mask"])
    is_real_mask = (real_arr29[..., :27].sum(axis=-1) != 0)  # [N, 8, 6]
    print(f"real_arr29: {real_arr29.shape}, is_real total: {is_real_mask.sum()}")

    arr_dict = {"real": real_arr29}
    for m in MODEL_ORDER[1:]:
        path = os.path.join(SAMPLES_DIR, f"{m}.npy")
        if os.path.exists(path):
            arr_dict[m] = np.load(path)
            print(f"  loaded {m}: {arr_dict[m].shape}")
        else:
            print(f"  WARN: {m}.npy missing — skip")

    # Sanity check on real
    real_df = aggregate_family_features(real_arr29, is_real_mask)
    print("\nReal data feature distribution check:")
    for col in FAMILY_FEATURES:
        vc = real_df[col].value_counts().sort_index()
        print(f"  {col}: {dict(vc)}")

    # Compute family SRMSE
    print("\n" + "=" * 70)
    print("Computing family-aggregated SRMSE on 5D joint ...")
    df = evaluate_family_srmse_all(arr_dict, is_real_mask)
    out_csv = os.path.join(RESULTS_DIR, "family_srmse_table.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
