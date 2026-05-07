"""快速测试：只跑 §C 家庭联合性 8 指标（4 旧 + 4 新），不跑其他绘图，便于验证 M5-M8。"""
import os
import numpy as np
import pandas as pd

from common import load_split, pack_unconditional, unpack_unconditional, assemble_activity_29
from extra_metrics import OUT_DIR, family_coord_metrics_extended, plot_family_coord_bars

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
MODEL_ORDER = [
    "real", "main", "main_no_nash",
    "cgan", "cwgan", "cvae",
    "gan", "wgan", "vae",
]


def main():
    test_d = load_split("test")
    flat = pack_unconditional(test_d)
    real_unp = unpack_unconditional(flat)
    real_arr29 = assemble_activity_29(real_unp, test_d["home_zone"], test_d["member_mask"])
    real_mask_per_act = (real_arr29[..., :27].sum(axis=-1) != 0)
    print(f"real arr29: {real_arr29.shape}, is_real total: {real_mask_per_act.sum()}")

    all_arr = {"real": real_arr29}
    for m in MODEL_ORDER[1:]:
        path = os.path.join(SAMPLES_DIR, f"{m}.npy")
        if os.path.exists(path):
            all_arr[m] = np.load(path)

    rows = []
    for name, arr in all_arr.items():
        print(f"\n[{name}] computing ...")
        own_mask = (arr[..., :27].sum(axis=-1) != 0) if name == "real" else real_mask_per_act
        m = family_coord_metrics_extended(
            arr, test_d["family_attr"], test_d["member_attr"],
            test_d["member_mask"], test_d["home_zone"],
            real_mask_per_act=own_mask,
        )
        m["model"] = name
        rows.append(m)
        print(f"  P_joint={m['P_joint']:.3f}  V_vio={m['V_vio']:.3f}  H_vio={m['H_vio']:.3f}  P_pc={m['P_pc']:.3f}")
        print(f"  P_chap={m['P_chap']:.3f}  n_tour={m['n_tour']:.3f}  size={m['mean_joint_size']:.2f}  P_overlap={m['P_overlap']:.3f}")

    coord_df = pd.DataFrame(rows)[
        ["model", "P_joint", "V_vio", "H_vio", "P_pc", "P_chap", "n_tour", "mean_joint_size", "P_overlap"]
    ]
    coord_df.to_csv(os.path.join(OUT_DIR, "family_coord.csv"), index=False)
    plot_family_coord_bars(coord_df, os.path.join(OUT_DIR, "family_coord_bars.png"))
    print(f"\nSaved: {OUT_DIR}/family_coord.csv + family_coord_bars.png")
    print(coord_df.to_string(index=False))


if __name__ == "__main__":
    main()
