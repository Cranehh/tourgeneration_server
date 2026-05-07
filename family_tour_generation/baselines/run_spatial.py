"""§5 驱动：real vs main 目的地空间分布 choropleth + 简要摘要"""
import os
import numpy as np
import pandas as pd

from common import load_split, pack_unconditional, unpack_unconditional, assemble_activity_29
from spatial_metrics import (
    compute_zone_freq, plot_destination_choropleth_pair, zone_distribution_summary,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SPATIAL_DIR = os.path.join(RESULTS_DIR, "spatial")
os.makedirs(SPATIAL_DIR, exist_ok=True)
SHP_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                         "数据", "北京市交通小区", "北京市交通小区.shp")


def main():
    print("Loading test set + real arr29 ...")
    test_d = load_split("test")
    flat = pack_unconditional(test_d)
    real_unp = unpack_unconditional(flat)
    real_arr29 = assemble_activity_29(real_unp, test_d["home_zone"], test_d["member_mask"])
    is_real_mask = (real_arr29[..., :27].sum(axis=-1) != 0)

    print("Loading main.npy ...")
    main_arr29 = np.load(os.path.join(SAMPLES_DIR, "main.npy"))

    print("Computing zone frequencies ...")
    real_freq = compute_zone_freq(real_arr29, is_real_mask)
    main_freq = compute_zone_freq(main_arr29, is_real_mask)

    summary = zone_distribution_summary(real_freq, main_freq)
    print("\nDistribution summary (real vs main):")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save summary
    pd.DataFrame([summary]).to_csv(os.path.join(SPATIAL_DIR, "zone_summary.csv"), index=False)

    print("\nPlotting choropleth ...")
    plot_destination_choropleth_pair(
        real_freq, main_freq, SHP_PATH,
        os.path.join(SPATIAL_DIR, "destination_choropleth.png"),
    )
    print(f"  saved: {SPATIAL_DIR}/destination_choropleth.png + zone_summary.csv")


if __name__ == "__main__":
    main()
