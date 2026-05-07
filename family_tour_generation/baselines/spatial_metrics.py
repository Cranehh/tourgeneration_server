"""§5 目的地空间分布：real vs main + diff 三联 choropleth。

shp 来源：数据/北京市交通小区/北京市交通小区.shp（2006 个交通小区，EPSG:3857）。
zoneID（字符串）与 arr29[..., 28] 的整数 zone (0..2005) 的映射 = shp 行索引。
zone=2006 是 "unknown/padding" 标识，过滤掉。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from common import NUM_ZONES, ACTIVITY_CORE_DIM


def compute_zone_freq(arr29: np.ndarray, is_real_mask: np.ndarray) -> np.ndarray:
    """目的地频率分布 → 归一化概率向量 [NUM_ZONES]"""
    flat = arr29.reshape(-1, 29)
    if is_real_mask is not None:
        flat = flat[is_real_mask.reshape(-1).astype(bool)]
    dests = flat[:, 28].astype(int)
    dests = dests[(dests >= 0) & (dests < NUM_ZONES)]
    counts = np.bincount(dests, minlength=NUM_ZONES).astype(np.float64)
    total = counts.sum()
    return counts / max(total, 1.0)


def plot_destination_choropleth_pair(real_freq: np.ndarray, main_freq: np.ndarray,
                                      shp_path: str, save_path: str,
                                      use_latlon: bool = True):
    """3 联 choropleth: Real / Main / Diff(Main - Real)

    Real / Main: 用 LogNorm 颜色，零值灰
    Diff: 用 RdBu_r 发散色标，0 居中
    """
    import geopandas as gpd

    print(f"  Loading {shp_path} ...")
    gdf = gpd.read_file(shp_path)
    if use_latlon:
        gdf = gdf.to_crs("EPSG:4326")
    assert len(gdf) == NUM_ZONES, f"shp rows {len(gdf)} != NUM_ZONES {NUM_ZONES}"

    gdf = gdf.copy()
    gdf["real_freq"] = real_freq
    gdf["main_freq"] = main_freq
    gdf["diff"] = main_freq - real_freq

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # ---- Panel 1: Real ----
    ax = axes[0]
    nonzero_real = real_freq[real_freq > 0]
    vmin = nonzero_real.min() if len(nonzero_real) > 0 else 1e-6
    vmax = max(real_freq.max(), main_freq.max())
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    gdf.plot(column="real_freq", ax=ax, cmap="YlOrRd", norm=norm,
             edgecolor="white", linewidth=0.05, missing_kwds={"color": "lightgrey"})
    ax.set_title(f"Real (n_unique_zones={int((real_freq > 0).sum())})", fontsize=12)
    ax.set_axis_off()

    # ---- Panel 2: Main ----
    ax = axes[1]
    gdf.plot(column="main_freq", ax=ax, cmap="YlOrRd", norm=norm,
             edgecolor="white", linewidth=0.05, missing_kwds={"color": "lightgrey"})
    ax.set_title(f"Ours (Main) (n_unique_zones={int((main_freq > 0).sum())})", fontsize=12)
    ax.set_axis_off()

    # 共享 colorbar (前两幅)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:2].tolist(), shrink=0.7, location="bottom", pad=0.05)
    cbar.set_label("Destination frequency (log scale)")

    # ---- Panel 3: Diff ----
    ax = axes[2]
    diff = main_freq - real_freq
    vabs = max(abs(diff.min()), abs(diff.max()))
    diff_norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    gdf.plot(column="diff", ax=ax, cmap="RdBu_r", norm=diff_norm,
             edgecolor="white", linewidth=0.05)
    ax.set_title("Diff (Main - Real)", fontsize=12)
    ax.set_axis_off()

    sm2 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=diff_norm)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=ax, shrink=0.7, location="bottom", pad=0.05)
    cbar2.set_label("Probability difference (Main - Real)")

    plt.suptitle("Destination spatial distribution comparison", fontsize=14, y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def zone_distribution_summary(real_freq: np.ndarray, main_freq: np.ndarray) -> dict:
    """概率分布的小型摘要：top-k 重叠率、KL、Jensen-Shannon"""
    eps = 1e-10
    p = real_freq + eps
    q = main_freq + eps
    p = p / p.sum()
    q = q / q.sum()

    kl = float(np.sum(p * np.log(p / q)))
    m = 0.5 * (p + q)
    js = 0.5 * float(np.sum(p * np.log(p / m))) + 0.5 * float(np.sum(q * np.log(q / m)))

    top20_real = set(np.argsort(real_freq)[-20:])
    top20_main = set(np.argsort(main_freq)[-20:])
    top20_overlap = len(top20_real & top20_main) / 20

    return {
        "KL_real_vs_main": kl,
        "JS_divergence": js,
        "top20_overlap": top20_overlap,
        "n_unique_zones_real": int((real_freq > 0).sum()),
        "n_unique_zones_main": int((main_freq > 0).sum()),
    }
