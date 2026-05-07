"""
SRMSE 指标模块（与 数据处理21.ipynb cell 108 完全一致）+
活动张量 [N, 8, 6, 29] → SRMSE-ready DataFrame 的解码工具

关键约定：
  - 输入张量 [N, 8, 6, 29]，按 notebook cell 65-75 处理：
      reshape → [N*48, 29]
      filter is_real（前 27 列 sum != 0）
      argmax 还原 purpose / mode / driver / joint
      用 hardcoded TIME_MEAN/STD 反 z-score 还原 hour 时间段
      重命名 col 27 → 出发地, col 28 → 目的地
  - 最终保留 8 列与 notebook 一致：
      ['activity_出发时间1小时时间段','activity_到达时间1小时时间段',
       'purpose','mode','driver','joint','出发地','目的地']
"""
import itertools
import numpy as np
import pandas as pd

from common import TIME_MEAN, TIME_STD, ACTIVITY_CORE_DIM


SRMSE_COLS_4D = [
    "activity_出发时间1小时时间段",
    "activity_到达时间1小时时间段",
    "出发地",
    "目的地",
]
SRMSE_COLS_5D_PURPOSE = SRMSE_COLS_4D + ["purpose"]
SRMSE_COLS_5D_MODE = SRMSE_COLS_4D + ["mode"]
SRMSE_COLS_8D = [
    "activity_出发时间1小时时间段",
    "activity_到达时间1小时时间段",
    "purpose", "mode", "driver", "joint",
    "出发地", "目的地",
]


def calc_srmse(raw_df, synth_df, max_value_list):
    """SRMSE 计算（与 notebook cell 108 完全一致）

    输出：dict, key=(col_tuple), value=srmse；并含 'avg_srmse_{r}_dims' 平均值
    """
    srmse_results = {}
    for r in range(1, len(max_value_list) + 1):
        number_of_combinations = 0
        srmse_sum = 0.0
        for cols in itertools.combinations(raw_df.columns, r):
            real_counts = raw_df.groupby(list(cols)).size().reset_index(name="count")
            synth_counts = synth_df.groupby(list(cols)).size().reset_index(name="count")
            merged = pd.merge(
                real_counts, synth_counts, on=list(cols), how="outer",
                suffixes=("_real", "_synth"),
            ).fillna(0)
            merged["pi_real"] = merged["count_real"]
            merged["pi_synth"] = merged["count_synth"]
            N_cnt = len(merged)
            numerator = np.sqrt(((merged["pi_synth"] - merged["pi_real"]) ** 2).sum() / N_cnt)
            denominator = (merged["pi_real"].sum()) / N_cnt
            srmse = numerator / denominator if denominator != 0 else np.nan
            srmse_results[cols] = srmse
            srmse_sum += srmse
            number_of_combinations += 1
        avg_srmse = srmse_sum / number_of_combinations if number_of_combinations > 0 else np.nan
        srmse_results[f"avg_srmse_{r}_dims"] = avg_srmse
    return srmse_results


def activity_tensor_to_srmse_df(activity_29: np.ndarray, is_real_mask: np.ndarray = None) -> pd.DataFrame:
    """把 [N, 8, 6, 29] 张量解码成 SRMSE-ready DataFrame（8 列）

    流程：
      1. reshape → [N*48, 29]
      2. filter is_real：
         - 默认（is_real_mask=None）：用自身前 27 列 sum != 0 判定（自掩码）
         - 与 notebook 对齐（is_real_mask 传入）：用外部布尔 mask（通常来自真实测试集）过滤，确保 raw 和 synth 行数对齐
      3. cols 0,1 反 z-score → 整数 hour 时间段
      4. cols 2:12 / 12:23 / 23:25 / 25:27 → argmax → 类别
      5. col 27, 28 → 出发地, 目的地（已是整数）
    """
    flat = activity_29.reshape(-1, 29)
    if is_real_mask is None:
        is_real = flat[:, :ACTIVITY_CORE_DIM].sum(axis=-1) != 0
    else:
        is_real = is_real_mask.reshape(-1).astype(bool)
        assert len(is_real) == flat.shape[0], f"mask len {len(is_real)} != flat len {flat.shape[0]}"
    flat = flat[is_real]

    departure_z = flat[:, 0]
    arrival_z = flat[:, 1]
    departure_hour = np.maximum(np.round(departure_z * TIME_STD + TIME_MEAN), 0).astype(int)
    arrival_hour = np.maximum(np.round(arrival_z * TIME_STD + TIME_MEAN), 0).astype(int)

    purpose = flat[:, 2:12].argmax(axis=-1)
    mode = flat[:, 12:23].argmax(axis=-1)
    driver = flat[:, 23:25].argmax(axis=-1)
    joint = flat[:, 25:27].argmax(axis=-1)

    origin = flat[:, 27].astype(int)
    dest = flat[:, 28].astype(int)

    df = pd.DataFrame({
        "activity_出发时间1小时时间段": departure_hour,
        "activity_到达时间1小时时间段": arrival_hour,
        "purpose": purpose,
        "mode": mode,
        "driver": driver,
        "joint": joint,
        "出发地": origin,
        "目的地": dest,
    })
    return df


def evaluate_4_srmse(raw_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """4 套 SRMSE：4D / 5D-purpose / 5D-mode / 8D-Full

    每套返回单一 SRMSE 值（最高 r 维联合分布，与 notebook cell 110/111/112/113 一致）。
    """
    out = {}
    for label, cols in [
        ("SRMSE_4D", SRMSE_COLS_4D),
        ("SRMSE_5D_purpose", SRMSE_COLS_5D_PURPOSE),
        ("SRMSE_5D_mode", SRMSE_COLS_5D_MODE),
        ("SRMSE_8D_full", SRMSE_COLS_8D),
    ]:
        r = calc_srmse(raw_df[cols], synth_df[cols], cols)
        # 最高 r 维联合分布的 SRMSE（即 notebook 4 个 cell 各自打印的那个值）
        out[label] = r[tuple(cols)]
    return out
