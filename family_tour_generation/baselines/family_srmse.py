"""家庭聚合 SRMSE：把每户家庭压缩为 5 维特征向量，在 9 模型上计算单一 SRMSE 值

5 个家庭聚合特征：
  total_trips_bin   {0, 1-2, 3-5, 6-10, 11+}                          → 5 档
  joint_ratio_bin   {0, (0,0.25], (0.25,0.5], (0.5,0.75], (0.75,1]}   → 5 档
  dominant_mode     argmax mode share at family level                  → 11 档 (0..10)
  dominant_purpose  argmax purpose share at family level               → 10 档 (0..9)
  n_unique_dests_bin {1, 2, 3-4, 5-7, 8+}                              → 5 档
联合分布：5 × 5 × 11 × 10 × 5 = 13,750 cell。

GT-aligned mask 一致原则：
  与个人 trip SRMSE（notebook cell 67、evaluate_srmse.py）一致，real 和 synth 都用从真实测试集
  推出来的 [N, 8, 6] is_real mask 过滤。这意味着 total_trips_bin 在所有模型上完全一致——
  family-level SRMSE 的判别信号来自其他 4 个特征。
"""
import numpy as np
import pandas as pd

from srmse_metric import calc_srmse

FAMILY_FEATURES = [
    "total_trips_bin",
    "joint_ratio_bin",
    "dominant_mode",
    "dominant_purpose",
    "n_unique_dests_bin",
]


def _bin_total_trips(n: int) -> int:
    if n == 0: return 0
    if n <= 2: return 1
    if n <= 5: return 2
    if n <= 10: return 3
    return 4


def _bin_joint_ratio(r: float) -> int:
    if r <= 0.0: return 0
    if r <= 0.25: return 1
    if r <= 0.5: return 2
    if r <= 0.75: return 3
    return 4


def _bin_n_unique_dests(n: int) -> int:
    if n <= 1: return 0
    if n == 2: return 1
    if n <= 4: return 2
    if n <= 7: return 3
    return 4


def aggregate_family_features(arr29: np.ndarray, is_real_mask: np.ndarray) -> pd.DataFrame:
    """从 [N, 8, 6, 29] 张量聚合每户 5 个特征。

    Args:
        arr29: [N, 8, 6, 29] activity tensor
        is_real_mask: [N, 8, 6] or 一维 [N*48] bool mask（GT-aligned）
    Returns:
        DataFrame[N, 5] 列 = FAMILY_FEATURES
    """
    N = arr29.shape[0]
    mask = is_real_mask.reshape(N, 8, 6).astype(bool)

    feats = np.zeros((N, 5), dtype=np.int64)
    for i in range(N):
        valid = mask[i]
        n_valid = int(valid.sum())
        if n_valid == 0:
            # 全在家：5 个特征都置 0 档
            continue

        acts = arr29[i][valid]  # [n_valid, 29]

        joint_idx = acts[:, 25:27].argmax(axis=-1)  # 0/1
        n_joint = int((joint_idx == 1).sum())
        joint_ratio = n_joint / n_valid

        mode_share = acts[:, 12:23].sum(axis=0)
        dominant_mode = int(mode_share.argmax())

        purpose_share = acts[:, 2:12].sum(axis=0)
        dominant_purpose = int(purpose_share.argmax())

        dests = acts[:, 28].astype(np.int64)
        n_uniq = int(np.unique(dests).size)

        feats[i, 0] = _bin_total_trips(n_valid)
        feats[i, 1] = _bin_joint_ratio(joint_ratio)
        feats[i, 2] = dominant_mode
        feats[i, 3] = dominant_purpose
        feats[i, 4] = _bin_n_unique_dests(n_uniq)

    return pd.DataFrame(feats, columns=FAMILY_FEATURES)


def calc_family_srmse(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    """对 5D 联合分布计算单一 SRMSE 值（取 r=5 的最高维联合）"""
    res = calc_srmse(real_df[FAMILY_FEATURES], synth_df[FAMILY_FEATURES], FAMILY_FEATURES)
    return res[tuple(FAMILY_FEATURES)]


def evaluate_family_srmse_all(arr_dict: dict, is_real_mask: np.ndarray) -> pd.DataFrame:
    """对 dict{model_name → arr29} 全部计算 family SRMSE，返回 DataFrame。
    arr_dict 必须含 'real' 键作 reference。
    """
    if "real" not in arr_dict:
        raise ValueError("arr_dict must contain 'real' key as reference")

    real_df = aggregate_family_features(arr_dict["real"], is_real_mask)

    rows = []
    for name, arr in arr_dict.items():
        synth_df = aggregate_family_features(arr, is_real_mask)
        if name == "real":
            srmse = 0.0
        else:
            srmse = calc_family_srmse(real_df, synth_df)
        rows.append({
            "model": name,
            "family_SRMSE": srmse,
            "n_families": len(synth_df),
        })
    return pd.DataFrame(rows)
