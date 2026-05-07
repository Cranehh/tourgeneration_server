"""TR-C 论文 Results 补充实验：B/C/D/E/G 五项指标计算 + 绘图

输入约定：activity_29 [N, 8, 6, 29] np.ndarray，与 baselines 流程一致。
ground truth mask：从真实测试集 activity_29 cols 0:27 sum != 0 计算（与 notebook cell 67 对齐）。

输出：所有 PNG / CSV 写到 baselines/results/extra_metrics/ 子目录。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 中文字体（DiT env 可用 DejaVu Sans，中文回退到 SimHei 等）
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from common import (
    TIME_MEAN, TIME_STD, ACTIVITY_CORE_DIM,
    NUM_PURPOSE, NUM_MODE, NUM_DRIVER, NUM_JOINT, NUM_ZONES,
    MAX_MEMBERS, MAX_ACTIVITIES,
)
from srmse_metric import activity_tensor_to_srmse_df

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "extra_metrics")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------- 工具 ----------------------

def get_is_real_mask_from_real(real_arr29):
    """从真实测试集 [N, 8, 6, 29] 计算 is_real mask [N*8*6]，与 notebook cell 67 一致"""
    return (real_arr29[..., :ACTIVITY_CORE_DIM].sum(axis=-1) != 0).reshape(-1)


def to_dataframe_aligned(arr29, is_real_mask):
    """用 ground-truth is_real mask 对齐过滤，返回 SRMSE-ready 8 列 DataFrame"""
    return activity_tensor_to_srmse_df(arr29, is_real_mask=is_real_mask)


# ---------------------- B: 1D 边际分布 ----------------------

ATTR_DEFS = [
    ("activity_出发时间1小时时间段", "Departure hour", list(range(0, 24))),
    ("activity_到达时间1小时时间段", "Arrival hour", list(range(0, 24))),
    ("purpose", "Purpose", list(range(NUM_PURPOSE))),
    ("mode", "Mode", list(range(NUM_MODE))),
    ("driver", "Driver", [0, 1]),
    ("joint", "Joint", [0, 1]),
    ("出发地", "Origin (top 20)", "top20"),
    ("目的地", "Destination (top 20)", "top20"),
]


def _attr_distribution(df, col, bins):
    if bins == "top20":
        # 用 real_df 的 top 20 作为统一 bins，由调用方传入
        raise ValueError("top20 should be resolved by caller")
    counts = df[col].value_counts().reindex(bins, fill_value=0)
    return counts.values / max(counts.sum(), 1)


def plot_marginals(model_dfs: dict, save_path: str):
    """4×2 子图柱状图，对比所有模型的 1D 边际分布。图例放在图顶部居中。

    model_dfs: {model_name: DataFrame}（必须包含 'real' 作为基准）
    """
    real_df = model_dfs["real"]
    fig, axes = plt.subplots(4, 2, figsize=(16, 19))
    axes = axes.flatten()
    handles_for_legend = None
    labels_for_legend = None

    for ax_idx, (col, title, bins) in enumerate(ATTR_DEFS):
        ax = axes[ax_idx]
        if bins == "top20":
            top = real_df[col].value_counts().nlargest(20).index.tolist()
            bins_list = top
        else:
            bins_list = bins

        n_models = len(model_dfs)
        n_bins = len(bins_list)
        width = 0.8 / n_models

        for i, (name, df) in enumerate(model_dfs.items()):
            counts = df[col].value_counts().reindex(bins_list, fill_value=0)
            freq = counts.values / max(counts.sum(), 1)
            x = np.arange(n_bins) + (i - n_models / 2) * width + width / 2
            ax.bar(x, freq, width=width, label=name, alpha=0.85)

        ax.set_xticks(np.arange(n_bins))
        ax.set_xticklabels([str(b) for b in bins_list], rotation=45 if n_bins > 12 else 0, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()

    # 顶部居中放图例（不挡数据）
    fig.legend(handles_for_legend, labels_for_legend,
               loc="upper center", ncol=len(model_dfs), fontsize=10,
               bbox_to_anchor=(0.5, 0.985), frameon=True)
    plt.suptitle("Marginal Distribution Comparison (1D)", fontsize=14, y=0.999)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------- C: 家庭级联合性 ----------------------

CAR_MODE_INDICES = [7]  # 与 config.py nash_config['car_mode_indices'] 一致


def family_coord_metrics(arr29: np.ndarray, family_attr: np.ndarray, member_mask: np.ndarray,
                          home_zone: np.ndarray, real_mask_per_act: np.ndarray = None):
    """计算 4 个家庭级联合性指标

    arr29: [N, 8, 6, 29]
    family_attr: [N, 10] z-score 标准化后的家庭属性
        col 0 = 家庭成员数（z-score）, col 2 = 机动车数（z-score）, col 8/9 = have_student one-hot
    member_mask: [N, 8] bool（真实成员有效性）
    home_zone: [N] int
    real_mask_per_act: [N, 8, 6] bool, optional, 若提供则只在 ground-truth 有效位置计算

    返回: dict 含 P_joint, V_vio, H_vio, P_pc 4 个标量
    """
    N = arr29.shape[0]
    # 反 z-score 还原家庭属性（用 PopulationDataEncoder fit 的 scaler 常量近似）
    # 家庭成员数标准化前均值约 2.95，方差约 1.32（从 train 数据估）；机动车数 0.50/0.53
    # 这里我们直接用相对比较：找出 have_student=1（col 8/9 中较大那个为正类索引 1）
    have_student = family_attr[:, 9] > family_attr[:, 8]  # one-hot 第二列为 "有学生"

    # 活动级 mask
    if real_mask_per_act is None:
        act_real = arr29[..., :27].sum(axis=-1) != 0  # [N, 8, 6]
    else:
        act_real = real_mask_per_act

    # ---- 1) P_joint: 联合出行比例（每户家庭内 joint==1 的活动占有效活动的比例，再求户级平均）----
    joint = arr29[..., 25:27].argmax(axis=-1)  # 0 or 1; 1 = joint travel
    fam_joint_count = (joint * act_real.astype(int)).sum(axis=(1, 2))  # [N]
    fam_act_count = act_real.sum(axis=(1, 2))  # [N]
    valid_fam = fam_act_count > 0
    P_joint = (fam_joint_count[valid_fam] / fam_act_count[valid_fam]).mean()

    # ---- 2) V_vio: 车辆约束违反率 ----
    # 反解车辆数：col 2 是 z-score，原始大致整数 [0, 5]
    # 用经验常量：mean ≈ 0.502, std ≈ 0.532（从 train 估计；与 train_with_ss_rollout.py 中的 0.53169051/0.50155405 一致）
    num_vehicles = np.round(family_attr[:, 2] * 0.53169051 + 0.50155405).clip(0, 5).astype(int)  # [N]
    driver = arr29[..., 23:25].argmax(axis=-1)  # 0/1
    mode = arr29[..., 12:23].argmax(axis=-1)  # 0..10
    is_car = np.isin(mode, CAR_MODE_INDICES)  # [N, 8, 6]
    drives_car = (driver == 1) & is_car & act_real  # [N, 8, 6]
    # 简化：每户的"开车人次"（不分时间段聚合）
    drivers_per_fam = drives_car.sum(axis=(1, 2))  # [N]
    # 简单约束：开车人次 > 车辆数 视为违反（更严格的"同一时刻"版本需活动重叠分析）
    V_vio = (drivers_per_fam > num_vehicles).mean()

    # ---- 3) H_vio: home-anchor 违反率 ----
    # 每位有效成员的最后一个有效活动的 dest 是否 == home_zone
    last_act_idx = act_real.cumsum(axis=2).argmax(axis=2)  # [N, 8] 最后真实活动的 t
    last_act_idx = np.clip(last_act_idx, 0, MAX_ACTIVITIES - 1)
    has_any_act = act_real.any(axis=2)  # [N, 8]
    valid_member = member_mask & has_any_act  # 既是真实成员又至少 1 个活动

    # 提取每个 (n, m) 的最后活动 dest
    n_idx, m_idx = np.indices((N, MAX_MEMBERS))
    last_dest = arr29[n_idx, m_idx, last_act_idx, 28].astype(int)  # [N, 8]
    home_broadcast = home_zone[:, None]  # [N, 1]
    not_anchored = (last_dest != home_broadcast) & valid_member
    total_valid = valid_member.sum()
    H_vio = not_anchored.sum() / max(total_valid, 1)

    # ---- 4) P_pc: 父母带孩子比例（有学生家庭中至少 1 个 joint==1 活动的家庭比例）----
    fam_has_joint_act = (joint * act_real.astype(int)).sum(axis=(1, 2)) > 0  # [N]
    has_student_total = have_student.sum()
    if has_student_total > 0:
        P_pc = fam_has_joint_act[have_student].mean()
    else:
        P_pc = np.nan

    return dict(P_joint=float(P_joint), V_vio=float(V_vio), H_vio=float(H_vio), P_pc=float(P_pc))


# ---------------------- C-extended: 4 个新家庭联合性指标 (M5-M8) ----------------------

SCHOOL_PURPOSE_IDX = 2  # 来自 activity_pattern_gmm.py:116 ('purpose_school' 在第 3 位)


def chaperone_accuracy(arr29: np.ndarray, family_attr: np.ndarray, member_attr: np.ndarray,
                        member_mask: np.ndarray, real_mask_per_act: np.ndarray = None) -> float:
    """M5: have_student==1 家庭中，孩子的 school 活动是否被家长陪同。

    陪同定义（空间+时间双重匹配）：
      - 同家庭另一位成员在 ±1 小时内有活动到达"学校所在 zone"（dest==child's school dest）
      - 这捕捉"家长送学校→自己继续行程"的典型陪同场景
    "孩子"：family 内 age z-score 最小（最年轻）的成员。
    返回标量：被陪同 / 全部 school 活动。
    """
    have_student = family_attr[:, 9] > family_attr[:, 8]
    if real_mask_per_act is None:
        act_real = arr29[..., :27].sum(axis=-1) != 0
    else:
        act_real = real_mask_per_act

    purpose = arr29[..., 2:12].argmax(axis=-1)
    dest = arr29[..., 28].astype(int)
    start_z = arr29[..., 0]
    start_h = np.maximum(np.round(start_z * TIME_STD + TIME_MEAN), 0).astype(int)

    age_z = member_attr[:, :, 0]
    age_masked = np.where(member_mask, age_z, np.inf)
    youngest_idx = age_masked.argmin(axis=1)

    n_school_total = 0
    n_chaperoned = 0
    for n in np.where(have_student)[0]:
        ym = int(youngest_idx[n])
        for t in range(MAX_ACTIVITIES):
            if not act_real[n, ym, t] or purpose[n, ym, t] != SCHOOL_PURPOSE_IDX:
                continue
            n_school_total += 1
            child_dep = int(start_h[n, ym, t])
            child_dest = int(dest[n, ym, t])
            chaperoned = False
            for m_other in range(MAX_MEMBERS):
                if m_other == ym or not member_mask[n, m_other]:
                    continue
                for t_o in range(MAX_ACTIVITIES):
                    if not act_real[n, m_other, t_o]:
                        continue
                    if int(dest[n, m_other, t_o]) != child_dest:
                        continue
                    other_start = int(start_h[n, m_other, t_o])
                    if abs(other_start - child_dep) <= 1:
                        chaperoned = True
                        break
                if chaperoned:
                    break
            if chaperoned:
                n_chaperoned += 1
    if n_school_total == 0:
        return float("nan")
    return n_chaperoned / n_school_total


def tour_structure(arr29: np.ndarray, member_mask: np.ndarray, home_zone: np.ndarray,
                    real_mask_per_act: np.ndarray = None) -> float:
    """M6: 每位有效成员的 home-based tour 数（户级平均）。

    Tour 定义：从 home 出发、回到 home 的活动闭环。
    用 origin/dest 列在 t 序列上检测：dest[t]==home → 一个 tour 闭合。
    返回标量：所有有效成员的平均 tour 数。
    """
    if real_mask_per_act is None:
        act_real = arr29[..., :27].sum(axis=-1) != 0
    else:
        act_real = real_mask_per_act
    dest = arr29[..., 28].astype(int)                  # [N, 8, 6]
    n_tours_per_member = np.zeros(arr29.shape[:2], dtype=np.int32)  # [N, 8]
    for n in range(arr29.shape[0]):
        h = home_zone[n]
        for m in range(MAX_MEMBERS):
            if not member_mask[n, m]:
                continue
            valid_t = np.where(act_real[n, m])[0]
            if valid_t.size == 0:
                continue
            for t in valid_t:
                if dest[n, m, t] == h:
                    n_tours_per_member[n, m] += 1
    valid_members = member_mask & act_real.any(axis=2)
    if valid_members.sum() == 0:
        return float("nan")
    return float(n_tours_per_member[valid_members].mean())


def joint_group_size(arr29: np.ndarray, member_mask: np.ndarray,
                      real_mask_per_act: np.ndarray = None) -> float:
    """M7: 联合出行的平均同行人数（mean joint group size）。

    对每条 joint==1 的活动，统计同家庭、同时间段（出发 hour bucket 一致）joint==1 的成员数；
    返回所有 joint 活动的人数均值。
    """
    if real_mask_per_act is None:
        act_real = arr29[..., :27].sum(axis=-1) != 0
    else:
        act_real = real_mask_per_act
    joint = arr29[..., 25:27].argmax(axis=-1)
    start_z = arr29[..., 0]
    start_hour = np.maximum(np.round(start_z * TIME_STD + TIME_MEAN), 0).astype(int)

    sizes = []
    for n in range(arr29.shape[0]):
        # 收集每户每小时的 joint 成员
        bucket = {}
        for m in range(MAX_MEMBERS):
            if not member_mask[n, m]:
                continue
            for t in range(MAX_ACTIVITIES):
                if not act_real[n, m, t] or joint[n, m, t] != 1:
                    continue
                h = int(start_hour[n, m, t])
                bucket.setdefault(h, set()).add(m)
        for h, members in bucket.items():
            if len(members) >= 1:
                sizes.append(len(members))
    if not sizes:
        return float("nan")
    return float(np.mean(sizes))


def parent_child_temporal_overlap(arr29: np.ndarray, family_attr: np.ndarray,
                                   member_attr: np.ndarray, member_mask: np.ndarray,
                                   real_mask_per_act: np.ndarray = None) -> float:
    """M8: 父母-子女出行时段重叠率（户级平均）。

    在 have_student==1 家庭中：
      - 子女 = 家庭内 age z-score 最小（最年轻）的成员
      - 父母 = 其他有效成员的并集
    各自的 out-of-home 时段 = 所有有效非 home 活动的 [start, end] 并集
    重叠率 = |child ∩ parent| / |child ∪ parent|（hour 桶上的离散计算）
    返回标量：户级平均重叠率（IoU）。
    """
    if real_mask_per_act is None:
        act_real = arr29[..., :27].sum(axis=-1) != 0
    else:
        act_real = real_mask_per_act
    have_student = family_attr[:, 9] > family_attr[:, 8]
    age_z = member_attr[:, :, 0]
    age_masked = np.where(member_mask, age_z, np.inf)
    youngest_idx = age_masked.argmin(axis=1)
    purpose = arr29[..., 2:12].argmax(axis=-1)
    start_z = arr29[..., 0]
    end_z = arr29[..., 1]

    def member_oh_window(n, m):
        """成员 m 的 out-of-home 小时占用，返回 set of hour buckets 0..23"""
        occupied = set()
        for t in range(MAX_ACTIVITIES):
            if not act_real[n, m, t] or purpose[n, m, t] == 0:  # 0 = home
                continue
            sh = max(0, int(round(float(start_z[n, m, t]) * TIME_STD + TIME_MEAN)))
            eh = max(0, int(round(float(end_z[n, m, t]) * TIME_STD + TIME_MEAN)))
            occupied.update(range(min(sh, eh), max(sh, eh) + 1))
        return occupied

    ratios = []
    for n in np.where(have_student)[0]:
        ym = int(youngest_idx[n])
        child = member_oh_window(n, ym)
        parent = set()
        for m_other in range(MAX_MEMBERS):
            if m_other == ym or not member_mask[n, m_other]:
                continue
            parent |= member_oh_window(n, m_other)
        union = child | parent
        inter = child & parent
        if not union:
            continue
        ratios.append(len(inter) / len(union))
    if not ratios:
        return float("nan")
    return float(np.mean(ratios))


def family_coord_metrics_extended(arr29, family_attr, member_attr, member_mask, home_zone,
                                    real_mask_per_act=None) -> dict:
    """组合调用：原有 4 指标 + 4 个新指标，返回 8 字段 dict"""
    base = family_coord_metrics(arr29, family_attr, member_mask, home_zone, real_mask_per_act)
    base["P_chap"] = chaperone_accuracy(arr29, family_attr, member_attr, member_mask, real_mask_per_act)
    base["n_tour"] = tour_structure(arr29, member_mask, home_zone, real_mask_per_act)
    base["mean_joint_size"] = joint_group_size(arr29, member_mask, real_mask_per_act)
    base["P_overlap"] = parent_child_temporal_overlap(arr29, family_attr, member_attr, member_mask, real_mask_per_act)
    return base


def plot_family_coord_bars(coord_df: pd.DataFrame, save_path: str):
    """2×4 柱状图：每子图一个指标，9 模型并列。指标顺序与列顺序一致。"""
    metrics = ["P_joint", "V_vio", "H_vio", "P_pc", "P_chap", "n_tour", "mean_joint_size", "P_overlap"]
    titles = {
        "P_joint": "P_joint (joint trip rate)",
        "V_vio": "V_vio (vehicle violation)",
        "H_vio": "H_vio (home anchor violation)",
        "P_pc": "P_pc (parent-child joint family rate)",
        "P_chap": "P_chap (chaperone accuracy)",
        "n_tour": "n_tour (avg home-based tours/member)",
        "mean_joint_size": "mean joint group size",
        "P_overlap": "P_overlap (parent-child time IoU)",
    }
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    models = coord_df["model"].tolist()
    n_models = len(models)
    for i, met in enumerate(metrics):
        ax = axes[i]
        if met not in coord_df.columns:
            ax.set_title(f"{titles.get(met, met)} (missing)")
            continue
        vals = coord_df[met].values
        colors = ["#444444" if m == "real" else "#2c7fb8" for m in models]
        ax.bar(np.arange(n_models), vals, color=colors, alpha=0.85)
        ax.set_xticks(np.arange(n_models))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_title(titles.get(met, met), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Family-level coordination metrics (8 total)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------- D: Trip length 分布 ----------------------

def compute_trip_lengths(arr29: np.ndarray, distance_matrix: np.ndarray, is_real_mask: np.ndarray):
    """提取每条有效活动的 (origin, dest) 距离

    arr29: [N, 8, 6, 29]
    distance_matrix: [2006, 2006] z-score 距离
    is_real_mask: [N*8*6] bool
    返回: 1D ndarray of trip distances
    """
    flat = arr29.reshape(-1, 29)
    flat = flat[is_real_mask]
    origin = flat[:, 27].astype(int).clip(0, NUM_ZONES - 1)
    dest = flat[:, 28].astype(int).clip(0, NUM_ZONES - 1)
    return distance_matrix[origin, dest]


def plot_trip_length_kde(model_dists: dict, save_path: str, x_label="Standardized OD distance"):
    """KDE 叠图（每个模型一条线）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, dists in model_dists.items():
        if len(dists) > 0:
            # 用 hist with density=True 模拟 KDE（避免依赖 seaborn）
            ax.hist(dists, bins=80, density=True, histtype="step", linewidth=1.8, label=name)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_title("Trip length distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def trip_length_stats(model_dists: dict) -> pd.DataFrame:
    """每模型的 mean/std/p50/p90"""
    rows = []
    for name, d in model_dists.items():
        if len(d) > 0:
            rows.append({
                "model": name,
                "mean": float(np.mean(d)),
                "std": float(np.std(d)),
                "p50": float(np.percentile(d, 50)),
                "p90": float(np.percentile(d, 90)),
                "n_trips": len(d),
            })
        else:
            rows.append({"model": name, "mean": np.nan, "std": np.nan,
                         "p50": np.nan, "p90": np.nan, "n_trips": 0})
    return pd.DataFrame(rows)


# ---------------------- E: 活动持续时间 ----------------------

def compute_durations(arr29, real_mask_per_act):
    """每条活动的 dwell duration = next_trip_departure(t+1) - this_trip_arrival(t)

    单位 = 小时（z-score 反解：差值 * TIME_STD）。
    只在 (this 有效 AND next 有效) 的位置计算。
    purpose 取当前 trip 的目的（即在目的地从事的活动类型）。

    返回: (dwell_h [n_valid_pairs], purposes [n_valid_pairs])
    """
    arrival_z = arr29[..., 1]                              # [N, M, T]
    next_departure_z = np.zeros_like(arrival_z)
    next_departure_z[..., :-1] = arr29[..., 1:, 0]         # 后移一位
    purpose_idx = arr29[..., 2:12].argmax(axis=-1)         # [N, M, T]

    # 有效条件：this_real AND next_real（最后一位 t=5 一律 invalid）
    next_real = np.zeros_like(real_mask_per_act)
    next_real[..., :-1] = real_mask_per_act[..., 1:]
    valid = real_mask_per_act & next_real

    dwell_h = (next_departure_z - arrival_z) * TIME_STD
    return dwell_h[valid], purpose_idx[valid]


def plot_duration_by_purpose(model_data: dict, save_path: str, max_dur=16):
    """2×5 子图，每个子图一个 purpose，叠加各模型 dwell duration KDE"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()
    handles, labels = None, None
    for p in range(NUM_PURPOSE):
        ax = axes[p]
        for name, (dur, purp) in model_data.items():
            mask = (purp == p) & (dur >= 0) & (dur <= max_dur)
            if mask.sum() > 5:
                ax.hist(dur[mask], bins=40, density=True, histtype="step", linewidth=1.5, label=name)
        ax.set_title(f"Purpose {p}")
        ax.set_xlabel("Dwell duration (hours)")
        ax.set_xlim(0, max_dur)
        ax.grid(alpha=0.3)
        if p == 0:
            handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(model_data),
                   fontsize=9, bbox_to_anchor=(0.5, 0.99), frameon=True)
    plt.suptitle("Activity dwell duration by purpose", fontsize=14, y=0.999)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------- G: Mode share by 子群 ----------------------

def split_by_income(family_attr_zscore):
    """按 col 7 (z-score 收入) 的 33%/67% 分位分箱

    注意：原始收入是分级编码（A-K → 1-5），离散值少，q25 经常正好落在某档边界
    上。改用三分位（1/3, 2/3）更稳健，并用 <= 防止边界严格不等导致空子组。
    """
    income = family_attr_zscore[:, 7]
    q33, q67 = np.percentile(income, [100 / 3, 200 / 3])
    if q33 == q67:  # 极端情况下分位重合
        unique = np.unique(income)
        q33 = unique[len(unique) // 3] if len(unique) >= 3 else unique[0]
        q67 = unique[2 * len(unique) // 3] if len(unique) >= 3 else unique[-1]
    low = income <= q33
    high = income > q67
    mid = ~low & ~high
    return {"low_income": low, "mid_income": mid, "high_income": high}


def split_by_family_type(family_attr_zscore):
    """family type 分类
    col 0 = 家庭成员数（z-score）；用经验常数 mean=2.95, std=1.32 反解
    col 8/9 = have_student one-hot（col 9 表示 yes）
    """
    n_member = np.round(family_attr_zscore[:, 0] * 1.32 + 2.95).clip(1, 8).astype(int)
    has_student = family_attr_zscore[:, 9] > family_attr_zscore[:, 8]
    single = n_member == 1
    couple = (n_member == 2) & (~has_student)
    family = has_student
    other = ~(single | couple | family)
    return {"single": single, "couple": couple, "family_with_children": family, "other": other}


def mode_share(arr29, mask, family_subgroup_mask, real_mask_per_act):
    """对一个家庭子群的所有有效活动统计 mode 频率向量 [11]"""
    sub_mask = family_subgroup_mask[:, None, None] & real_mask_per_act  # [N, 8, 6]
    flat_mask = sub_mask.reshape(-1)
    flat = arr29.reshape(-1, 29)
    if flat_mask.sum() == 0:
        return np.zeros(NUM_MODE)
    sub_arr = flat[flat_mask]
    modes = sub_arr[:, 12:23].argmax(axis=-1)
    cnt = np.bincount(modes, minlength=NUM_MODE)
    return cnt / cnt.sum()


def plot_mode_share_subgroup(model_arr29: dict, family_attr: np.ndarray,
                              real_mask_per_act: np.ndarray, save_dir: str):
    """两张图：按收入 / 按家庭类型分组的 mode share 对比"""
    income_groups = split_by_income(family_attr)
    family_groups = split_by_family_type(family_attr)

    rows = []
    for group_name, groups in [("income", income_groups), ("family_type", family_groups)]:
        n_groups = len(groups)
        n_models = len(model_arr29)
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5), sharey=True)
        if n_groups == 1:
            axes = [axes]
        for ax_idx, (sub_name, sub_mask) in enumerate(groups.items()):
            ax = axes[ax_idx]
            width = 0.8 / n_models
            for i, (m_name, arr) in enumerate(model_arr29.items()):
                share = mode_share(arr, sub_mask, sub_mask, real_mask_per_act)
                x = np.arange(NUM_MODE) + (i - n_models / 2) * width + width / 2
                ax.bar(x, share, width=width, label=m_name, alpha=0.85)
                rows.append({"split": group_name, "subgroup": sub_name, "model": m_name,
                             **{f"mode_{k}": float(share[k]) for k in range(NUM_MODE)}})
            ax.set_xticks(np.arange(NUM_MODE))
            ax.set_xticklabels([str(k) for k in range(NUM_MODE)])
            ax.set_title(f"{sub_name} (n_fam={int(sub_mask.sum())})")
            ax.set_xlabel("Mode")
            if ax_idx == 0:
                ax.set_ylabel("Mode share")
                ax.legend(fontsize=8, ncol=2)
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle(f"Mode share by {group_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mode_share_by_{group_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    return pd.DataFrame(rows)
