"""
Baseline 通用工具：数据加载 + 解码 + 张量打包/拆包

数据形状（与 train_with_ss_rollout.py 对齐）：
  - family raw npy:   [N, 81]  →  np.concatenate([data[:, :10], data[:, -1:]]) → [N, 11]
  - member raw npy:   [N, 8, 52]
  - activity raw npy: [N, 8, 6, 29]
      cols 0:2   连续 z-score (start/end time)
      cols 2:12  purpose one-hot (10)
      cols 12:23 mode one-hot (11)
      cols 23:25 driver one-hot (2)
      cols 25:27 joint one-hot (2)
      col 27     起点 zone (整数 0..2005)
      col 28     终点 zone (整数 0..2005)

约定的 baseline 内部表示：
  - family_attr: [N, 10]  连续/类编码后的家庭特征
  - home_zone:   [N]      0..2005
  - member_attr: [N, 8, 50]  成员特征 (52-1 work_zone -1 mask)
  - work_zone:   [N, 8]   0..2007
  - member_mask: [N, 8]   bool
  - activity_core: [N, 8, 6, 27]  连续 + 4 个 one-hot 类别
  - activity_dest: [N, 8, 6]      destination zone 0..2005
  - activity_mask: [N, 8, 6]      bool

为了让 SRMSE 评估口径与 notebook 一致：
  - 时间反 z-score 用 hardcoded mean=12.946, std=4.691（CLAUDE.md 注明值）
  - 起点 zone：t=0 → home_zone；t≥1 → 上一活动的 destination
"""
import os
import numpy as np
import torch

# z-score 常量（与 notebook PopulationDataEncoder fit 出的值一致）
TIME_MEAN = 12.946
TIME_STD = 4.691

DATA_DIR = os.environ.get("FAMILY_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "数据"))

# 维度
FAMILY_DIM = 10
MEMBER_DIM = 50  # member raw 52 - 1 (work_zone) - 1 (mask flag)
MEMBER_MASK_COL = -2  # 倒数第 2 列是 member_mask
WORK_ZONE_COL = -1   # 倒数第 1 列是 work_zone
MAX_MEMBERS = 8
MAX_ACTIVITIES = 6
ACTIVITY_CORE_DIM = 27  # 2 + 10 + 11 + 2 + 2
NUM_PURPOSE = 10
NUM_MODE = 11
NUM_DRIVER = 2
NUM_JOINT = 2
NUM_ZONES = 2006
HOME_ZONE_INVALID_THRESHOLD = NUM_ZONES  # 2006 / 2007 表示无效


def load_split(split: str):
    """加载 train 或 test，返回字典含所有需要的张量（numpy）"""
    assert split in ("train", "test")
    raw_family = np.load(f"{DATA_DIR}/family_sample_improved_cluster_{split}.npy")
    family_compact = np.concatenate([raw_family[:, :10], raw_family[:, -1:]], axis=1).astype(np.float32)
    family_attr = family_compact[:, :10]
    home_zone = family_compact[:, -1].astype(np.int64)

    member_raw = np.load(f"{DATA_DIR}/family_member_sample_improved_cluster_{split}.npy").astype(np.float32)
    member_attr = member_raw[:, :, :MEMBER_DIM]  # 前 50 维
    member_mask = (member_raw[:, :, MEMBER_MASK_COL] != 0)
    work_zone = member_raw[:, :, WORK_ZONE_COL].astype(np.int64)

    activity = np.load(f"{DATA_DIR}/family_activity_{split}.npy").astype(np.float32)
    activity_core = activity[:, :, :, :ACTIVITY_CORE_DIM]
    activity_origin = activity[:, :, :, 27].astype(np.int64)
    activity_dest = activity[:, :, :, 28].astype(np.int64)
    activity_mask = (activity_core.sum(axis=-1) != 0)

    return dict(
        family_attr=family_attr,
        home_zone=home_zone,
        member_attr=member_attr,
        member_mask=member_mask,
        work_zone=work_zone,
        activity_core=activity_core,
        activity_origin=activity_origin,
        activity_dest=activity_dest,
        activity_mask=activity_mask,
    )


# ----------------- 张量化与设备移动 -----------------
def to_torch(d, device="cpu"):
    out = {}
    for k, v in d.items():
        if v.dtype in (np.float32, np.float64):
            out[k] = torch.tensor(v, dtype=torch.float32, device=device)
        elif v.dtype == bool:
            out[k] = torch.tensor(v, dtype=torch.bool, device=device)
        else:
            out[k] = torch.tensor(v, dtype=torch.long, device=device)
    return out


# ----------------- 无条件 baseline: flat 拼接/拆解 -----------------
# flat 向量布局（连续目标 + 类别 logits 目标）：
#   family_attr (10) + member_attr (8*50=400) + activity_continuous (8*6*2=96)
#   + activity_purpose_logits (8*6*10=480) + activity_mode (8*6*11=528)
#   + activity_driver (8*6*2=96) + activity_joint (8*6*2=96)
#   + home_zone_logits (2006) + member_mask_logits (8 sigmoid)
#   + work_zone_logits (8 * 2008=16064) + dest_zone_logits (8*6*2006=96288)
# 这样直接放 GAN/VAE 输出层会非常大；为简化，**将所有 zone 都视为 z-score 标量**，最终采样后四舍五入到 [0, 2005]。
# member_mask 和 zone 的 z-score 化由 utility 函数完成。

# z-score zone：把 [0, 2005] 整数缩放到 [-1, 1]
def zone_to_scaled(zone: np.ndarray, max_z=NUM_ZONES - 1):
    z = zone.astype(np.float32).clip(0, max_z)
    return (z / max_z) * 2.0 - 1.0  # [-1, 1]


def scaled_to_zone(scaled: np.ndarray, max_z=NUM_ZONES - 1):
    z = (scaled + 1.0) / 2.0 * max_z
    return np.round(z).astype(np.int64).clip(0, max_z)


# 无条件 baseline 的 flat 维度
def flat_dim_unconditional():
    family = FAMILY_DIM                            # 10
    home_zone_scaled = 1                            # 1
    member_attr = MAX_MEMBERS * MEMBER_DIM          # 400
    member_mask_scaled = MAX_MEMBERS                # 8
    work_zone_scaled = MAX_MEMBERS                  # 8
    act_continuous = MAX_MEMBERS * MAX_ACTIVITIES * 2          # 96
    act_purpose = MAX_MEMBERS * MAX_ACTIVITIES * NUM_PURPOSE   # 480
    act_mode = MAX_MEMBERS * MAX_ACTIVITIES * NUM_MODE         # 528
    act_driver = MAX_MEMBERS * MAX_ACTIVITIES * NUM_DRIVER     # 96
    act_joint = MAX_MEMBERS * MAX_ACTIVITIES * NUM_JOINT       # 96
    act_dest_scaled = MAX_MEMBERS * MAX_ACTIVITIES             # 48
    act_mask_scaled = MAX_MEMBERS * MAX_ACTIVITIES             # 48
    return (family + home_zone_scaled + member_attr + member_mask_scaled + work_zone_scaled +
            act_continuous + act_purpose + act_mode + act_driver + act_joint +
            act_dest_scaled + act_mask_scaled)


def pack_unconditional(d) -> np.ndarray:
    """把 load_split 返回的 dict 打包成 [N, D] flat 向量（用于 GAN/WGAN/VAE 训练）"""
    N = d["family_attr"].shape[0]
    parts = [
        d["family_attr"].reshape(N, -1),
        zone_to_scaled(d["home_zone"]).reshape(N, 1),
        d["member_attr"].reshape(N, -1),
        d["member_mask"].astype(np.float32).reshape(N, -1) * 2.0 - 1.0,
        zone_to_scaled(d["work_zone"], max_z=NUM_ZONES + 1).reshape(N, -1),
        d["activity_core"][..., :2].reshape(N, -1),
        d["activity_core"][..., 2:12].reshape(N, -1),
        d["activity_core"][..., 12:23].reshape(N, -1),
        d["activity_core"][..., 23:25].reshape(N, -1),
        d["activity_core"][..., 25:27].reshape(N, -1),
        zone_to_scaled(d["activity_dest"]).reshape(N, -1),
        d["activity_mask"].astype(np.float32).reshape(N, -1) * 2.0 - 1.0,
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def unpack_unconditional(flat: np.ndarray) -> dict:
    """把 [N, D] flat 向量反向打包；类别字段保留为 logits/scaled 形式，由调用方 argmax 离散化"""
    N = flat.shape[0]
    cur = 0
    def take(d):
        nonlocal cur
        x = flat[:, cur:cur + d]
        cur += d
        return x

    family_attr = take(FAMILY_DIM)
    home_zone_s = take(1).reshape(N)
    member_attr = take(MAX_MEMBERS * MEMBER_DIM).reshape(N, MAX_MEMBERS, MEMBER_DIM)
    member_mask_s = take(MAX_MEMBERS).reshape(N, MAX_MEMBERS)
    work_zone_s = take(MAX_MEMBERS).reshape(N, MAX_MEMBERS)
    act_cont = take(MAX_MEMBERS * MAX_ACTIVITIES * 2).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, 2)
    act_purpose = take(MAX_MEMBERS * MAX_ACTIVITIES * NUM_PURPOSE).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, NUM_PURPOSE)
    act_mode = take(MAX_MEMBERS * MAX_ACTIVITIES * NUM_MODE).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, NUM_MODE)
    act_driver = take(MAX_MEMBERS * MAX_ACTIVITIES * NUM_DRIVER).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, NUM_DRIVER)
    act_joint = take(MAX_MEMBERS * MAX_ACTIVITIES * NUM_JOINT).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, NUM_JOINT)
    act_dest_s = take(MAX_MEMBERS * MAX_ACTIVITIES).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES)
    act_mask_s = take(MAX_MEMBERS * MAX_ACTIVITIES).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES)

    return dict(
        family_attr=family_attr,
        home_zone_scaled=home_zone_s,
        member_attr=member_attr,
        member_mask_scaled=member_mask_s,
        work_zone_scaled=work_zone_s,
        activity_continuous=act_cont,
        activity_purpose=act_purpose,
        activity_mode=act_mode,
        activity_driver=act_driver,
        activity_joint=act_joint,
        activity_dest_scaled=act_dest_s,
        activity_mask_scaled=act_mask_s,
    )


# ----------------- 条件 baseline: 条件向量 -----------------
def condition_dim():
    return FAMILY_DIM + 1 + MAX_MEMBERS * MEMBER_DIM + MAX_MEMBERS + MAX_MEMBERS  # 10 + 1 + 400 + 8 + 8 = 427


def pack_condition(d) -> np.ndarray:
    """条件向量：family + home_zone + member_attr + member_mask + work_zone"""
    N = d["family_attr"].shape[0]
    parts = [
        d["family_attr"].reshape(N, -1),
        zone_to_scaled(d["home_zone"]).reshape(N, 1),
        d["member_attr"].reshape(N, -1),
        d["member_mask"].astype(np.float32).reshape(N, -1) * 2.0 - 1.0,
        zone_to_scaled(d["work_zone"], max_z=NUM_ZONES + 1).reshape(N, -1),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def activity_target_dim():
    """条件 baseline 的活动输出维度（[N, D]）"""
    return (MAX_MEMBERS * MAX_ACTIVITIES * 2 +
            MAX_MEMBERS * MAX_ACTIVITIES * NUM_PURPOSE +
            MAX_MEMBERS * MAX_ACTIVITIES * NUM_MODE +
            MAX_MEMBERS * MAX_ACTIVITIES * NUM_DRIVER +
            MAX_MEMBERS * MAX_ACTIVITIES * NUM_JOINT +
            MAX_MEMBERS * MAX_ACTIVITIES +  # dest_scaled
            MAX_MEMBERS * MAX_ACTIVITIES)   # mask_scaled


def pack_activity_target(d) -> np.ndarray:
    N = d["activity_core"].shape[0]
    parts = [
        d["activity_core"][..., :2].reshape(N, -1),
        d["activity_core"][..., 2:12].reshape(N, -1),
        d["activity_core"][..., 12:23].reshape(N, -1),
        d["activity_core"][..., 23:25].reshape(N, -1),
        d["activity_core"][..., 25:27].reshape(N, -1),
        zone_to_scaled(d["activity_dest"]).reshape(N, -1),
        d["activity_mask"].astype(np.float32).reshape(N, -1) * 2.0 - 1.0,
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def unpack_activity_target(flat: np.ndarray) -> dict:
    N = flat.shape[0]
    cur = 0
    def take(d):
        nonlocal cur
        x = flat[:, cur:cur + d]
        cur += d
        return x
    act_cont = take(MAX_MEMBERS * MAX_ACTIVITIES * 2).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, 2)
    act_purpose = take(MAX_MEMBERS * MAX_ACTIVITIES * NUM_PURPOSE).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, NUM_PURPOSE)
    act_mode = take(MAX_MEMBERS * MAX_ACTIVITIES * NUM_MODE).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, NUM_MODE)
    act_driver = take(MAX_MEMBERS * MAX_ACTIVITIES * NUM_DRIVER).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, NUM_DRIVER)
    act_joint = take(MAX_MEMBERS * MAX_ACTIVITIES * NUM_JOINT).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES, NUM_JOINT)
    act_dest_s = take(MAX_MEMBERS * MAX_ACTIVITIES).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES)
    act_mask_s = take(MAX_MEMBERS * MAX_ACTIVITIES).reshape(N, MAX_MEMBERS, MAX_ACTIVITIES)
    return dict(
        activity_continuous=act_cont,
        activity_purpose=act_purpose,
        activity_mode=act_mode,
        activity_driver=act_driver,
        activity_joint=act_joint,
        activity_dest_scaled=act_dest_s,
        activity_mask_scaled=act_mask_s,
    )


# ----------------- 把生成的 dict 还原成 [N, 8, 6, 29] -----------------
def assemble_activity_29(generated: dict, home_zone: np.ndarray, member_mask: np.ndarray = None) -> np.ndarray:
    """把 generator 输出的 dict 拼接成 [N, 8, 6, 29] 张量（与 notebook results_activity 同结构）

    cols:
      0:2  连续 z-score (start, end)
      2:12 purpose one-hot
      12:23 mode one-hot
      23:25 driver one-hot
      25:27 joint one-hot
      27   起点 zone (整数；t=0 用 home_zone, t≥1 用 t-1 的 dest)
      28   终点 zone (整数)

    activity_mask_scaled 在 [-1, 1]：>=0 视为有效活动；无效活动整体置 0（is_real 标志为 False）
    member_mask（可选）：[N, 8] bool，无效成员的 6 个活动整体置 0
    """
    N = generated["activity_continuous"].shape[0]
    def to_onehot(logits, K):
        idx = logits.argmax(axis=-1)
        oh = np.zeros((*idx.shape, K), dtype=np.float32)
        np.put_along_axis(oh, idx[..., None], 1.0, axis=-1)
        return oh

    p_oh = to_onehot(generated["activity_purpose"], NUM_PURPOSE)
    m_oh = to_onehot(generated["activity_mode"], NUM_MODE)
    d_oh = to_onehot(generated["activity_driver"], NUM_DRIVER)
    j_oh = to_onehot(generated["activity_joint"], NUM_JOINT)
    dest = scaled_to_zone(generated["activity_dest_scaled"])
    cont = generated["activity_continuous"].astype(np.float32)

    origin = np.zeros_like(dest)
    origin[:, :, 0] = home_zone[:, None]
    origin[:, :, 1:] = dest[:, :, :-1]

    out = np.concatenate([
        cont,
        p_oh, m_oh, d_oh, j_oh,
        origin[..., None].astype(np.float32),
        dest[..., None].astype(np.float32),
    ], axis=-1)
    assert out.shape[-1] == 29, out.shape

    # 应用 activity_mask（生成器自己输出的）：mask=False 的活动置 0
    if "activity_mask_scaled" in generated:
        amask = generated["activity_mask_scaled"] >= 0  # [N, 8, 6]
        out = out * amask[..., None].astype(np.float32)

    # 应用 member_mask：无效成员整体置 0
    if member_mask is not None:
        mm = member_mask.astype(np.float32)[:, :, None, None]  # [N, 8, 1, 1]
        out = out * mm

    return out


def derive_is_real_mask(activity_29: np.ndarray) -> np.ndarray:
    """与 notebook cell 67 一致：sum(cols 0:27) != 0 标识有效活动"""
    return activity_29[..., :27].sum(axis=-1) != 0
