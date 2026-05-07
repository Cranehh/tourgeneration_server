"""加载主模型 ckpt，对测试集做 inference，转换 predictions 到 [N,8,6,29]，保存 samples/main.npy

用法：
  python run_main_inference.py
  python run_main_inference.py --output main_no_nash.npy --disable_nash  # （可选）推理开关消融
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 把上层目录加进 sys.path 以便 import family_tour_generation 模块
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from config import ModelConfig, TrainConfig
from model import create_model
from data import FamilyTourDataset, collate_fn

# 也把当前目录加进去，便于 import common
sys.path.insert(0, HERE)
from common import NUM_PURPOSE, NUM_MODE, NUM_DRIVER, NUM_JOINT


def load_test_batch_data(data_dir):
    """与 train_with_ss_rollout.py main() 的数据加载逻辑一致"""
    data = np.load(f"{data_dir}/family_sample_improved_cluster_test.npy")
    family_data = np.concatenate([data[:, :10], data[:, -1:]], axis=1)
    member_data = np.load(f"{data_dir}/family_member_sample_improved_cluster_test.npy")
    activity_data = np.load(f"{data_dir}/family_activity_test.npy")
    member_mask = (member_data[:, :, -2] != 0)
    activity_mask = activity_data[:, :, :, :27].sum(axis=-1) != 0
    family_pattern = np.load(f"{data_dir}/family_pattern_test.npy")
    individual_pattern = np.load(f"{data_dir}/person_pattern_test.npy")
    return FamilyTourDataset(
        family_data, member_data, activity_data,
        member_mask, activity_mask,
        family_pattern, individual_pattern,
    )


def predictions_to_arr29(preds: dict, home_zones: torch.Tensor, member_mask: torch.Tensor) -> np.ndarray:
    """把 predictions dict 转换成 [B,8,6,29] np.ndarray，与基线对齐

    preds keys:
      continuous: [B,M,T,2] z-score
      purpose:    [B,M,T,10] logits
      mode:       [B,M,T,11] logits
      driver:     [B,M,T,2]  logits
      joint:      [B,M,T,2]  logits
      destination:[B,M,T,2006] logits  (用 argmax 取 zone)
    """
    B, M, T = preds["continuous"].shape[:3]
    assert M == 8 and T == 6

    # 类别 → argmax → one-hot
    def to_onehot(logits, K):
        idx = logits.argmax(dim=-1)  # [B,M,T]
        oh = torch.zeros(*idx.shape, K, dtype=torch.float32, device=idx.device)
        oh.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return oh

    p_oh = to_onehot(preds["purpose"], NUM_PURPOSE)
    m_oh = to_onehot(preds["mode"], NUM_MODE)
    d_oh = to_onehot(preds["driver"], NUM_DRIVER)
    j_oh = to_onehot(preds["joint"], NUM_JOINT)
    cont = preds["continuous"].float()

    # 目的地
    if "destination" in preds:
        dest = preds["destination"].argmax(dim=-1)  # [B,M,T]
    else:
        dest = torch.zeros(B, M, T, dtype=torch.long, device=cont.device)

    # 起点：t=0 → home_zone, t≥1 → 上一步 dest
    origin = torch.zeros_like(dest)
    origin[:, :, 0] = home_zones.unsqueeze(-1)
    origin[:, :, 1:] = dest[:, :, :-1]

    arr = torch.cat([
        cont, p_oh, m_oh, d_oh, j_oh,
        origin.unsqueeze(-1).float(),
        dest.unsqueeze(-1).float(),
    ], dim=-1)
    assert arr.shape[-1] == 29

    # 应用 member_mask（无效成员整段置 0）
    mm = member_mask.float()[:, :, None, None]
    arr = arr * mm

    return arr.cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(os.path.dirname(PARENT), "checkpoints_ss_with_condition_nash_without_end5", "checkpoint_epoch_499.pt"))
    p.add_argument("--data_dir", default=os.path.join(os.path.dirname(PARENT), "数据"))
    p.add_argument("--output", default="main.npy")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--disable_nash", action="store_true", help="推理时关闭 Nash 层")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, ckpt: {args.ckpt}")

    # 配置：与 train_with_ss_rollout.py main() 一致
    model_config = ModelConfig(
        family_dim=10,
        member_dim=51,
        activity_dim=27,
        max_members=8,
        max_activities=6,
        d_model=256,
        d_emb=256,
        num_heads=16,
        num_decoder_layers=16,
        num_inducing_points=16,
        dropout=0.3,
        num_shared_experts=2,
        num_task_experts=2,
        num_cgc_layers=2,
    )

    model = create_model(model_config).to(device)

    # 外部 zone 特征
    if model_config.use_destination_prediction:
        lda_matrix = np.load(f"{args.data_dir}/position_poi_topic.npy")
        affordance_matrix = np.load(f"{args.data_dir}/position_activity_value.npy")
        impedance_matrix = np.load(f"{args.data_dir}/position_distance.npy")
        model.decoder.zone_module.load_external_features(lda_matrix, affordance_matrix, impedance_matrix)
        model.decoder.activity_zone_module.load_external_features(lda_matrix, affordance_matrix)
        model.encoder.zone_embedding.load_external_features(lda_matrix, affordance_matrix)

    # 加载 ckpt
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded ckpt | missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    if len(missing) > 0:
        print(f"  first 3 missing: {missing[:3]}")
    if len(unexpected) > 0:
        print(f"  first 3 unexpected: {unexpected[:3]}")

    if args.disable_nash:
        if hasattr(model.decoder, "nash_layer") and model.decoder.nash_layer is not None:
            model.decoder.nash_layer.set_enabled(False)
            print("Nash layer DISABLED for this run")
        else:
            print("WARN: model has no nash_layer to disable")

    model.eval()
    test_ds = load_test_batch_data(args.data_dir)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"Test set: {len(test_ds)} families, {len(loader)} batches")

    all_arr = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch = batch.to(device)
            preds, _ = model.generate(
                batch.family_attr, batch.member_attr, batch.member_mask,
                home_zones=batch.home_zones, work_positions=batch.work_positions,
                batch=batch, current_epoch=999,
            )
            arr29 = predictions_to_arr29(preds, batch.home_zones, batch.member_mask)
            all_arr.append(arr29)

    full = np.concatenate(all_arr, axis=0)
    out_path = os.path.join(HERE, "samples", args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, full)
    is_real = (full[..., :27].sum(axis=-1) != 0).mean()
    print(f"Saved {full.shape} → {out_path}  (is_real ratio = {is_real:.3f})")


if __name__ == "__main__":
    main()
