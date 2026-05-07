"""统一采样脚本：加载所有 baseline checkpoint，生成 N=测试集大小 个样本，保存到 baselines/samples/

无条件 baseline (GAN/WGAN/VAE)：从噪声生成完整 flat 向量；解包 → 组装 [N, 8, 6, 29]
条件 baseline (CGAN/CWGAN/CVAE)：用测试集的 family/member 做条件，生成活动 flat；解包 → 组装 [N, 8, 6, 29]
"""
import argparse
import os
import numpy as np
import torch

from common import (
    load_split, pack_condition, unpack_unconditional, unpack_activity_target,
    flat_dim_unconditional, condition_dim, activity_target_dim, assemble_activity_29,
    NUM_ZONES,
)
from networks import (
    Generator, VAEDecoder, ConditionalGenerator, ConditionalVAEDecoder,
)
from train_utils import device, set_seed, CKPT_ROOT

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)


def _resolve_ckpt(name):
    return os.path.join(CKPT_ROOT, name, "final.pt")


def _ensure_member_mask_for_unconditional(unpacked, n):
    """无条件 baseline 没有 condition；从 member_mask_scaled >=0 派生"""
    return (unpacked["member_mask_scaled"] >= 0)


def sample_gan(n, dev):
    ckpt = torch.load(_resolve_ckpt("gan"), map_location=dev)
    cfg = ckpt["config"]
    G = Generator(cfg["z_dim"], ckpt["out_dim"], hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(dev)
    G.load_state_dict(ckpt["G"]); G.eval()
    with torch.no_grad():
        z = torch.randn(n, cfg["z_dim"], device=dev)
        flat = G(z).cpu().numpy()
    return flat


def sample_wgan(n, dev):
    ckpt = torch.load(_resolve_ckpt("wgan"), map_location=dev)
    cfg = ckpt["config"]
    G = Generator(cfg["z_dim"], ckpt["out_dim"], hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(dev)
    G.load_state_dict(ckpt["G"]); G.eval()
    with torch.no_grad():
        z = torch.randn(n, cfg["z_dim"], device=dev)
        flat = G(z).cpu().numpy()
    return flat


def sample_vae(n, dev):
    ckpt = torch.load(_resolve_ckpt("vae"), map_location=dev)
    cfg = ckpt["config"]
    dec = VAEDecoder(cfg["z_dim"], ckpt["out_dim"], hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(dev)
    dec.load_state_dict(ckpt["decoder"]); dec.eval()
    with torch.no_grad():
        z = torch.randn(n, cfg["z_dim"], device=dev)
        flat = dec(z).cpu().numpy()
    return flat


def sample_cgan(cond, dev):
    ckpt = torch.load(_resolve_ckpt("cgan"), map_location=dev)
    cfg = ckpt["config"]
    G = ConditionalGenerator(ckpt["cond_dim"], cfg["z_dim"], ckpt["target_dim"],
                             hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(dev)
    G.load_state_dict(ckpt["G"]); G.eval()
    with torch.no_grad():
        c = torch.from_numpy(cond).to(dev)
        z = torch.randn(c.size(0), cfg["z_dim"], device=dev)
        flat = G(c, z).cpu().numpy()
    return flat


def sample_cwgan(cond, dev):
    ckpt = torch.load(_resolve_ckpt("cwgan"), map_location=dev)
    cfg = ckpt["config"]
    G = ConditionalGenerator(ckpt["cond_dim"], cfg["z_dim"], ckpt["target_dim"],
                             hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(dev)
    G.load_state_dict(ckpt["G"]); G.eval()
    with torch.no_grad():
        c = torch.from_numpy(cond).to(dev)
        z = torch.randn(c.size(0), cfg["z_dim"], device=dev)
        flat = G(c, z).cpu().numpy()
    return flat


def sample_cvae(cond, dev):
    ckpt = torch.load(_resolve_ckpt("cvae"), map_location=dev)
    cfg = ckpt["config"]
    dec = ConditionalVAEDecoder(cfg["z_dim"], ckpt["cond_dim"], ckpt["target_dim"],
                                hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(dev)
    dec.load_state_dict(ckpt["decoder"]); dec.eval()
    with torch.no_grad():
        c = torch.from_numpy(cond).to(dev)
        z = torch.randn(c.size(0), cfg["z_dim"], device=dev)
        flat = dec(z, c).cpu().numpy()
    return flat


def build_unconditional_arr29(flat):
    unpacked = unpack_unconditional(flat)
    n = flat.shape[0]
    member_mask = _ensure_member_mask_for_unconditional(unpacked, n)
    # 用 home_zone_scaled 反解
    from common import scaled_to_zone
    home_zone = scaled_to_zone(unpacked["home_zone_scaled"])  # [N]
    return assemble_activity_29(unpacked, home_zone, member_mask)


def build_conditional_arr29(flat, test_data):
    unpacked = unpack_activity_target(flat)
    return assemble_activity_29(unpacked, test_data["home_zone"], test_data["member_mask"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["gan", "wgan", "vae", "cgan", "cwgan", "cvae"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    dev = device()
    test_data = load_split("test")
    n_test = test_data["family_attr"].shape[0]
    cond = pack_condition(test_data)
    print(f"Test set size N = {n_test}, generating samples for: {args.models}")

    for name in args.models:
        ckpt_path = _resolve_ckpt(name)
        if not os.path.exists(ckpt_path):
            print(f"  [{name}] checkpoint not found at {ckpt_path}, skip")
            continue
        if name == "gan":
            flat = sample_gan(n_test, dev); arr29 = build_unconditional_arr29(flat)
        elif name == "wgan":
            flat = sample_wgan(n_test, dev); arr29 = build_unconditional_arr29(flat)
        elif name == "vae":
            flat = sample_vae(n_test, dev); arr29 = build_unconditional_arr29(flat)
        elif name == "cgan":
            flat = sample_cgan(cond, dev); arr29 = build_conditional_arr29(flat, test_data)
        elif name == "cwgan":
            flat = sample_cwgan(cond, dev); arr29 = build_conditional_arr29(flat, test_data)
        elif name == "cvae":
            flat = sample_cvae(cond, dev); arr29 = build_conditional_arr29(flat, test_data)
        else:
            print(f"  [{name}] unknown, skip"); continue

        out = os.path.join(SAMPLES_DIR, f"{name}.npy")
        np.save(out, arr29)
        from common import derive_is_real_mask
        ratio = derive_is_real_mask(arr29).mean()
        print(f"  [{name}] saved {arr29.shape} → {out} (is_real ratio = {ratio:.3f})")


if __name__ == "__main__":
    main()
