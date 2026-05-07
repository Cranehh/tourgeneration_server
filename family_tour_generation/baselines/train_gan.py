"""B1: 无条件 vanilla GAN
输入：噪声 z
输出：家庭+成员+活动联合 flat 向量
损失：BCE
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from common import load_split, pack_unconditional, flat_dim_unconditional
from networks import Generator, Discriminator
from train_utils import set_seed, make_loader, save_ckpt, device


def train(args):
    set_seed()
    dev = device()
    print(f"[GAN] device={dev}")

    d = load_split("train")
    flat = pack_unconditional(d)
    print(f"[GAN] train flat shape: {flat.shape}")
    out_dim = flat_dim_unconditional()
    assert flat.shape[1] == out_dim

    loader = make_loader(flat, batch_size=args.batch_size)
    G = Generator(args.z_dim, out_dim, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    D = Discriminator(out_dim, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    G.train(); D.train()
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"GAN ep{ep}/{args.epochs}", ncols=100, leave=False)
        for (real,) in pbar:
            real = real.to(dev)
            bsz = real.size(0)

            # --- D step ---
            optD.zero_grad()
            z = torch.randn(bsz, args.z_dim, device=dev)
            fake = G(z).detach()
            d_real = D(real).squeeze(-1)
            d_fake = D(fake).squeeze(-1)
            loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            loss_d.backward()
            optD.step()

            # --- G step ---
            optG.zero_grad()
            z = torch.randn(bsz, args.z_dim, device=dev)
            fake = G(z)
            d_fake = D(fake).squeeze(-1)
            loss_g = bce(d_fake, torch.ones_like(d_fake))
            loss_g.backward()
            optG.step()

            pbar.set_postfix(d=f"{loss_d.item():.3f}", g=f"{loss_g.item():.3f}")
        print(f"[GAN] epoch {ep} done | last_d={loss_d.item():.4f} last_g={loss_g.item():.4f}")

    save_ckpt("gan", {
        "name": "gan",
        "G": G.state_dict(),
        "D": D.state_dict(),
        "config": vars(args),
        "out_dim": out_dim,
    })
    print("[GAN] checkpoint saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z_dim", type=int, default=100)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    args = p.parse_args()
    train(args)
