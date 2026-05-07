"""B4: 条件 GAN (CGAN)
条件：家庭+成员属性
目标：活动 flat
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from common import load_split, pack_condition, pack_activity_target, condition_dim, activity_target_dim
from networks import ConditionalGenerator, ConditionalDiscriminator
from train_utils import set_seed, make_loader, save_ckpt, device


def train(args):
    set_seed()
    dev = device()
    print(f"[CGAN] device={dev}")

    d = load_split("train")
    cond = pack_condition(d)
    target = pack_activity_target(d)
    cond_dim = condition_dim()
    target_dim = activity_target_dim()
    print(f"[CGAN] cond_dim={cond_dim}, target_dim={target_dim}")

    loader = make_loader(cond, target, batch_size=args.batch_size)
    G = ConditionalGenerator(cond_dim, args.z_dim, target_dim, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    D = ConditionalDiscriminator(target_dim, cond_dim, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    G.train(); D.train()
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"CGAN ep{ep}/{args.epochs}", ncols=100, leave=False)
        for c, t in pbar:
            c = c.to(dev); t = t.to(dev)
            bsz = c.size(0)

            optD.zero_grad()
            z = torch.randn(bsz, args.z_dim, device=dev)
            fake = G(c, z).detach()
            d_real = D(t, c).squeeze(-1)
            d_fake = D(fake, c).squeeze(-1)
            loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            loss_d.backward()
            optD.step()

            optG.zero_grad()
            z = torch.randn(bsz, args.z_dim, device=dev)
            fake = G(c, z)
            d_fake = D(fake, c).squeeze(-1)
            loss_g = bce(d_fake, torch.ones_like(d_fake))
            loss_g.backward()
            optG.step()

            pbar.set_postfix(d=f"{loss_d.item():.3f}", g=f"{loss_g.item():.3f}")
        print(f"[CGAN] epoch {ep} done | last_d={loss_d.item():.4f} last_g={loss_g.item():.4f}")

    save_ckpt("cgan", {
        "name": "cgan", "G": G.state_dict(), "D": D.state_dict(),
        "config": vars(args), "cond_dim": cond_dim, "target_dim": target_dim,
    })
    print("[CGAN] checkpoint saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    args = p.parse_args()
    train(args)
