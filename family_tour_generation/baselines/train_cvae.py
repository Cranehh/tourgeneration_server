"""B6: 条件 VAE (CVAE)"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from common import load_split, pack_condition, pack_activity_target, condition_dim, activity_target_dim
from networks import ConditionalVAEEncoder, ConditionalVAEDecoder
from train_utils import set_seed, make_loader, save_ckpt, device


def kl_normal(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def train(args):
    set_seed()
    dev = device()
    print(f"[CVAE] device={dev}")

    d = load_split("train")
    cond = pack_condition(d)
    target = pack_activity_target(d)
    cd = condition_dim()
    td = activity_target_dim()
    loader = make_loader(cond, target, batch_size=args.batch_size)

    enc = ConditionalVAEEncoder(td, cd, args.z_dim, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    dec = ConditionalVAEDecoder(args.z_dim, cd, td, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    mse = nn.MSELoss()

    enc.train(); dec.train()
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"CVAE ep{ep}/{args.epochs}", ncols=100, leave=False)
        for c, t in pbar:
            c = c.to(dev); t = t.to(dev)
            opt.zero_grad()
            mu, logvar = enc(t, c)
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
            t_rec = dec(z, c)
            loss_rec = mse(t_rec, t)
            loss_kl = kl_normal(mu, logvar)
            loss = loss_rec + args.beta * loss_kl
            loss.backward()
            opt.step()
            pbar.set_postfix(rec=f"{loss_rec.item():.3f}", kl=f"{loss_kl.item():.3f}")
        print(f"[CVAE] epoch {ep} done | last_rec={loss_rec.item():.4f} kl={loss_kl.item():.4f}")

    save_ckpt("cvae", {
        "name": "cvae", "encoder": enc.state_dict(), "decoder": dec.state_dict(),
        "config": vars(args), "cond_dim": cd, "target_dim": td,
    })
    print("[CVAE] checkpoint saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--beta", type=float, default=0.01)
    args = p.parse_args()
    train(args)
