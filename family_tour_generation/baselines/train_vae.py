"""B3: 无条件 VAE
重构损失：MSE on flat（包括 logits 部分；标签部分是 one-hot[-1,1] tanh-friendly，用 MSE 即可）
KL 损失：标准 KL between q(z|x) 和 N(0,I)
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from common import load_split, pack_unconditional, flat_dim_unconditional
from networks import VAEEncoder, VAEDecoder
from train_utils import set_seed, make_loader, save_ckpt, device


def kl_normal(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def train(args):
    set_seed()
    dev = device()
    print(f"[VAE] device={dev}")

    d = load_split("train")
    flat = pack_unconditional(d)
    out_dim = flat_dim_unconditional()
    loader = make_loader(flat, batch_size=args.batch_size)

    enc = VAEEncoder(out_dim, args.z_dim, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    dec = VAEDecoder(args.z_dim, out_dim, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    mse = nn.MSELoss()

    enc.train(); dec.train()
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"VAE ep{ep}/{args.epochs}", ncols=100, leave=False)
        for (x,) in pbar:
            x = x.to(dev)
            opt.zero_grad()
            mu, logvar = enc(x)
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
            x_rec = dec(z)
            loss_rec = mse(x_rec, x)
            loss_kl = kl_normal(mu, logvar)
            loss = loss_rec + args.beta * loss_kl
            loss.backward()
            opt.step()
            pbar.set_postfix(rec=f"{loss_rec.item():.3f}", kl=f"{loss_kl.item():.3f}")
        print(f"[VAE] epoch {ep} done | last_rec={loss_rec.item():.4f} kl={loss_kl.item():.4f}")

    save_ckpt("vae", {
        "name": "vae", "encoder": enc.state_dict(), "decoder": dec.state_dict(),
        "config": vars(args), "out_dim": out_dim,
    })
    print("[VAE] checkpoint saved.")


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
