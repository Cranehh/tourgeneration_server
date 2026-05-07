"""B2: 无条件 WGAN-GP
critic:G 步比 = 5:1, gradient penalty λ=10
"""
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from common import load_split, pack_unconditional, flat_dim_unconditional
from networks import Generator, Critic, gradient_penalty
from train_utils import set_seed, make_loader, save_ckpt, device


def train(args):
    set_seed()
    dev = device()
    print(f"[WGAN] device={dev}")

    d = load_split("train")
    flat = pack_unconditional(d)
    out_dim = flat_dim_unconditional()
    loader = make_loader(flat, batch_size=args.batch_size)

    G = Generator(args.z_dim, out_dim, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    C = Critic(out_dim, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optC = optim.Adam(C.parameters(), lr=args.lr, betas=(0.5, 0.9))

    G.train(); C.train()
    iters = 0
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"WGAN ep{ep}/{args.epochs}", ncols=100, leave=False)
        for (real,) in pbar:
            real = real.to(dev)
            bsz = real.size(0)

            # --- critic step ---
            optC.zero_grad()
            z = torch.randn(bsz, args.z_dim, device=dev)
            fake = G(z).detach()
            c_real = C(real)
            c_fake = C(fake)
            gp = gradient_penalty(C, real, fake, lam=args.gp)
            loss_c = c_fake.mean() - c_real.mean() + gp
            loss_c.backward()
            optC.step()

            iters += 1
            # --- G step every n_critic iterations ---
            if iters % args.n_critic == 0:
                optG.zero_grad()
                z = torch.randn(bsz, args.z_dim, device=dev)
                fake = G(z)
                c_fake = C(fake)
                loss_g = -c_fake.mean()
                loss_g.backward()
                optG.step()
                pbar.set_postfix(c=f"{loss_c.item():.3f}", g=f"{loss_g.item():.3f}")
        print(f"[WGAN] epoch {ep} done | last_c={loss_c.item():.4f}")

    save_ckpt("wgan", {
        "name": "wgan", "G": G.state_dict(), "C": C.state_dict(),
        "config": vars(args), "out_dim": out_dim,
    })
    print("[WGAN] checkpoint saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z_dim", type=int, default=100)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--gp", type=float, default=10.0)
    args = p.parse_args()
    train(args)
