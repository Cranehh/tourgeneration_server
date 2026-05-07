"""B5: 条件 WGAN-GP (CWGAN-GP)"""
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from common import load_split, pack_condition, pack_activity_target, condition_dim, activity_target_dim
from networks import ConditionalGenerator, ConditionalCritic, gradient_penalty
from train_utils import set_seed, make_loader, save_ckpt, device


def train(args):
    set_seed()
    dev = device()
    print(f"[CWGAN] device={dev}")

    d = load_split("train")
    cond = pack_condition(d)
    target = pack_activity_target(d)
    cd = condition_dim()
    td = activity_target_dim()
    loader = make_loader(cond, target, batch_size=args.batch_size)

    G = ConditionalGenerator(cd, args.z_dim, td, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    C = ConditionalCritic(td, cd, hidden=args.hidden, n_layers=args.n_layers).to(dev)
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optC = optim.Adam(C.parameters(), lr=args.lr, betas=(0.5, 0.9))

    G.train(); C.train()
    iters = 0
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"CWGAN ep{ep}/{args.epochs}", ncols=100, leave=False)
        for c, t in pbar:
            c = c.to(dev); t = t.to(dev)
            bsz = c.size(0)

            optC.zero_grad()
            z = torch.randn(bsz, args.z_dim, device=dev)
            fake = G(c, z).detach()
            c_real = C(t, c)
            c_fake = C(fake, c)
            gp = gradient_penalty(C, t, fake, cond=c, lam=args.gp)
            loss_c = c_fake.mean() - c_real.mean() + gp
            loss_c.backward()
            optC.step()

            iters += 1
            if iters % args.n_critic == 0:
                optG.zero_grad()
                z = torch.randn(bsz, args.z_dim, device=dev)
                fake = G(c, z)
                c_fake = C(fake, c)
                loss_g = -c_fake.mean()
                loss_g.backward()
                optG.step()
                pbar.set_postfix(c=f"{loss_c.item():.3f}", g=f"{loss_g.item():.3f}")
        print(f"[CWGAN] epoch {ep} done | last_c={loss_c.item():.4f}")

    save_ckpt("cwgan", {
        "name": "cwgan", "G": G.state_dict(), "C": C.state_dict(),
        "config": vars(args), "cond_dim": cd, "target_dim": td,
    })
    print("[CWGAN] checkpoint saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--gp", type=float, default=10.0)
    args = p.parse_args()
    train(args)
