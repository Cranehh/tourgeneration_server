"""轻量 MLP 网络工厂（GAN/WGAN/VAE 系基线共用）

设计：
  - Generator / VAE Decoder：Linear → LeakyReLU → ... → Linear（输出维度灵活）
  - Discriminator / Critic：Linear → LeakyReLU → ... → Linear(1)
  - VAE Encoder：Linear → ReLU → ... → 输出 (mu, logvar)
  - 条件编码器：把家庭+成员条件向量映射到 d_cond 维
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, n_layers=3, activation="leaky_relu", dropout=0.0,
                 final_activation=None):
        super().__init__()
        act = {
            "leaky_relu": lambda: nn.LeakyReLU(0.2, inplace=True),
            "relu": lambda: nn.ReLU(inplace=True),
        }[activation]
        layers = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(prev, hidden))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        if final_activation == "tanh":
            layers.append(nn.Tanh())
        elif final_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    """无条件生成器：z → flat 输出"""
    def __init__(self, z_dim, out_dim, hidden=512, n_layers=3):
        super().__init__()
        self.net = MLP(z_dim, out_dim, hidden=hidden, n_layers=n_layers, activation="leaky_relu")

    def forward(self, z):
        return self.net(z)


class ConditionalGenerator(nn.Module):
    """条件生成器：(cond, z) → 活动 flat 输出"""
    def __init__(self, cond_dim, z_dim, out_dim, hidden=512, n_layers=3):
        super().__init__()
        self.net = MLP(cond_dim + z_dim, out_dim, hidden=hidden, n_layers=n_layers, activation="leaky_relu")

    def forward(self, cond, z):
        return self.net(torch.cat([cond, z], dim=-1))


class Discriminator(nn.Module):
    """二元判别器（用于 vanilla GAN，输出 sigmoid 概率）"""
    def __init__(self, in_dim, hidden=512, n_layers=3):
        super().__init__()
        self.net = MLP(in_dim, 1, hidden=hidden, n_layers=n_layers, activation="leaky_relu")

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class Critic(nn.Module):
    """Wasserstein critic（无 sigmoid，输出 scalar）"""
    def __init__(self, in_dim, hidden=512, n_layers=3):
        super().__init__()
        self.net = MLP(in_dim, 1, hidden=hidden, n_layers=n_layers, activation="leaky_relu")

    def forward(self, x):
        return self.net(x)


class VAEEncoder(nn.Module):
    """VAE 编码器：x → (mu, logvar)"""
    def __init__(self, in_dim, z_dim, hidden=512, n_layers=3):
        super().__init__()
        self.backbone = MLP(in_dim, hidden, hidden=hidden, n_layers=n_layers, activation="relu")
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, x):
        h = self.backbone(x)
        return self.mu(h), self.logvar(h)


class VAEDecoder(nn.Module):
    """VAE 解码器：z → flat 输出"""
    def __init__(self, z_dim, out_dim, hidden=512, n_layers=3):
        super().__init__()
        self.net = MLP(z_dim, out_dim, hidden=hidden, n_layers=n_layers, activation="relu")

    def forward(self, z):
        return self.net(z)


class ConditionalVAEEncoder(nn.Module):
    """CVAE 编码器：(target, cond) → (mu, logvar)"""
    def __init__(self, target_dim, cond_dim, z_dim, hidden=512, n_layers=3):
        super().__init__()
        self.backbone = MLP(target_dim + cond_dim, hidden, hidden=hidden, n_layers=n_layers, activation="relu")
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, target, cond):
        h = self.backbone(torch.cat([target, cond], dim=-1))
        return self.mu(h), self.logvar(h)


class ConditionalVAEDecoder(nn.Module):
    """CVAE 解码器：(z, cond) → flat 输出"""
    def __init__(self, z_dim, cond_dim, out_dim, hidden=512, n_layers=3):
        super().__init__()
        self.net = MLP(z_dim + cond_dim, out_dim, hidden=hidden, n_layers=n_layers, activation="relu")

    def forward(self, z, cond):
        return self.net(torch.cat([z, cond], dim=-1))


class ConditionalDiscriminator(nn.Module):
    """条件判别器（输入 target+cond，输出 sigmoid 概率）"""
    def __init__(self, target_dim, cond_dim, hidden=512, n_layers=3):
        super().__init__()
        self.net = MLP(target_dim + cond_dim, 1, hidden=hidden, n_layers=n_layers, activation="leaky_relu")

    def forward(self, target, cond):
        return torch.sigmoid(self.net(torch.cat([target, cond], dim=-1)))


class ConditionalCritic(nn.Module):
    """条件 Wasserstein critic（输入 target+cond，输出 scalar）"""
    def __init__(self, target_dim, cond_dim, hidden=512, n_layers=3):
        super().__init__()
        self.net = MLP(target_dim + cond_dim, 1, hidden=hidden, n_layers=n_layers, activation="leaky_relu")

    def forward(self, target, cond):
        return self.net(torch.cat([target, cond], dim=-1))


def gradient_penalty(critic, real, fake, cond=None, lam=10.0):
    """WGAN-GP 梯度惩罚"""
    bsz = real.size(0)
    alpha = torch.rand(bsz, 1, device=real.device)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)
    if cond is None:
        d_interp = critic(interp)
    else:
        d_interp = critic(interp, cond)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(bsz, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lam
    return gp
