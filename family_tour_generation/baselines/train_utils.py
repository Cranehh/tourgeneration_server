"""训练共享工具：dataset 包装、batch 迭代、checkpoint 保存"""
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

CKPT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints")
SEED = 42


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(*tensors, batch_size=256, shuffle=True, num_workers=0, pin_memory=True):
    ds = TensorDataset(*[torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in tensors])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def save_ckpt(name: str, ckpt: dict):
    out_dir = os.path.join(CKPT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final.pt")
    torch.save(ckpt, path)
    return path


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
