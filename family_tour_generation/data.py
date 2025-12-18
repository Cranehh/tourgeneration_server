"""
数据结构和Dataset定义
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np


@dataclass
class FamilyTourBatch:
    """一个批次的家庭活动链数据"""
    
    # ========== 输入特征 ==========
    family_attr: torch.Tensor           # (B, Ff) 家庭属性
    member_attr: torch.Tensor           # (B, max_members, Fm) 成员属性
    member_mask: torch.BoolTensor       # (B, max_members) 有效成员标记
    
    # ========== 目标值 ==========
    activities: torch.Tensor            # (B, max_members, max_activities, Fa) 活动链
    activity_mask: torch.BoolTensor     # (B, max_members, max_activities) 有效活动标记

    family_pattern: torch.Tensor        # (B, pattern_dim) 家庭活动模式
    member_pattern: torch.Tensor        # (B, max_members, pattern_dim) 成员
    
    def to(self, device):
        """移动到指定设备"""
        return FamilyTourBatch(
            family_attr=self.family_attr.to(device),
            member_attr=self.member_attr.to(device),
            member_mask=self.member_mask.to(device),
            activities=self.activities.to(device),
            activity_mask=self.activity_mask.to(device),
            family_pattern = self.family_pattern.to(device),
            member_pattern = self.member_pattern.to(device)
        )
    
    @property
    def batch_size(self):
        return self.family_attr.size(0)
    
    @property
    def device(self):
        return self.family_attr.device


class FamilyTourDataset(Dataset):
    """家庭活动链数据集"""
    
    def __init__(
        self,
        family_data: np.ndarray,      # (N, Ff)
        member_data: np.ndarray,      # (N, max_members, Fm)
        activity_data: np.ndarray,    # (N, max_members, max_activities, Fa)
        member_mask: np.ndarray,      # (N, max_members)
        activity_mask: np.ndarray,     # (N, max_members, max_activities)
        family_pattern: np.ndarray,     # (N, pattern_dim)
        member_pattern: np.ndarray      # (N, max_members, pattern_dim)
    ):
        self.family_data = torch.FloatTensor(family_data)
        self.member_data = torch.FloatTensor(member_data)
        self.activity_data = torch.FloatTensor(activity_data)
        self.member_mask = torch.BoolTensor(member_mask)
        self.activity_mask = torch.BoolTensor(activity_mask)
        self.family_pattern = torch.FloatTensor(family_pattern)
        self.member_pattern = torch.FloatTensor(member_pattern)
        
    def __len__(self):
        return len(self.family_data)
    
    def __getitem__(self, idx):
        return {
            'family_attr': self.family_data[idx],
            'member_attr': self.member_data[idx],
            'activities': self.activity_data[idx],
            'member_mask': self.member_mask[idx],
            'activity_mask': self.activity_mask[idx],
            'family_pattern': self.family_pattern[idx],
            'member_pattern': self.member_pattern[idx]
        }


def collate_fn(batch: List[Dict]) -> FamilyTourBatch:
    """将样本列表整理为批次"""
    return FamilyTourBatch(
        family_attr=torch.stack([b['family_attr'] for b in batch]),
        member_attr=torch.stack([b['member_attr'] for b in batch]),
        member_mask=torch.stack([b['member_mask'] for b in batch]),
        activities=torch.stack([b['activities'] for b in batch]),
        activity_mask=torch.stack([b['activity_mask'] for b in batch]),
        family_pattern=torch.stack([b['family_pattern'] for b in batch]),
        member_pattern=torch.stack([b['member_pattern'] for b in batch])
    )


def create_dataloader(
    family_data: np.ndarray,
    member_data: np.ndarray,
    activity_data: np.ndarray,
    member_mask: np.ndarray,
    activity_mask: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """创建DataLoader"""
    dataset = FamilyTourDataset(
        family_data, member_data, activity_data, member_mask, activity_mask
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
