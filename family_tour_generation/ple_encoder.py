"""
PLE (Progressive Layered Extraction) 编码器
包含：
- 共享专家1: 家庭信息提取
- 共享专家2: 成员集合信息提取 (Set Transformer)
- 任务特定专家: 个体信息提取
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class MultiheadAttentionBlock(nn.Module):
    """MAB: 多头注意力 + 残差 + LayerNorm"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        key_padding_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, q_len, d_model)
            key: (batch, k_len, d_model)
            value: (batch, k_len, d_model)
            key_padding_mask: (batch, k_len) True表示需要mask的位置
        
        Returns:
            (batch, q_len, d_model)
        """
        # 注意力
        attn_out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        x = self.norm1(query + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class InducedSetAttentionBlock(nn.Module):
    """
    ISAB: Induced Set Attention Block
    通过诱导点降低计算复杂度: O(n^2) -> O(nm)
    """
    
    def __init__(self, d_model: int, num_heads: int, num_inducing_points: int, dropout: float = 0.1):
        super().__init__()
        
        # 可学习的诱导点
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, d_model))
        nn.init.xavier_uniform_(self.inducing_points)
        
        # 两个MAB
        self.mab1 = MultiheadAttentionBlock(d_model, num_heads, dropout)
        self.mab2 = MultiheadAttentionBlock(d_model, num_heads, dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n, d_model) 输入集合
            mask: (batch, n) True表示有效位置
        
        Returns:
            (batch, n, d_model)
        """
        batch_size = x.size(0)
        
        # 扩展诱导点到batch维度
        inducing = self.inducing_points.expand(batch_size, -1, -1)  # (batch, m, d_model)
        
        # 诱导点从输入集合聚合信息
        # key_padding_mask: True表示忽略
        key_mask = ~mask if mask is not None else None
        h = self.mab1(inducing, x, x, key_padding_mask=key_mask)  # (batch, m, d_model)
        
        # 输入从诱导点获取全局信息
        out = self.mab2(x, h, h)  # (batch, n, d_model)
        
        return out


class MemberSetEncoder(nn.Module):
    """
    成员集合编码器 (共享专家2)
    使用Set Transformer处理可变数量的家庭成员
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(config.member_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        
        # 两层ISAB
        self.isab1 = InducedSetAttentionBlock(
            config.d_model, config.num_heads, config.num_inducing_points, config.dropout
        )
        self.isab2 = InducedSetAttentionBlock(
            config.d_model, config.num_heads, config.num_inducing_points, config.dropout
        )
    
    def forward(self, member_attr: torch.Tensor, member_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Args:
            member_attr: (batch, max_members, member_dim)
            member_mask: (batch, max_members) True表示有效成员
        
        Returns:
            (batch, max_members, d_model) 每个成员融合了集合信息的表示
        """
        # 投影到模型维度
        x = self.input_proj(member_attr)  # (batch, max_members, d_model)
        
        # 两层ISAB
        x = self.isab1(x, member_mask)
        x = self.isab2(x, member_mask)
        
        return x


class FamilyExpert(nn.Module):
    """家庭信息专家 (共享专家1)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(config.family_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )
    
    def forward(self, family_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            family_attr: (batch, family_dim)
        
        Returns:
            (batch, d_model)
        """
        return self.net(family_attr)


class IndividualExpert(nn.Module):
    """个体信息专家 (任务特定专家)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(config.member_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )
    
    def forward(self, member_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            member_attr: (batch, max_members, member_dim)
        
        Returns:
            (batch, max_members, d_model)
        """
        return self.net(member_attr)


class PLEEncoder(nn.Module):
    """
    PLE编码器
    
    三个专家:
    - 共享专家1: 家庭信息提取
    - 共享专家2: 成员集合信息提取
    - 任务特定专家: 个体信息提取
    
    门控网络根据个体特征动态融合三个专家的输出
    """
    
    def __init__(self, 
                 config: ModelConfig,
                 num_family_patterns: int = 118,
                 num_individual_patterns: int = 207):
        super().__init__()
        self.config = config
        
        # 三个专家
        self.family_expert = FamilyExpert(config)
        self.member_set_expert = MemberSetEncoder(config)
        self.individual_expert = IndividualExpert(config)
        
        # 门控网络: 根据个体特征计算三个专家的融合权重
        self.gate = nn.Sequential(
            nn.Linear(config.member_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 3)  # 三个专家的权重
        )
    
    def forward(
        self, 
        family_attr: torch.Tensor, 
        member_attr: torch.Tensor, 
        member_mask: torch.BoolTensor
    ) -> tuple:
        """
        Args:
            family_attr: (batch, family_dim) 家庭属性
            member_attr: (batch, max_members, member_dim) 成员属性
            member_mask: (batch, max_members) 有效成员标记
        
        Returns:
            member_repr: (batch, max_members, d_model) 每个成员的融合表示
            family_repr: (batch, d_model) 家庭级表示
        """
        batch_size, max_members, _ = member_attr.shape
        
        # 共享专家1: 家庭级表示
        family_repr = self.family_expert(family_attr)  # (batch, d_model)
        
        # 共享专家2: 成员集合表示
        member_set_repr = self.member_set_expert(member_attr, member_mask)  # (batch, max_members, d_model)
        
        # 任务特定专家: 个体表示
        individual_repr = self.individual_expert(member_attr)  # (batch, max_members, d_model)
        
        # 门控权重
        gate_weights = F.softmax(self.gate(member_attr), dim=-1)  # (batch, max_members, 3)
        
        # 扩展家庭表示到每个成员
        family_repr_expanded = family_repr.unsqueeze(1).expand(-1, max_members, -1)  # (batch, max_members, d_model)
        
        # 堆叠三个专家的输出
        expert_outputs = torch.stack([
            family_repr_expanded,   # 共享专家1
            member_set_repr,        # 共享专家2
            individual_repr         # 任务特定专家
        ], dim=2)  # (batch, max_members, 3, d_model)
        
        # 加权融合
        member_repr = torch.einsum('bmk,bmkd->bmd', gate_weights, expert_outputs)  # (batch, max_members, d_model)
        
        # mask无效成员
        member_repr = member_repr * member_mask.unsqueeze(-1).float()
        
        return member_repr, family_repr
