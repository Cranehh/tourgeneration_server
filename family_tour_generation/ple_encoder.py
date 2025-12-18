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

class FamilyPatternExpert(nn.Module):
    """
    家庭活动模式预测专家 (新增专家1)
    
    输入: 家庭属性 (B, family_dim)
    输出: 家庭活动模式概率分布 (B, num_family_patterns)
    """
    
    def __init__(
        self,
        family_dim: int,
        num_family_patterns: int = 118,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_family_patterns = num_family_patterns
        
        self.net = nn.Sequential(
            nn.Linear(family_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_family_patterns)
        )
    
    def forward(self, family_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            family_attr: (B, family_dim) 家庭属性
        
        Returns:
            family_pattern_prob: (B, num_family_patterns) 家庭活动模式概率分布
        """
        logits = self.net(family_attr)
        return F.softmax(logits, dim=-1)
    
    def forward_with_logits(self, family_attr: torch.Tensor):
        """
        返回 logits 和概率，便于计算损失
        
        Returns:
            logits: (B, num_family_patterns)
            probs: (B, num_family_patterns)
        """
        logits = self.net(family_attr)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class IndividualPatternExpert(nn.Module):
    """
    个人活动模式预测专家 (新增专家2)
    
    输入: 个人属性 (B, max_members, member_dim) + 家庭模式概率 (B, num_family_patterns)
    输出: 个人活动模式概率分布 (B, max_members, num_individual_patterns)
    """
    
    def __init__(
        self,
        member_dim: int,
        num_family_patterns: int = 118,
        num_individual_patterns: int = 207,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_family_patterns = num_family_patterns
        self.num_individual_patterns = num_individual_patterns
        
        # 家庭模式嵌入投影
        self.family_pattern_proj = nn.Linear(num_family_patterns, hidden_dim)
        
        # 个人属性投影
        self.member_proj = nn.Linear(member_dim, hidden_dim)
        
        # 融合后预测个人模式
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_individual_patterns)
        )
    
    def forward(
        self,
        member_attr: torch.Tensor,
        family_pattern_prob: torch.Tensor,
        member_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Args:
            member_attr: (B, max_members, member_dim) 成员属性
            family_pattern_prob: (B, num_family_patterns) 家庭活动模式概率
            member_mask: (B, max_members) 有效成员掩码
        
        Returns:
            individual_pattern_prob: (B, max_members, num_individual_patterns) 个人活动模式概率分布
        """
        batch_size, max_members, _ = member_attr.shape
        
        # 投影家庭模式
        family_feat = self.family_pattern_proj(family_pattern_prob)  # (B, hidden_dim)
        family_feat = family_feat.unsqueeze(1).expand(-1, max_members, -1)  # (B, max_members, hidden_dim)
        
        # 投影个人属性
        member_feat = self.member_proj(member_attr)  # (B, max_members, hidden_dim)
        
        # 融合
        fused = torch.cat([member_feat, family_feat], dim=-1)  # (B, max_members, hidden_dim*2)
        
        # 预测个人模式
        logits = self.fusion_net(fused)  # (B, max_members, num_individual_patterns)
        probs = F.softmax(logits, dim=-1)
        
        # 掩码无效成员
        if member_mask is not None:
            probs = probs * member_mask.unsqueeze(-1).float()
        
        return probs
    
    def forward_with_logits(
        self,
        member_attr: torch.Tensor,
        family_pattern_prob: torch.Tensor,
        member_mask: torch.BoolTensor = None
    ):
        """
        返回 logits 和概率，便于计算损失
        
        Returns:
            logits: (B, max_members, num_individual_patterns)
            probs: (B, max_members, num_individual_patterns)
        """
        batch_size, max_members, _ = member_attr.shape
        
        family_feat = self.family_pattern_proj(family_pattern_prob)
        family_feat = family_feat.unsqueeze(1).expand(-1, max_members, -1)
        
        member_feat = self.member_proj(member_attr)
        fused = torch.cat([member_feat, family_feat], dim=-1)
        
        logits = self.fusion_net(fused)
        probs = F.softmax(logits, dim=-1)
        
        if member_mask is not None:
            probs = probs * member_mask.unsqueeze(-1).float()
        
        return logits, probs

class PLEEncoder(nn.Module):
    """
    PLE编码器
    
    五个专家:
    - 共享专家1: 家庭信息提取
    - 共享专家2: 成员集合信息提取
    - 任务特定专家: 个体信息提取
    - 模式专家1: 家庭活动模式预测 (新增)
    - 模式专家2: 个人活动模式预测 (新增)
    
    门控网络根据个体特征和模式信息动态融合专家输出
    """
    
    def __init__(
        self, 
        config: ModelConfig,
        num_family_patterns: int = 118,
        num_individual_patterns: int = 207,
        use_pattern_experts: bool = True
    ):
        super().__init__()
        self.config = config
        self.use_pattern_experts = use_pattern_experts
        self.num_family_patterns = num_family_patterns
        self.num_individual_patterns = num_individual_patterns
        
        # 原有三个专家
        self.family_expert = FamilyExpert(config)
        self.member_set_expert = MemberSetEncoder(config)
        self.individual_expert = IndividualExpert(config)
        self.use_pattern_experts = getattr(config, 'use_pattern_condition', True)
        # 新增: 模式预测专家
        if self.use_pattern_experts:
            self.family_pattern_expert = FamilyPatternExpert(
                family_dim=config.family_dim,
                num_family_patterns=num_family_patterns,
                hidden_dim=config.d_model,
                dropout=config.dropout
            )
            self.individual_pattern_expert = IndividualPatternExpert(
                member_dim=config.member_dim,
                num_family_patterns=num_family_patterns,
                num_individual_patterns=num_individual_patterns,
                hidden_dim=config.d_model,
                dropout=config.dropout
            )
            
            # 模式嵌入投影 (将模式概率映射到 d_model 维度)
            self.family_pattern_proj = nn.Linear(num_family_patterns, config.d_model)
            self.individual_pattern_proj = nn.Linear(num_individual_patterns, config.d_model)
            
            # 门控网络: 5个专家
            self.gate = nn.Sequential(
                nn.Linear(config.member_dim + num_individual_patterns, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, 5)
            )
        else:
            # 不使用模式专家时，保持原有3个专家
            self.gate = nn.Sequential(
                nn.Linear(config.member_dim, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, 3)
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
        
        if self.use_pattern_experts:
            # 模式专家1: 预测家庭活动模式
            family_pattern_prob = self.family_pattern_expert(family_attr)  # (batch, num_family_patterns)
            
            # 模式专家2: 预测个人活动模式
            individual_pattern_prob = self.individual_pattern_expert(
                member_attr, family_pattern_prob, member_mask
            )  # (batch, max_members, num_individual_patterns)
            
            # 模式嵌入
            family_pattern_emb = self.family_pattern_proj(family_pattern_prob)  # (batch, d_model)
            individual_pattern_emb = self.individual_pattern_proj(individual_pattern_prob)  # (batch, max_members, d_model)
            
            # 扩展家庭级表示到每个成员
            family_repr_expanded = family_repr.unsqueeze(1).expand(-1, max_members, -1)
            family_pattern_emb_expanded = family_pattern_emb.unsqueeze(1).expand(-1, max_members, -1)
            
            # 门控输入: 成员属性 + 个人模式概率
            gate_input = torch.cat([member_attr, individual_pattern_prob], dim=-1)
            gate_weights = F.softmax(self.gate(gate_input), dim=-1)  # (batch, max_members, 5)
            
            # 堆叠五个专家的输出
            expert_outputs = torch.stack([
                family_repr_expanded,        # 共享专家1: 家庭信息
                member_set_repr,             # 共享专家2: 成员集合信息
                individual_repr,             # 任务特定专家: 个体信息
                family_pattern_emb_expanded, # 模式专家1: 家庭模式嵌入
                individual_pattern_emb       # 模式专家2: 个人模式嵌入
            ], dim=2)  # (batch, max_members, 5, d_model)
            
            # 保存模式输出用于计算损失
            pattern_outputs = {
                'family_pattern_prob': family_pattern_prob,
                'individual_pattern_prob': individual_pattern_prob
            }
        else:
            # 不使用模式专家
            family_repr_expanded = family_repr.unsqueeze(1).expand(-1, max_members, -1)
            gate_weights = F.softmax(self.gate(member_attr), dim=-1)  # (batch, max_members, 3)
            
            expert_outputs = torch.stack([
                family_repr_expanded,
                member_set_repr,
                individual_repr
            ], dim=2)  # (batch, max_members, 3, d_model)
            
            pattern_outputs = None
        
        # 加权融合
        member_repr = torch.einsum('bmk,bmkd->bmd', gate_weights, expert_outputs)
        
        # mask无效成员
        member_repr = member_repr * member_mask.unsqueeze(-1).float()
        
        return member_repr, family_repr, pattern_outputs
    
    def forward_with_pattern_logits(
        self, 
        family_attr: torch.Tensor, 
        member_attr: torch.Tensor, 
        member_mask: torch.BoolTensor
    ) -> tuple:
        """
        返回带 logits 的前向传播，便于计算模式预测损失
        
        Returns:
            member_repr: (batch, max_members, d_model)
            family_repr: (batch, d_model)
            pattern_outputs: dict, 包含 logits 和 probs
        """
        if not self.use_pattern_experts:
            member_repr, family_repr, _ = self.forward(family_attr, member_attr, member_mask)
            return member_repr, family_repr, None
        
        batch_size, max_members, _ = member_attr.shape
        
        # 共享专家
        family_repr = self.family_expert(family_attr)
        member_set_repr = self.member_set_expert(member_attr, member_mask)
        individual_repr = self.individual_expert(member_attr)
        
        # 模式专家 (带 logits)
        family_pattern_logits, family_pattern_prob = self.family_pattern_expert.forward_with_logits(family_attr)
        individual_pattern_logits, individual_pattern_prob = self.individual_pattern_expert.forward_with_logits(
            member_attr, family_pattern_prob, member_mask
        )
        
        # 模式嵌入
        family_pattern_emb = self.family_pattern_proj(family_pattern_prob)
        individual_pattern_emb = self.individual_pattern_proj(individual_pattern_prob)
        
        # 扩展
        family_repr_expanded = family_repr.unsqueeze(1).expand(-1, max_members, -1)
        family_pattern_emb_expanded = family_pattern_emb.unsqueeze(1).expand(-1, max_members, -1)
        
        # 门控
        gate_input = torch.cat([member_attr, individual_pattern_prob], dim=-1)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)
        
        # 堆叠专家输出
        expert_outputs = torch.stack([
            family_repr_expanded,
            member_set_repr,
            individual_repr,
            family_pattern_emb_expanded,
            individual_pattern_emb
        ], dim=2)
        
        # 加权融合
        member_repr = torch.einsum('bmk,bmkd->bmd', gate_weights, expert_outputs)
        member_repr = member_repr * member_mask.unsqueeze(-1).float()
        
        pattern_outputs = {
            'family_pattern_logits': family_pattern_logits,
            'family_pattern_prob': family_pattern_prob,
            'individual_pattern_logits': individual_pattern_logits,
            'individual_pattern_prob': individual_pattern_prob
        }
        
        return member_repr, family_repr, pattern_outputs
