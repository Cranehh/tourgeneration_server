"""
PLE (Progressive Layered Extraction) 编码器 - 重构版本

设计思路：
1. 统一专家结构：每个专家都处理 家庭信息 + 个人信息
2. 四个任务：
   - family_repr: 家庭信息表示
   - member_repr: 个人信息表示
   - family_pattern_prob: 家庭模式概率
   - individual_pattern_prob: 个人模式概率
3. 多层CGC结构：
   - 共享专家 + 任务特定专家（结构相同，都处理家庭+个人信息）
   - 共享门控（除最后一层外）：输出共享特征供下一层感知
   - 任务特定门控：每个任务从共享专家+任务专家中选择

参考: Ma et al. "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from config import ModelConfig


# ==================== 基础模块 ====================

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
        attn_out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        x = self.norm1(query + attn_out)
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
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, d_model))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab1 = MultiheadAttentionBlock(d_model, num_heads, dropout)
        self.mab2 = MultiheadAttentionBlock(d_model, num_heads, dropout)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        batch_size = x.size(0)
        inducing = self.inducing_points.expand(batch_size, -1, -1)
        key_mask = ~mask if mask is not None else None
        h = self.mab1(inducing, x, x, key_padding_mask=key_mask)
        out = self.mab2(x, h, h)
        return out


# ==================== 统一专家结构 ====================

class UnifiedExpert(nn.Module):
    """
    统一的专家结构

    每个专家同时处理家庭信息和个人信息：
    - 家庭分支: family_input -> family_output (batch, d_model)
    - 个人分支: member_input -> member_output (batch, max_members, d_model)
    - 融合: 家庭信息广播到每个成员，与个人信息融合

    输出：
    - family_output: (batch, d_model) 家庭级表示
    - member_output: (batch, max_members, d_model) 成员级表示（融合了家庭信息）
    """

    def __init__(
        self,
        family_input_dim: int,
        member_input_dim: int,
        output_dim: int,
        config: ModelConfig,
        use_set_attention: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.output_dim = output_dim
        self.use_set_attention = use_set_attention

        # ========== 家庭分支 ==========
        self.family_encoder = nn.Sequential(
            nn.Linear(family_input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # ========== 个人分支 ==========
        if use_set_attention:
            # 使用Set Transformer处理成员集合
            self.member_input_proj = nn.Sequential(
                nn.Linear(member_input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            self.member_isab1 = InducedSetAttentionBlock(
                output_dim, config.num_heads, config.num_inducing_points, dropout
            )
            self.member_isab2 = InducedSetAttentionBlock(
                output_dim, config.num_heads, config.num_inducing_points, dropout
            )
        else:
            # 简单MLP
            self.member_encoder = nn.Sequential(
                nn.Linear(member_input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        # ========== 融合层 ==========
        # 将家庭信息融入成员表示
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 从成员表示聚合回家庭表示
        self.family_aggregator = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(
        self,
        family_input: torch.Tensor,
        member_input: torch.Tensor,
        member_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            family_input: (batch, family_input_dim) 家庭特征
            member_input: (batch, max_members, member_input_dim) 成员特征
            member_mask: (batch, max_members) 有效成员掩码

        Returns:
            family_output: (batch, output_dim) 家庭级输出
            member_output: (batch, max_members, output_dim) 成员级输出
        """
        batch_size, max_members = member_input.shape[:2]

        # 家庭分支
        family_feat = self.family_encoder(family_input)  # (batch, output_dim)

        # 个人分支
        if self.use_set_attention:
            member_feat = self.member_input_proj(member_input)
            member_feat = self.member_isab1(member_feat, member_mask)
            member_feat = self.member_isab2(member_feat, member_mask)
        else:
            member_feat = self.member_encoder(member_input)
        # member_feat: (batch, max_members, output_dim)

        # 融合：将家庭信息广播到每个成员
        family_expanded = family_feat.unsqueeze(1).expand(-1, max_members, -1)
        fused = torch.cat([member_feat, family_expanded], dim=-1)
        member_output = self.fusion(fused)  # (batch, max_members, output_dim)

        # 聚合：从成员表示得到增强的家庭表示
        mask_expanded = member_mask.unsqueeze(-1).float()
        pooled = (member_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        family_output = family_feat + self.family_aggregator(pooled)  # 残差连接

        # 应用成员掩码
        member_output = member_output * mask_expanded

        return family_output, member_output


# ==================== 门控网络 ====================

class GatingNetwork(nn.Module):
    """
    门控网络
    根据输入特征动态选择专家权重
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, ..., input_dim) 门控输入

        Returns:
            weights: (batch, ..., num_experts) softmax归一化的专家权重
        """
        logits = self.gate(x)
        return F.softmax(logits, dim=-1)


# ==================== CGC层 ====================

class CGCLayer(nn.Module):
    """
    Customized Gate Control Layer

    结构：
    1. 共享专家：处理家庭+个人信息，输出被所有任务共享
    2. 任务特定专家：每个任务有自己的专家，也处理家庭+个人信息
    3. 共享门控（非最后一层）：融合共享专家输出，传递给下一层
    4. 任务特定门控：每个任务从共享专家+任务专家中选择

    四个任务：
    1. family_repr: 家庭表示
    2. member_repr: 成员表示
    3. family_pattern: 家庭模式
    4. individual_pattern: 个人模式
    """

    def __init__(
        self,
        config: ModelConfig,
        num_shared_experts: int,
        num_task_experts: int,
        family_input_dim: int,
        member_input_dim: int,
        is_first_layer: bool = False,
        is_last_layer: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.config = config
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.num_tasks = 4
        self.d_model = config.d_model
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        # ========== 共享专家 ==========
        self.shared_experts = nn.ModuleList()
        for i in range(num_shared_experts):
            expert = UnifiedExpert(
                family_input_dim=family_input_dim,
                member_input_dim=member_input_dim,
                output_dim=config.d_model,
                config=config,
                use_set_attention=is_first_layer,  # 第一层使用Set Attention
                dropout=dropout
            )
            self.shared_experts.append(expert)

        # ========== 任务特定专家 ==========
        # 每个任务的专家结构与共享专家相同
        self.task_experts = nn.ModuleDict()
        task_names = ['family_repr', 'member_repr', 'family_pattern', 'individual_pattern']

        for task_name in task_names:
            experts = nn.ModuleList()
            for i in range(num_task_experts):
                expert = UnifiedExpert(
                    family_input_dim=family_input_dim,
                    member_input_dim=member_input_dim,
                    output_dim=config.d_model,
                    config=config,
                    use_set_attention=is_first_layer,
                    dropout=dropout
                )
                experts.append(expert)
            self.task_experts[task_name] = experts

        # ========== 共享门控（非最后一层）==========
        if not is_last_layer:
            # 共享门控：从所有共享专家中选择，输出共享特征给下一层
            # 门控输入：家庭特征 + 成员特征的聚合
            gate_input_dim = family_input_dim + member_input_dim
            self.shared_gate_family = GatingNetwork(
                input_dim=gate_input_dim,
                num_experts=num_shared_experts,
                hidden_dim=config.d_model // 2,
                dropout=dropout
            )
            self.shared_gate_member = GatingNetwork(
                input_dim=gate_input_dim,
                num_experts=num_shared_experts,
                hidden_dim=config.d_model // 2,
                dropout=dropout
            )

        # ========== 任务特定门控 ==========
        # 每个任务从 共享专家 + 该任务的特定专家 中选择
        total_experts_per_task = num_shared_experts + num_task_experts
        gate_input_dim = family_input_dim + member_input_dim

        self.task_gates = nn.ModuleDict()
        for task_name in task_names:
            # 家庭级门控
            family_gate = GatingNetwork(
                input_dim=gate_input_dim,
                num_experts=total_experts_per_task,
                hidden_dim=config.d_model // 2,
                dropout=dropout
            )
            # 成员级门控
            member_gate = GatingNetwork(
                input_dim=gate_input_dim,
                num_experts=total_experts_per_task,
                hidden_dim=config.d_model // 2,
                dropout=dropout
            )
            self.task_gates[f'{task_name}_family'] = family_gate
            self.task_gates[f'{task_name}_member'] = member_gate

    def forward(
        self,
        family_input: torch.Tensor,
        member_input: torch.Tensor,
        member_mask: torch.BoolTensor,
        prev_shared_family: torch.Tensor = None,
        prev_shared_member: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        CGC层前向传播

        Args:
            family_input: (batch, family_input_dim) 家庭特征
            member_input: (batch, max_members, member_input_dim) 成员特征
            member_mask: (batch, max_members) 有效成员掩码
            prev_shared_family: (batch, d_model) 上一层共享门控的家庭输出
            prev_shared_member: (batch, max_members, d_model) 上一层共享门控的成员输出

        Returns:
            outputs: dict containing:
                - family_repr: (batch, d_model)
                - member_repr: (batch, max_members, d_model)
                - family_pattern: (batch, d_model)
                - individual_pattern: (batch, max_members, d_model)
                - shared_family: (batch, d_model) 共享门控输出（非最后一层）
                - shared_member: (batch, max_members, d_model) 共享门控输出（非最后一层）
        """
        batch_size, max_members = member_input.shape[:2]

        # ========== 1. 计算所有专家输出 ==========

        # 共享专家输出
        shared_family_outputs = []  # List of (batch, d_model)
        shared_member_outputs = []  # List of (batch, max_members, d_model)

        for expert in self.shared_experts:
            family_out, member_out = expert(family_input, member_input, member_mask)
            shared_family_outputs.append(family_out)
            shared_member_outputs.append(member_out)

        # 任务特定专家输出
        task_family_outputs = {}  # task_name -> List of (batch, d_model)
        task_member_outputs = {}  # task_name -> List of (batch, max_members, d_model)

        for task_name, experts in self.task_experts.items():
            family_outs = []
            member_outs = []
            for expert in experts:
                family_out, member_out = expert(family_input, member_input, member_mask)
                family_outs.append(family_out)
                member_outs.append(member_out)
            task_family_outputs[task_name] = family_outs
            task_member_outputs[task_name] = member_outs

        # ========== 2. 构建门控输入 ==========
        # 家庭级门控输入：家庭特征 + 成员特征的masked mean
        mask_expanded = member_mask.unsqueeze(-1).float()
        member_pooled = (member_input * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        gate_input_family = torch.cat([family_input, member_pooled], dim=-1)  # (batch, family_dim + member_dim)

        # 成员级门控输入：家庭特征广播 + 成员特征
        family_expanded = family_input.unsqueeze(1).expand(-1, max_members, -1)
        gate_input_member = torch.cat([family_expanded, member_input], dim=-1)  # (batch, max_members, family_dim + member_dim)

        # ========== 3. 共享门控（非最后一层）==========
        outputs = {}

        if not self.is_last_layer:
            # 家庭级共享门控
            shared_family_stack = torch.stack(shared_family_outputs, dim=1)  # (batch, num_shared, d_model)
            shared_family_weights = self.shared_gate_family(gate_input_family)  # (batch, num_shared)
            shared_family = torch.einsum('bn,bnd->bd', shared_family_weights, shared_family_stack)

            # 成员级共享门控
            shared_member_stack = torch.stack(shared_member_outputs, dim=2)  # (batch, max_members, num_shared, d_model)
            shared_member_weights = self.shared_gate_member(gate_input_member)  # (batch, max_members, num_shared)
            shared_member = torch.einsum('bmn,bmnd->bmd', shared_member_weights, shared_member_stack)

            # 残差连接（如果有上一层的共享输出）
            if prev_shared_family is not None:
                shared_family = shared_family + prev_shared_family
            if prev_shared_member is not None:
                shared_member = shared_member + prev_shared_member

            # 应用成员掩码
            shared_member = shared_member * mask_expanded

            outputs['shared_family'] = shared_family
            outputs['shared_member'] = shared_member

        # ========== 4. 任务特定门控 ==========
        task_names = ['family_repr', 'member_repr', 'family_pattern', 'individual_pattern']

        for task_name in task_names:
            # 堆叠该任务可用的专家输出：共享专家 + 任务特定专家
            # 家庭级
            all_family_outputs = shared_family_outputs + task_family_outputs[task_name]
            family_stack = torch.stack(all_family_outputs, dim=1)  # (batch, num_shared + num_task, d_model)

            family_weights = self.task_gates[f'{task_name}_family'](gate_input_family)
            task_family = torch.einsum('bn,bnd->bd', family_weights, family_stack)

            # 成员级
            all_member_outputs = shared_member_outputs + task_member_outputs[task_name]
            member_stack = torch.stack(all_member_outputs, dim=2)  # (batch, max_members, num_shared + num_task, d_model)

            member_weights = self.task_gates[f'{task_name}_member'](gate_input_member)
            task_member = torch.einsum('bmn,bmnd->bmd', member_weights, member_stack)

            # 应用成员掩码
            task_member = task_member * mask_expanded

            # 根据任务类型决定输出
            if task_name in ['family_repr', 'family_pattern']:
                # 家庭级任务：主要使用家庭级输出
                outputs[task_name] = task_family
            else:
                # 成员级任务：主要使用成员级输出
                outputs[task_name] = task_member

        return outputs


# ==================== 模式预测头 ====================

class PatternPredictionHead(nn.Module):
    """
    模式预测头
    将CGC层的pattern表示转换为概率分布
    """

    def __init__(
        self,
        d_model: int,
        num_family_patterns: int,
        num_individual_patterns: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.family_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_family_patterns)
        )

        self.individual_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_individual_patterns)
        )

    def forward(
        self,
        family_pattern: torch.Tensor,
        individual_pattern: torch.Tensor,
        member_mask: torch.BoolTensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            family_pattern: (batch, d_model)
            individual_pattern: (batch, max_members, d_model)
            member_mask: (batch, max_members)

        Returns:
            family_pattern_logits: (batch, num_family_patterns)
            family_pattern_prob: (batch, num_family_patterns)
            individual_pattern_logits: (batch, max_members, num_individual_patterns)
            individual_pattern_prob: (batch, max_members, num_individual_patterns)
        """
        family_logits = self.family_head(family_pattern)
        family_prob = F.softmax(family_logits, dim=-1)

        individual_logits = self.individual_head(individual_pattern)
        individual_prob = F.softmax(individual_logits, dim=-1)

        if member_mask is not None:
            individual_prob = individual_prob * member_mask.unsqueeze(-1).float()

        return family_logits, family_prob, individual_logits, individual_prob


# ==================== PLE编码器（重构版）====================

class PLEEncoder(nn.Module):
    """
    PLE编码器（重构版）

    架构:
    1. 可选的位置嵌入融合
    2. 多层CGC (Customized Gate Control)
       - 每层包含：共享专家 + 任务特定专家
       - 共享门控（非最后一层）：输出共享特征供下一层感知
       - 任务特定门控：每个任务从共享专家+任务专家中选择
    3. 模式预测头

    四个任务:
    - family_repr: 家庭信息表示
    - member_repr: 成员信息表示
    - family_pattern_prob: 家庭模式概率
    - individual_pattern_prob: 个人模式概率
    """

    def __init__(
        self,
        config: ModelConfig,
        num_family_patterns: int = 118,
        num_individual_patterns: int = 207,
        num_shared_experts: int = 2,
        num_task_experts: int = 1,
        num_cgc_layers: int = 3
    ):
        super().__init__()
        self.config = config
        self.num_family_patterns = num_family_patterns
        self.num_individual_patterns = num_individual_patterns
        self.num_cgc_layers = num_cgc_layers
        self.use_pattern_experts = getattr(config, 'use_pattern_condition', True)

        # ========== 位置嵌入模块 ==========
        self.use_zone_embedding = getattr(config, 'use_destination_prediction', False)
        if self.use_zone_embedding:
            from zone_embedding import FamilyMemberZoneEmbedding
            self.zone_embedding = FamilyMemberZoneEmbedding(config)

            # 家庭特征融合层（原始特征 + home嵌入）
            self.family_zone_fusion = nn.Sequential(
                nn.Linear(config.family_dim + config.zone_embed_dim, config.family_dim),
                nn.ReLU(),
                nn.LayerNorm(config.family_dim)
            )

            # 成员特征融合层（原始特征 + work嵌入）
            self.member_zone_fusion = nn.Sequential(
                nn.Linear(config.member_dim + config.zone_embed_dim, config.member_dim),
                nn.ReLU(),
                nn.LayerNorm(config.member_dim)
            )

        # ========== CGC层 ==========
        self.cgc_layers = nn.ModuleList()
        for i in range(num_cgc_layers):
            is_first = (i == 0)
            is_last = (i == num_cgc_layers - 1)

            # 确定输入维度
            if is_first:
                family_input_dim = config.family_dim
                member_input_dim = config.member_dim
            else:
                # 后续层：输入来自上一层的共享门控输出
                family_input_dim = config.d_model
                member_input_dim = config.d_model

            layer = CGCLayer(
                config=config,
                num_shared_experts=num_shared_experts,
                num_task_experts=num_task_experts,
                family_input_dim=family_input_dim,
                member_input_dim=member_input_dim,
                is_first_layer=is_first,
                is_last_layer=is_last,
                dropout=config.dropout
            )
            self.cgc_layers.append(layer)

        # ========== 模式预测头 ==========
        if self.use_pattern_experts:
            self.pattern_head = PatternPredictionHead(
                d_model=config.d_model,
                num_family_patterns=num_family_patterns,
                num_individual_patterns=num_individual_patterns,
                dropout=config.dropout
            )

    def forward(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.BoolTensor,
        home_zones: torch.Tensor = None,
        work_zones: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        前向传播

        Args:
            family_attr: (batch, family_dim) 家庭属性
            member_attr: (batch, max_members, member_dim) 成员属性
            member_mask: (batch, max_members) 有效成员标记
            home_zones: (batch,) 家庭住所zone ID
            work_zones: (batch, max_members) 成员工作地zone ID

        Returns:
            member_repr: (batch, max_members, d_model) 成员表示
            family_repr: (batch, d_model) 家庭表示
            pattern_outputs: dict 包含模式预测结果
        """
        # ========== 1. 融合位置嵌入 ==========
        if self.use_zone_embedding and home_zones is not None and work_zones is not None:
            home_embed = self.zone_embedding.get_home_embedding(home_zones)
            family_attr = self.family_zone_fusion(
                torch.cat([family_attr, home_embed], dim=-1)
            )

            work_embed = self.zone_embedding.get_work_embedding(work_zones)
            member_attr = self.member_zone_fusion(
                torch.cat([member_attr, work_embed], dim=-1)
            )

        # ========== 2. 多层CGC ==========
        family_input = family_attr
        member_input = member_attr
        shared_family = None
        shared_member = None

        for i, cgc_layer in enumerate(self.cgc_layers):
            outputs = cgc_layer(
                family_input=family_input,
                member_input=member_input,
                member_mask=member_mask,
                prev_shared_family=shared_family,
                prev_shared_member=shared_member
            )

            # 获取共享门控输出（用于下一层）
            if 'shared_family' in outputs:
                shared_family = outputs['shared_family']
                shared_member = outputs['shared_member']

                # 下一层的输入使用共享门控的输出
                family_input = shared_family
                member_input = shared_member

        # 获取最后一层的任务输出
        family_repr = outputs['family_repr']
        member_repr = outputs['member_repr']
        family_pattern = outputs['family_pattern']
        individual_pattern = outputs['individual_pattern']

        # ========== 3. 模式预测 ==========
        if self.use_pattern_experts:
            family_logits, family_prob, individual_logits, individual_prob = self.pattern_head(
                family_pattern, individual_pattern, member_mask
            )

            pattern_outputs = {
                'family_pattern_prob': family_prob,
                'individual_pattern_prob': individual_prob,
                'family_pattern_logits': family_logits,
                'individual_pattern_logits': individual_logits
            }
        else:
            pattern_outputs = None

        return member_repr, family_repr, pattern_outputs

    def forward_with_pattern_logits(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.BoolTensor,
        home_zones: torch.Tensor = None,
        work_zones: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        返回带logits的前向传播（与forward相同，保持接口兼容）
        """
        return self.forward(family_attr, member_attr, member_mask, home_zones, work_zones)


