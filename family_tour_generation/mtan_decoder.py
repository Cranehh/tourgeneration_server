"""
MTAN Decoder
包含:
- 任务特定注意力 (MTAN): 每个成员从家庭上下文选择性提取信息
- Cross-Role Attention: 成员间信息融合
- 时间硬约束: 确保活动时间顺序合理
- 共享Transformer Decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from config import ModelConfig
from embedding import ActivityEmbedding, PositionalEncoding


# ==================== 新增：模式条件模块 ====================

class PatternCrossAttention(nn.Module):
    """
    模式交叉注意力
    让解码器从预测的活动模式中提取条件信息
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self,
            hidden: torch.Tensor,
            pattern_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden: (batch, seq_len, d_model) 解码器隐状态
            pattern_emb: (batch, d_model) 模式嵌入

        Returns:
            (batch, seq_len, d_model) 融合模式信息后的隐状态
        """
        # 扩展 pattern_emb 作为 key 和 value
        pattern_kv = pattern_emb.unsqueeze(1)  # (batch, 1, d_model)

        attn_out, _ = self.cross_attn(hidden, pattern_kv, pattern_kv)

        return self.norm(hidden + attn_out)


class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization (adaLN) 调制

    用模式条件调制 LayerNorm 的 scale 和 shift
    参考: DiT (Diffusion Transformer)
    """

    def __init__(self, d_model: int, condition_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # 从条件生成 scale (gamma) 和 shift (beta)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, d_model * 2)
        )

        # 初始化为恒等映射
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, ..., d_model) 输入
            condition: (batch, condition_dim) 条件向量

        Returns:
            (batch, ..., d_model) 调制后的输出
        """
        # 生成 scale 和 shift
        modulation = self.adaLN_modulation(condition)  # (batch, d_model * 2)
        gamma, beta = modulation.chunk(2, dim=-1)  # 各 (batch, d_model)

        # 扩展到与 x 相同的维度
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        # 应用 adaLN: norm(x) * (1 + gamma) + beta
        return self.norm(x) * (1 + gamma) + beta


class PatternConditionModule(nn.Module):
    """
    模式条件模块

    组合交叉注意力和 adaLN 调制
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_family_patterns: int = 118,
            num_individual_patterns: int = 207,
            dropout: float = 0.1
    ):
        super().__init__()

        # 模式嵌入投影
        self.family_pattern_proj = nn.Linear(num_family_patterns, d_model)
        self.individual_pattern_proj = nn.Linear(num_individual_patterns, d_model)

        # 模式交叉注意力
        self.family_pattern_attn = PatternCrossAttention(d_model, num_heads, dropout)
        self.individual_pattern_attn = PatternCrossAttention(d_model, num_heads, dropout)

        # adaLN 调制 (用组合的模式条件)
        self.adaLN = AdaLNModulation(d_model, d_model * 2)

        # 融合家庭和个人模式
        self.pattern_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2)
        )

    def forward(
            self,
            hidden: torch.Tensor,
            family_pattern_prob: torch.Tensor,
            individual_pattern_prob: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden: (batch, seq_len, d_model) 或 (batch, max_members, seq_len, d_model)
            family_pattern_prob: (batch, num_family_patterns)
            individual_pattern_prob: (batch, num_individual_patterns) 或 (batch, max_members, num_individual_patterns)

        Returns:
            调制后的 hidden
        """
        # 投影模式概率到嵌入空间
        family_emb = self.family_pattern_proj(family_pattern_prob)  # (batch, d_model)

        # 处理 individual_pattern_prob 的维度
        if individual_pattern_prob.dim() == 3:
            # (batch, max_members, num_patterns) -> 需要逐成员处理
            batch_size, max_members, _ = individual_pattern_prob.shape
            individual_emb = self.individual_pattern_proj(individual_pattern_prob)  # (batch, max_members, d_model)
        else:
            individual_emb = self.individual_pattern_proj(individual_pattern_prob)  # (batch, d_model)

        # 根据 hidden 的维度决定处理方式
        if hidden.dim() == 4:
            # (batch, max_members, seq_len, d_model)
            batch_size, max_members, seq_len, d_model = hidden.shape

            # 展平处理
            hidden_flat = hidden.view(batch_size * max_members, seq_len, d_model)

            # 扩展 family_emb
            family_emb_expanded = family_emb.unsqueeze(1).expand(-1, max_members, -1)
            family_emb_flat = family_emb_expanded.reshape(batch_size * max_members, d_model)

            # individual_emb 已经是 (batch, max_members, d_model)
            individual_emb_flat = individual_emb.view(batch_size * max_members, d_model)

            # 交叉注意力
            hidden_flat = self.family_pattern_attn(hidden_flat, family_emb_flat)
            hidden_flat = self.individual_pattern_attn(hidden_flat, individual_emb_flat)

            # 融合条件用于 adaLN
            combined_condition = torch.cat([family_emb_flat, individual_emb_flat], dim=-1)
            combined_condition = self.pattern_fusion(combined_condition)

            # adaLN 调制
            hidden_flat = self.adaLN(hidden_flat, combined_condition)

            # 恢复形状
            hidden = hidden_flat.view(batch_size, max_members, seq_len, d_model)

        else:
            # (batch, seq_len, d_model)
            hidden = self.family_pattern_attn(hidden, family_emb)
            hidden = self.individual_pattern_attn(hidden, individual_emb)

            combined_condition = torch.cat([family_emb, individual_emb], dim=-1)
            combined_condition = self.pattern_fusion(combined_condition)
            hidden = self.adaLN(hidden, combined_condition)

        return hidden


class PatternConditionedDecoderLayer(nn.Module):
    """
    带模式条件调制的 Decoder Layer

    结构: Self-Attention → Cross-Attention → Pattern Condition → FFN
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            num_family_patterns: int = 118,
            num_individual_patterns: int = 207,
            dropout: float = 0.1
    ):
        super().__init__()

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-Attention (with memory)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Pattern Condition (adaLN 调制)
        self.pattern_proj_family = nn.Linear(num_family_patterns, d_model)
        self.pattern_proj_individual = nn.Linear(num_individual_patterns, d_model)
        self.adaLN = AdaLNModulation(d_model, d_model * 2)
        self.pattern_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU()
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            family_pattern_prob: torch.Tensor = None,
            individual_pattern_prob: torch.Tensor = None,
            tgt_mask: torch.Tensor = None,
            tgt_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: (batch * max_members, seq_len, d_model)
            memory: (batch * max_members, seq_len, d_model)
            family_pattern_prob: (batch, num_family_patterns)
            individual_pattern_prob: (batch, max_members, num_individual_patterns)
            tgt_mask: causal mask
            tgt_key_padding_mask: padding mask

        Returns:
            (batch * max_members, seq_len, d_model)
        """
        # 1. Self-Attention
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_out))

        # 2. Cross-Attention
        attn_out, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(attn_out))

        # 3. Pattern Condition (adaLN)
        if family_pattern_prob is not None and individual_pattern_prob is not None:
            tgt = self._apply_pattern_condition(tgt, family_pattern_prob, individual_pattern_prob)

        # 4. FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + ffn_out)

        return tgt

    def _apply_pattern_condition(
            self,
            hidden: torch.Tensor,
            family_pattern_prob: torch.Tensor,
            individual_pattern_prob: torch.Tensor
    ) -> torch.Tensor:
        """
        应用模式条件调制

        Args:
            hidden: (batch * max_members, seq_len, d_model)
            family_pattern_prob: (batch, num_family_patterns)
            individual_pattern_prob: (batch, max_members, num_individual_patterns)
        """
        batch_size = family_pattern_prob.size(0)
        max_members = individual_pattern_prob.size(1)
        seq_len = hidden.size(1)

        # 投影模式信息
        family_emb = self.pattern_proj_family(family_pattern_prob)  # (batch, d_model)
        individual_emb = self.pattern_proj_individual(individual_pattern_prob)  # (batch, max_members, d_model)

        # 扩展 family_emb
        family_emb = family_emb.unsqueeze(1).expand(-1, max_members, -1)  # (batch, max_members, d_model)

        # 展平
        family_emb_flat = family_emb.reshape(batch_size * max_members, -1)  # (batch * max_members, d_model)
        individual_emb_flat = individual_emb.reshape(batch_size * max_members, -1)  # (batch * max_members, d_model)

        # 融合条件
        combined = torch.cat([family_emb_flat, individual_emb_flat], dim=-1)
        combined = self.pattern_fusion(combined)  # (batch * max_members, d_model * 2)

        # adaLN 调制
        hidden = self.adaLN(hidden, combined)

        return hidden


class PatternConditionedDecoder(nn.Module):
    """
    带模式条件的 Decoder
    每层都注入模式信息
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            num_layers: int,
            num_family_patterns: int = 118,
            num_individual_patterns: int = 207,
            dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            PatternConditionedDecoderLayer(
                d_model, num_heads, d_ff,
                num_family_patterns, num_individual_patterns, dropout
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            family_pattern_prob: torch.Tensor = None,
            individual_pattern_prob: torch.Tensor = None,
            tgt_mask: torch.Tensor = None,
            tgt_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: (batch * max_members, seq_len, d_model)
            memory: (batch * max_members, seq_len, d_model)
            family_pattern_prob: (batch, num_family_patterns)
            individual_pattern_prob: (batch, max_members, num_individual_patterns)
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output, memory,
                family_pattern_prob, individual_pattern_prob,
                tgt_mask, tgt_key_padding_mask
            )

        return output


class CrossRoleAttention(nn.Module):
    """
    跨成员注意力层
    让每个成员能看到其他成员当前的状态
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.cross_attn = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self, 
        member_states: torch.Tensor, 
        member_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Args:
            member_states: (batch, max_members, d_model) 各成员当前状态
            member_mask: (batch, max_members) 有效成员标记
        
        Returns:
            (batch, max_members, d_model) 融合后的状态
        """
        batch_size, max_members, d_model = member_states.shape
        attn_mask = torch.eye(max_members, device=member_states.device).bool()
        # 构建attention mask: 每个成员注意其他成员，不注意自己
        # (batch, max_members, max_members)
        # self_mask = torch.eye(max_members, device=member_states.device).bool()
        # self_mask = self_mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, max_members, max_members)
        #
        # # 无效成员也需要mask
        # invalid_mask = ~member_mask.unsqueeze(1).expand(-1, max_members, -1)  # (batch, max_members, max_members)
        #
        # # 合并mask: 自己 + 无效成员
        # attn_mask = self_mask | invalid_mask  # True表示忽略
        #
        # # 转换为attention mask格式 (需要float, -inf表示忽略)
        # attn_mask_float = torch.zeros_like(attn_mask, dtype=torch.float)
        # attn_mask_float = attn_mask_float.clone()
        # attn_mask_float = attn_mask_float.masked_fill(attn_mask, float('-inf'))
        #
        # # Cross attention
        #
        # H = self.cross_attn.num_heads
        #
        # attn_mask_expanded = attn_mask_float.repeat_interleave(H, dim=0)  # (B*H, M, M)
        key_padding_mask = ~member_mask
        # 执行attention
        attn_out, _ = self.cross_attn(
            member_states, member_states, member_states,
            # attn_mask=attn_mask,  # (M, M)
            key_padding_mask=key_padding_mask  # (batch * max_members, max_members)
        )
        nan_member_states = torch.isnan(member_states).sum()
        nan_attn_out = torch.isnan(attn_out).sum()
        # 残差连接
        output = self.norm(member_states + attn_out)
        
        # 确保输出不与输入共享内存
        output = output.contiguous()
        
        return output


class TaskSpecificAttention(nn.Module):
    """
    任务特定注意力 (MTAN风格)
    每个成员从家庭上下文中选择性提取信息
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self, 
        member_states: torch.Tensor, 
        family_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            member_states: (batch, max_members, d_model) 成员状态
            family_context: (batch, d_model) 家庭上下文
        
        Returns:
            (batch, max_members, d_model) 增强后的成员状态
        """
        # 扩展family_context作为key和value
        family_kv = family_context.unsqueeze(1)  # (batch, 1, d_model)
        
        # 每个成员作为query
        attn_out, _ = self.attn(member_states, family_kv, family_kv)
        
        output = self.norm(member_states + attn_out)
        return output.contiguous()


class OutputHeads(nn.Module):
    """
    输出头
    分别预测各个活动属性
    支持接收模式信息作为条件
    """

    def __init__(self, config: ModelConfig,
                 num_family_patterns: int = 118,
                 num_individual_patterns: int = 207):
        super().__init__()
        self.config = config
        self.num_family_patterns = num_family_patterns
        self.num_individual_patterns = num_individual_patterns

        # 模式信息投影
        self.family_pattern_proj = nn.Linear(num_family_patterns, config.d_model)
        self.individual_pattern_proj = nn.Linear(num_individual_patterns, config.d_model)

        ## norm化
        self.individual_pattern_norm = nn.LayerNorm(config.d_model)
        self.family_pattern_norm = nn.LayerNorm(config.d_model)

        # 输入维度：d_model
        input_dim = config.d_model

        # 连续属性回归头
        self.continuous_head = nn.Sequential(
            nn.Linear(input_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.continuous_dim)
        )

        # 离散属性分类头
        self.purpose_head = nn.Sequential(
            nn.Linear(input_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_purposes)
        )

        self.mode_head = nn.Sequential(
            nn.Linear(input_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_modes)
        )

        self.driver_head = nn.Sequential(
            nn.Linear(input_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_driver_status)
        )

        self.joint_head = nn.Sequential(
            nn.Linear(input_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_joint_status)
        )
        # ===== 新增：目的地预测头 =====
        if self.use_destination:
            # Query投影（融合hidden + origin嵌入 + purpose信息）
            self.dest_query_proj = nn.Sequential(
                nn.Linear(config.d_model + config.zone_embed_dim + config.num_purposes, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model)
            )

            # Key投影
            self.dest_key_proj = nn.Linear(config.d_model, config.d_model)

            # 可学习的偏置权重
            self.impedance_scale = nn.Parameter(torch.tensor(1.0))
            self.affordance_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
            self,
            hidden: torch.Tensor,
            family_pattern_prob: torch.Tensor = None,
            individual_pattern_prob: torch.Tensor = None,
            origin_zones: torch.Tensor = None  # 新增参数: [...] 当前活动的origin
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden: (*, d_model) decoder输出
            family_pattern_prob: (B, num_family_patterns)
            individual_pattern_prob: (B, M, num_individual_patterns)
            origin_zones: (B, M) 或 (B, M, T) 当前活动的出发地zone ID
        """
        family_pattern_emb = self.family_pattern_proj(family_pattern_prob)  # (B, d_model)
        if hidden.dim() == 4:
            family_pattern_emb = family_pattern_emb.unsqueeze(1).unsqueeze(2).expand(-1, hidden.size(1), hidden.size(2),
                                                                                     -1)
        elif hidden.dim() == 3:
            family_pattern_emb = family_pattern_emb.unsqueeze(1).expand(-1, hidden.size(1), -1)
        hidden = family_pattern_emb + hidden  # (*, d_model)
        hidden = self.family_pattern_norm(hidden)

        # 如果有模式信息，拼接到 hidden
        # 投影模式信息
        individual_pattern_emb = self.individual_pattern_proj(individual_pattern_prob)  # (B, M, d_model//4)

        # 扩展到与 hidden 相同的维度
        if hidden.dim() == 4:
            # hidden: (B, M, T, d_model)
            individual_pattern_emb = individual_pattern_emb.unsqueeze(2).expand(-1, -1, hidden.size(2), -1)
        elif hidden.dim() == 3 and individual_pattern_emb.dim() == 3:
            # hidden: (B, M, d_model), pattern_emb: (B, M, d_model//4)
            pass  # 维度已匹配

        # 拼接
        hidden = individual_pattern_emb + hidden  # (*, d_model + d_model//4)
        hidden = self.individual_pattern_norm(hidden)
        purpose_logits = self.purpose_head(hidden),
        results = {
            'continuous': self.continuous_head(hidden),
            'purpose': self.purpose_head(hidden),
            'mode': self.mode_head(hidden),
            'driver': self.driver_head(hidden),
            'joint': self.joint_head(hidden)
        }

        # ===== 新增：目的地预测 =====
        if self.use_destination and origin_zones is not None:
            dest_logits = self._predict_destination(
                hidden=hidden,
                purpose_logits=purpose_logits,
                origin_zones=origin_zones
            )
            results['destination'] = dest_logits

        return results

    def _predict_destination(
            self,
            hidden: torch.Tensor,  # [..., d_model]
            purpose_logits: torch.Tensor,  # [..., num_purposes]
            origin_zones: torch.Tensor  # [...]
    ) -> torch.Tensor:
        """
        目的地预测
        Returns: [..., num_zones]
        """
        # 1. 获取origin的嵌入
        origin_embed = self.zone_module.get_zone_embedding(origin_zones)  # [..., zone_embed_dim]

        # 2. 获取purpose概率
        purpose_probs = F.softmax(purpose_logits, dim=-1)  # [..., num_purposes]

        # 3. 构建Query（融合hidden + origin + purpose）
        query_input = torch.cat([hidden, origin_embed, purpose_probs], dim=-1)
        query = self.dest_query_proj(query_input)  # [..., d_model]

        # 4. 获取所有zone的Key
        zone_keys = self.zone_module.get_all_zone_keys()  # [num_zones, d_model]
        zone_keys = self.dest_key_proj(zone_keys)  # [num_zones, d_model]

        # 5. 计算注意力分数
        # query: [..., d_model], zone_keys: [num_zones, d_model]
        d_k = query.size(-1)
        attn_scores = torch.einsum('...d,nd->...n', query, zone_keys) / (d_k ** 0.5)
        # attn_scores: [..., num_zones]

        # 6. 添加空间阻抗偏置（距离越远，分数越低）
        impedance = self.zone_module.get_impedance(origin_zones)  # [..., num_zones]
        attn_scores = attn_scores - self.impedance_scale * impedance

        # 7. 添加affordance偏置（活动类型匹配度）
        affordance_scores = self.zone_module.get_affordance_scores(purpose_probs)  # [..., num_zones]
        attn_scores = attn_scores + self.affordance_scale * affordance_scores

        return attn_scores  # 返回logits，不做softmax


class TimeConstraintModule(nn.Module):
    """
    时间硬约束模块
    确保: 当前活动的开始时间 >= 上一个活动的结束时间
    
    在z-score空间中操作
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def apply_constraint(
        self, 
        pred_continuous: torch.Tensor, 
        prev_continuous: torch.Tensor,
        is_first_activity: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Args:
            pred_continuous: (*, 2) 预测的 [开始时间z-score, 结束时间z-score]
            prev_continuous: (*, 2) 上一个活动的 [开始时间z-score, 结束时间z-score]
            is_first_activity: (*,) 是否是第一个活动
        
        Returns:
            (*, 2) 约束后的时间
        """
        pred_start = pred_continuous[..., 0]    # 预测的开始时间
        pred_end = pred_continuous[..., 1]      # 预测的结束时间
        prev_end = prev_continuous[..., 1]      # 上一个活动的结束时间
        
        # 约束1: 开始时间 >= 上一个活动的结束时间
        # 对于第一个活动，不施加此约束
        constrained_start = torch.where(
            is_first_activity,
            pred_start,
            torch.maximum(pred_start, prev_end)
        )
        
        # 约束2: 结束时间 >= 开始时间
        # 计算预测的持续时间 (结束 - 开始)
        pred_duration = pred_end - pred_start
        # 持续时间不能为负
        constrained_duration = F.relu(pred_duration) + 0.01  # 最小持续时间
        constrained_end = constrained_start + constrained_duration
        
        return torch.stack([constrained_start, constrained_end], dim=-1)


class MTANDecoder(nn.Module):
    """
    MTAN Decoder
    
    结构:
    1. 活动嵌入 + 位置编码
    2. 任务特定注意力 (从家庭上下文提取)
    3. Cross-Role注意力 (成员间交互)
    4. Transformer Decoder层
    5. 输出头 + 时间约束
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 活动嵌入
        self.activity_embedding = ActivityEmbedding(config)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_activities, config.dropout)
        
        # 任务特定注意力
        self.task_attention = TaskSpecificAttention(config)
        
        # Cross-Role注意力
        self.cross_role_attention = CrossRoleAttention(config)

        # Transformer Decoder层（带模式条件）
        self.use_pattern_conditioned_decoder = getattr(config, 'use_pattern_condition', True)

        if self.use_pattern_conditioned_decoder:
            self.transformer_decoder = PatternConditionedDecoder(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                num_layers=config.num_decoder_layers,
                num_family_patterns=getattr(config, 'num_family_patterns', 118),
                num_individual_patterns=getattr(config, 'num_individual_patterns', 207),
                dropout=config.dropout
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=config.num_decoder_layers
            )

        # ===== 新增：Zone模块 =====
        self.use_destination = getattr(config, 'use_destination_prediction', False)
        if self.use_destination:
            from zone_embedding import ZoneEmbeddingModule
            self.zone_module = ZoneEmbeddingModule(config)
        else:
            self.zone_module = None

        # ===== 修改：OutputHeads传入zone_module =====
        self.output_heads = OutputHeads(
            config,
            zone_module=self.zone_module,  # 新增
            num_family_patterns=getattr(config, 'num_family_patterns', 118),
            num_individual_patterns=getattr(config, 'num_individual_patterns', 207)
        )
        
        # 时间约束
        self.time_constraint = TimeConstraintModule(config)
        
        # 起始token (用于自回归生成的第一步)
        self.start_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # 新增: 模式条件模块
        self.use_pattern_condition = getattr(config, 'use_pattern_condition', True)
        if self.use_pattern_condition:
            self.pattern_condition = PatternConditionModule(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_family_patterns=getattr(config, 'num_family_patterns', 118),
                num_individual_patterns=getattr(config, 'num_individual_patterns', 207),
                dropout=config.dropout
            )
    
    def forward(
        self,
        member_repr: torch.Tensor,
        family_repr: torch.Tensor,
        target_activities: torch.Tensor,
        member_mask: torch.BoolTensor,
        activity_mask: torch.BoolTensor,
        pattern_outputs: Dict[str, torch.Tensor] = None,
        home_zones: torch.Tensor = None,  # 新增: [B]
        target_destinations: torch.Tensor = None  # 新增: [B, M, T] 用于计算origin
    ) -> Dict[str, torch.Tensor]:
        """
        训练阶段: Teacher Forcing
        
        Args:
            member_repr: (batch, max_members, d_model) PLE编码的成员表示
            family_repr: (batch, d_model) 家庭表示
            target_activities: (batch, max_members, max_activities, activity_dim) 目标活动链
            member_mask: (batch, max_members) 有效成员
            activity_mask: (batch, max_members, max_activities) 有效活动
        
        Returns:
            predictions: dict of (batch, max_members, max_activities, *)
        """
        batch_size = member_repr.size(0)
        max_members = self.config.max_members
        max_activities = self.config.max_activities
        
        # 嵌入目标活动 (用于teacher forcing)
        # 输入是shifted: 用前一个活动预测当前活动
        activity_emb = self.activity_embedding(target_activities)  # (batch, max_members, max_activities, d_model)
        
        # 添加位置编码
        activity_emb = activity_emb.view(batch_size * max_members, max_activities, -1)
        activity_emb = self.pos_encoding(activity_emb)
        activity_emb = activity_emb.view(batch_size, max_members, max_activities, -1)

        nan_activity_emb = torch.isnan(activity_emb).sum()

        # 准备decoder输入: shift right, 加入start token
        start_tokens = self.start_token.expand(batch_size, max_members, 1, -1)
        nan_start_tokens = torch.isnan(start_tokens).sum()
        decoder_input = torch.cat([start_tokens, activity_emb[:, :, :-1, :]], dim=2)
        # (batch, max_members, max_activities, d_model)
        
        # 任务特定注意力: 融入家庭上下文
        # 先聚合每个成员的序列信息
        member_states = decoder_input.mean(dim=2)  # (batch, max_members, d_model)
        member_states = member_states.clone() + member_repr  # 加入PLE编码
        member_states = self.task_attention(member_states, family_repr)
        nan_member_states = torch.isnan(member_states).sum()
        nan_member_repr = torch.isnan(member_repr).sum()
        
        # Cross-Role注意力: 成员间交互
        # TODO:这一步有问题
        member_states = self.cross_role_attention(member_states, member_mask)
        nan_member_states = torch.isnan(member_states).sum()
        # 将成员上下文融入decoder输入
        member_context = member_states.unsqueeze(2).expand(-1, -1, max_activities, -1)
        decoder_input = decoder_input.clone() + member_context
        nan_member_context = torch.isnan(member_context).sum()
        
        # 准备memory (来自成员表示和家庭表示)
        # 将member_repr作为memory
        memory = member_repr  # (batch, max_members, d_model)
        
        # Transformer Decoder
        # 需要处理每个成员的序列
        # 重塑为 (batch * max_members, max_activities, d_model)
        nan_decoder_input = torch.isnan(decoder_input).sum()
        decoder_input_flat = decoder_input.view(batch_size * max_members, max_activities, -1)
        memory_flat = memory.unsqueeze(2).expand(-1, -1, max_activities, -1)
        memory_flat = memory_flat.view(batch_size * max_members, max_activities, -1)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(max_activities, max_activities, device=decoder_input.device),
            diagonal=1
        ).bool()
        
        # Activity mask for padding
        activity_mask_flat = activity_mask.view(batch_size * max_members, max_activities)
        tgt_key_padding_mask = ~activity_mask_flat

        nan1 = torch.isnan(decoder_input_flat).sum()
        nan2 = torch.isnan(memory_flat).sum()
        nan3 = torch.isnan(causal_mask).sum()
        nan4 = torch.isnan(tgt_key_padding_mask).sum()



        causal_mask = causal_mask.clone()
        causal_mask.fill_diagonal_(False)
        # Decoder
        # Decoder
        if self.use_pattern_conditioned_decoder and pattern_outputs is not None:
            decoded = self.transformer_decoder(
                tgt=decoder_input_flat,
                memory=memory_flat,
                family_pattern_prob=pattern_outputs.get('family_pattern_prob'),
                individual_pattern_prob=pattern_outputs.get('individual_pattern_prob'),
                tgt_mask=causal_mask,
                tgt_key_padding_mask=None
            )
        else:
            decoded = self.transformer_decoder(
                tgt=decoder_input_flat,
                memory=memory_flat,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=None
            )

        nan_decoded = torch.isnan(decoded).sum()
        # 恢复形状
        decoded = decoded.view(batch_size, max_members, max_activities, -1)

        # 新增: 对 decoder 输出进行模式调制
        if self.use_pattern_condition and pattern_outputs is not None:
            decoded = self.pattern_condition(
                decoded,
                pattern_outputs['family_pattern_prob'],
                pattern_outputs['individual_pattern_prob']
            )

        # ===== 计算origin_zones =====
        origin_zones = None
        if self.use_destination and home_zones is not None and target_destinations is not None:

            # 第一个活动的origin是home
            # home_zones: [B] -> [B, M, 1]
            home_expanded = home_zones.unsqueeze(1).unsqueeze(2).expand(-1, max_members, 1)

            # 后续活动的origin是前一个活动的destination
            # target_destinations: [B, M, T]
            prev_destinations = target_destinations[:, :, :-1]  # [B, M, T-1]

            # 拼接: [B, M, T]
            origin_zones = torch.cat([home_expanded, prev_destinations], dim=2)

        # 输出预测
        predictions = self.output_heads(
            hidden=decoded,
            family_pattern_prob=pattern_outputs.get('family_pattern_prob') if pattern_outputs else None,
            individual_pattern_prob=pattern_outputs.get('individual_pattern_prob') if pattern_outputs else None,
            origin_zones=origin_zones  # 新增
        )
        
        # 应用时间约束
        predictions['continuous'] = self._apply_time_constraints(
            predictions['continuous'], 
            target_activities[..., :2],
            activity_mask
        )
        
        return predictions
    
    def _apply_time_constraints(
        self,
        pred_continuous: torch.Tensor,
        target_continuous: torch.Tensor,
        activity_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        训练时应用时间约束
        
        Args:
            pred_continuous: (batch, max_members, max_activities, 2)
            target_continuous: (batch, max_members, max_activities, 2)
            activity_mask: (batch, max_members, max_activities)
        """
        batch_size, max_members, max_activities, _ = pred_continuous.shape
        
        result = pred_continuous.clone()
        
        for t in range(max_activities):
            is_first = (t == 0)
            is_first_tensor = torch.full(
                (batch_size, max_members), is_first, 
                device=pred_continuous.device, dtype=torch.bool
            )
            
            if t > 0:
                # 使用目标的上一个活动时间作为约束参考 (teacher forcing)
                prev_continuous = target_continuous[:, :, t-1, :]
            else:
                prev_continuous = torch.zeros_like(result[:, :, 0, :])
            
            # 避免in-place操作
            constrained_time = self.time_constraint.apply_constraint(
                result[:, :, t, :],
                prev_continuous,
                is_first_tensor
            )
            result = result.clone()
            result[:, :, t, :] = constrained_time
        
        return result
    
    def generate(
        self,
        member_repr: torch.Tensor,
        family_repr: torch.Tensor,
        member_mask: torch.BoolTensor,
        max_length: int = None,
        pattern_outputs: Dict[str, torch.Tensor] = None, # 新增参数,
        home_zones: torch.Tensor = None,  # 新增: [B]
    ) -> Dict[str, torch.Tensor]:
        """
        推理阶段: 自回归生成
        
        Args:
            member_repr: (batch, max_members, d_model)
            family_repr: (batch, d_model)
            member_mask: (batch, max_members)
            max_length: 最大生成长度
        
        Returns:
            generated: dict of (batch, max_members, seq_len, *)
        """

        if max_length is None:
            max_length = self.config.max_activities
        
        batch_size = member_repr.size(0)
        max_members = self.config.max_members
        device = member_repr.device
        
        # 初始化
        generated_continuous = []
        generated_purpose = []
        generated_mode = []
        generated_driver = []
        generated_joint = []
        # 初始化
        if self.use_destination and home_zones is not None:
            prev_destination = home_zones.unsqueeze(1).expand(-1, max_members)  # [B, M]
        else:
            prev_destination = None
        
        # 当前输入 (start token)
        current_input = self.start_token.expand(batch_size, max_members, 1, -1)
        
        # 任务特定注意力和Cross-Role注意力预计算
        member_states = member_repr.clone()
        member_states = self.task_attention(member_states, family_repr)
        member_states = self.cross_role_attention(member_states, member_mask)
        
        # 上一个活动的时间 (用于硬约束)
        prev_continuous = torch.zeros(batch_size, max_members, 2, device=device)

        generated_destination = []
        for t in range(max_length):
            # 位置编码
            pos_input = self.pos_encoding(current_input.view(batch_size * max_members, -1, self.config.d_model))
            pos_input = pos_input.view(batch_size, max_members, -1, self.config.d_model)
            
            # 加入成员上下文
            seq_len = pos_input.size(2)
            member_context = member_states.unsqueeze(2).expand(-1, -1, seq_len, -1)
            decoder_input = pos_input + member_context
            
            # Memory
            memory = member_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)
            
            # 重塑
            decoder_input_flat = decoder_input.view(batch_size * max_members, seq_len, -1)
            memory_flat = memory.view(batch_size * max_members, seq_len, -1)
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device),
                diagonal=1
            ).bool()
            
            # Decode
            if self.use_pattern_conditioned_decoder and pattern_outputs is not None:
                decoded = self.transformer_decoder(
                    tgt=decoder_input_flat,
                    memory=memory_flat,
                    family_pattern_prob=pattern_outputs.get('family_pattern_prob'),
                    individual_pattern_prob=pattern_outputs.get('individual_pattern_prob'),
                    tgt_mask=causal_mask
                )
            else:
                decoded = self.transformer_decoder(
                    tgt=decoder_input_flat,
                    memory=memory_flat,
                    tgt_mask=causal_mask
                )
            # 恢复形状
            decoded = decoded.view(batch_size, max_members, decoded.shape[1], -1)

            # 新增: 对 decoder 输出进行模式调制
            if self.use_pattern_condition and pattern_outputs is not None:
                decoded = self.pattern_condition(
                    decoded,
                    pattern_outputs['family_pattern_prob'],
                    pattern_outputs['individual_pattern_prob']
                )
            decoded = decoded.view(batch_size * max_members, decoded.shape[2], -1)
            # 取最后一个位置的输出
            last_hidden = decoded[:, -1, :]  # (batch * max_members, d_model)
            last_hidden = last_hidden.view(batch_size, max_members, -1)


            # 预测
            step_pred = self.output_heads(last_hidden,
                                        pattern_outputs['family_pattern_prob'],
                                        pattern_outputs['individual_pattern_prob'],
                                        origin_zones=prev_destination
                                        )

            # 获取预测的destination
            if 'destination' in step_pred:
                generated_destination.append(step_pred['destination'])
                prev_destination = step_pred['destination'].argmax(dim=-1)
            # 应用时间约束
            is_first = (t == 0)
            is_first_tensor = torch.full(
                (batch_size, max_members), is_first, device=device, dtype=torch.bool
            )
            constrained_continuous = self.time_constraint.apply_constraint(
                step_pred['continuous'],
                prev_continuous,
                is_first_tensor
            )
            
            # 保存预测
            generated_continuous.append(constrained_continuous)
            # generated_purpose.append(step_pred['purpose'].argmax(dim=-1))
            # generated_mode.append(step_pred['mode'].argmax(dim=-1))
            # generated_driver.append(step_pred['driver'].argmax(dim=-1))
            # generated_joint.append(step_pred['joint'].argmax(dim=-1))

            generated_purpose.append(step_pred['purpose'])
            generated_mode.append(step_pred['mode'])
            generated_driver.append(step_pred['driver'])
            generated_joint.append(step_pred['joint'])
            
            # 更新prev_continuous
            prev_continuous = constrained_continuous
            
            # 构建下一步输入
            next_activity_emb = self.activity_embedding.embed_from_indices(
                constrained_continuous,
                step_pred['purpose'].argmax(dim=-1),
                step_pred['mode'].argmax(dim=-1),
                step_pred['driver'].argmax(dim=-1),
                step_pred['joint'].argmax(dim=-1)
            )  # (batch, max_members, d_model)
            
            current_input = torch.cat([
                current_input, 
                next_activity_emb.unsqueeze(2)
            ], dim=2)

        results = {
            'continuous': torch.stack(generated_continuous, dim=2),
            'purpose': torch.stack(generated_purpose, dim=2),
            'mode': torch.stack(generated_mode, dim=2),
            'driver': torch.stack(generated_driver, dim=2),
            'joint': torch.stack(generated_joint, dim=2)
        }

        if generated_destination:
            results['destination'] = torch.stack(generated_destination, dim=2)

        # 堆叠结果
        return results

def autoregressive_rollout(
        decoder: MTANDecoder,
        member_repr: torch.Tensor,
        family_repr: torch.Tensor,
        target_activities: torch.Tensor,
        member_mask: torch.BoolTensor,
        start_pos: int,
        rollout_length: int = 3,
        pattern_probs: Dict[str, torch.Tensor] = None,
        home_zones: torch.Tensor = None,           # ← 新增
        target_destinations: torch.Tensor = None   # ← 新增
) -> Dict[str, torch.Tensor]:
    """
    从 start_pos 开始做 rollout_length 步自回归

    Args:
        decoder: MTANDecoder 实例
        member_repr: (B, max_members, d_model)
        family_repr: (B, d_model)
        target_activities: (B, max_members, max_activities, 27)
        member_mask: (B, max_members)
        start_pos: 开始 rollout 的位置
        rollout_length: rollout 步数

    Returns:
        predictions: 只包含 rollout 片段的预测 (长度为 rollout_length)
    """
    batch_size, max_members = member_repr.shape[:2]
    device = member_repr.device

    # 用真实标签构建 start_pos 之前的输入
    if start_pos > 0:
        prefix_activities = target_activities[:, :, :start_pos, :]
        prefix_emb = decoder.activity_embedding(prefix_activities)
        prefix_emb = decoder.pos_encoding(
            prefix_emb.view(batch_size * max_members, start_pos, -1)
        ).view(batch_size, max_members, start_pos, -1)
    else:
        prefix_emb = None

    # 准备上下文
    member_states = decoder.task_attention(member_repr, family_repr)
    member_states = decoder.cross_role_attention(member_states, member_mask)

    # 初始化
    start_token = decoder.start_token.expand(batch_size, max_members, 1, -1)
    if prefix_emb is not None:
        current_input = torch.cat([start_token, prefix_emb], dim=2)
    else:
        current_input = start_token

    # 上一个时间点（用于时间约束）
    if start_pos > 0:
        prev_continuous = target_activities[:, :, start_pos - 1, :2]
    else:
        prev_continuous = torch.zeros(batch_size, max_members, 2, device=device)

    # ===== 新增：初始化origin =====
    if decoder.use_destination and home_zones is not None and target_destinations is not None:
        if start_pos == 0:
            prev_destination = home_zones.unsqueeze(1).expand(-1, max_members)
        else:
            prev_destination = target_destinations[:, :, start_pos - 1]
    else:
        prev_destination = None

    # Rollout
    rollout_preds = {'continuous': [], 'purpose': [], 'mode': [], 'driver': [], 'joint': [],'destination': []}

    for t in range(rollout_length):
        seq_len = current_input.size(2)

        # 位置编码 + 成员上下文
        pos_input = decoder.pos_encoding(
            current_input.view(batch_size * max_members, seq_len, -1)
        ).view(batch_size, max_members, seq_len, -1)
        member_context = member_states.unsqueeze(2).expand(-1, -1, seq_len, -1)
        decoder_input = pos_input + member_context

        # Memory
        memory = member_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)

        # Decode
        decoder_input_flat = decoder_input.view(batch_size * max_members, seq_len, -1)
        memory_flat = memory.view(batch_size * max_members, seq_len, -1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        if decoder.use_pattern_conditioned_decoder and pattern_probs is not None:
            decoded = decoder.transformer_decoder(
                tgt=decoder_input_flat,
                memory=memory_flat,
                family_pattern_prob=pattern_probs.get('family_pattern_prob'),
                individual_pattern_prob=pattern_probs.get('individual_pattern_prob'),
                tgt_mask=causal_mask
            )
        else:
            decoded = decoder.transformer_decoder(
                tgt=decoder_input_flat,
                memory=memory_flat,
                tgt_mask=causal_mask
            )
        decoded = decoded.view(batch_size, max_members, decoded.shape[1], -1)
        # 新增: 对 decoder 输出进行模式调制
        if decoder.use_pattern_condition and pattern_probs is not None:
            decoded = decoder.pattern_condition(
                decoded,
                pattern_probs['family_pattern_prob'],
                pattern_probs['individual_pattern_prob']
            )
        decoded = decoded.view(batch_size * max_members, decoded.shape[2], -1)
        last_hidden = decoded[:, -1, :].view(batch_size, max_members, -1)

        step_pred = decoder.output_heads(
            last_hidden,
            pattern_probs['family_pattern_prob'] if pattern_probs else None,
            pattern_probs['individual_pattern_prob'] if pattern_probs else None,
            origin_zones=prev_destination  # ← 新增
        )

        # 获取预测的destination
        if 'destination' in step_pred:
            rollout_preds['destination'].append(step_pred['destination'])
            prev_destination = step_pred['destination'].argmax(dim=-1)

        # 时间约束
        is_first = (start_pos + t == 0)
        is_first_tensor = torch.full((batch_size, max_members), is_first, device=device, dtype=torch.bool)
        step_pred['continuous'] = decoder.time_constraint.apply_constraint(
            step_pred['continuous'], prev_continuous, is_first_tensor
        )

        # 保存
        rollout_preds['continuous'].append(step_pred['continuous'])
        rollout_preds['purpose'].append(step_pred['purpose'])
        rollout_preds['mode'].append(step_pred['mode'])
        rollout_preds['driver'].append(step_pred['driver'])
        rollout_preds['joint'].append(step_pred['joint'])

        # 构建下一步输入（用预测而非真实标签）
        next_emb = decoder.activity_embedding.embed_from_indices(
            step_pred['continuous'],
            step_pred['purpose'].argmax(dim=-1),
            step_pred['mode'].argmax(dim=-1),
            step_pred['driver'].argmax(dim=-1),
            step_pred['joint'].argmax(dim=-1)
        )
        current_input = torch.cat([current_input, next_emb.unsqueeze(2)], dim=2)
        prev_continuous = step_pred['continuous']

    results = {
        'continuous': torch.stack(rollout_preds['continuous'], dim=2),
        'purpose': torch.stack(rollout_preds['purpose'], dim=2),
        'mode': torch.stack(rollout_preds['mode'], dim=2),
        'driver': torch.stack(rollout_preds['driver'], dim=2),
        'joint': torch.stack(rollout_preds['joint'], dim=2)
    }
    if rollout_preds['destination']:
        results['destination'] = torch.stack(rollout_preds['destination'], dim=2)

    return results
