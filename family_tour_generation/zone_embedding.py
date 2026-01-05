"""
Zone嵌入和空间特征模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ZoneEmbeddingModule(nn.Module):
    """Zone嵌入模块"""

    def __init__(self, config):
        super().__init__()
        self.num_zones = config.num_zones
        self.zone_none = config.zone_none
        self.zone_padding = config.zone_padding
        self.vocab_size = config.num_zones + 2  # 真实zone + NONE + PADDING

        self.zone_embed_dim = config.zone_embed_dim
        self.n_lda_topics = config.n_lda_topics
        self.n_affordance_purposes = config.n_affordance_purposes
        self.d_model = config.d_model

        # 可学习zone嵌入
        self.zone_embed = nn.Embedding(self.vocab_size, config.zone_embed_dim)
        self._init_embeddings()

        # 外部特征（buffer，不参与梯度）
        self.register_buffer('zone_lda', torch.zeros(self.vocab_size, config.n_lda_topics))
        self.register_buffer('zone_affordance', torch.zeros(self.vocab_size, config.n_affordance_purposes))
        self.register_buffer('zone_impedance', torch.zeros(config.num_zones, config.num_zones))

        # zone特征投影到d_model
        total_feat_dim = config.zone_embed_dim + config.n_lda_topics + config.n_affordance_purposes
        self.zone_proj = nn.Sequential(
            nn.Linear(total_feat_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )

    def _init_embeddings(self):
        nn.init.normal_(self.zone_embed.weight[:self.num_zones], mean=0, std=0.02)
        nn.init.zeros_(self.zone_embed.weight[self.zone_none])
        nn.init.zeros_(self.zone_embed.weight[self.zone_padding])

    def load_external_features(self, lda_matrix, affordance_matrix, impedance_matrix):
        """
        加载预计算特征
        Args:
            lda_matrix: [num_zones, n_topics]
            affordance_matrix: [num_zones, n_purposes]
            impedance_matrix: [num_zones, num_zones]
        """
        device = self.zone_embed.weight.device

        self.zone_lda[:self.num_zones] = torch.from_numpy(lda_matrix).float().to(device)
        self.zone_affordance[:self.num_zones] = torch.from_numpy(affordance_matrix).float().to(device)

        # 阻抗：log变换 + 标准化
        # imp = np.log1p(impedance_matrix)
        # imp = (imp - imp.mean()) / (imp.std() + 1e-6)
        self.zone_impedance = torch.from_numpy(impedance_matrix).float().to(device)

    def get_zone_embedding(self, zone_ids):
        """获取zone的基础嵌入"""
        return self.zone_embed(zone_ids)

    def get_all_zone_keys(self):
        """
        获取所有真实zone的融合特征（用于attention的Key）
        Returns: [num_zones, d_model]
        """
        zone_ids = torch.arange(self.num_zones, device=self.zone_embed.weight.device)

        embed = self.zone_embed(zone_ids)                    # [num_zones, embed_dim]
        a = torch.isnan(embed).sum()
        lda = self.zone_lda[:self.num_zones]                 # [num_zones, n_topics]
        b = torch.isnan(lda).sum()
        affordance = self.zone_affordance[:self.num_zones]   # [num_zones, n_purposes]
        c = torch.isnan(affordance).sum()
        combined = torch.cat([embed, lda, affordance], dim=-1)
        return self.zone_proj(combined)  # [num_zones, d_model]

    def get_impedance(self, origin_zones):
        """
        获取从origin到所有zone的阻抗
        Args:
            origin_zones: [...] origin zone IDs
        Returns:
            [..., num_zones]
        """
        # 处理特殊zone
        valid_mask = origin_zones < self.num_zones
        safe_origins = origin_zones.clamp(0, self.num_zones - 1)

        # 查表
        original_shape = origin_zones.shape
        impedance = self.zone_impedance[safe_origins.view(-1)]
        impedance = impedance.view(*original_shape, self.num_zones)

        # 无效origin的阻抗设为0
        impedance = impedance * valid_mask.unsqueeze(-1).float()

        return impedance

    def get_affordance_scores(self, purpose_probs):
        """
        根据活动类型概率获取各zone的affordance分数
        Args:
            purpose_probs: [..., num_purposes]
        Returns:
            [..., num_zones]
        """
        affordance = self.zone_affordance[:self.num_zones]  # [num_zones, n_purposes]

        # 截取到affordance的维度
        n_aff = min(purpose_probs.size(-1), self.n_affordance_purposes)
        probs = purpose_probs[..., :n_aff]
        aff = affordance[:, :n_aff]

        # 加权求和: [..., n_purposes] x [num_zones, n_purposes]^T -> [..., num_zones]
        scores = torch.einsum('...p,np->...n', probs, aff)

        return scores


class FamilyMemberZoneEmbedding(nn.Module):
    """
    家庭和成员的位置嵌入模块

    - 家庭位置（home_zone）: 融入家庭特征
    - 成员工作位置（work_zone）: 融入成员特征
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        num_zones = config.num_zones
        embed_dim = config.zone_embed_dim
        n_lda = config.n_lda_topics
        n_affordance = config.n_affordance_purposes

        self.vocab_size = num_zones + 2  # 0-2005: 真实zone, 2006: NONE, 2007: PADDING
        self.zone_none = num_zones
        self.zone_padding = num_zones + 1

        # ===== 共享的zone基础嵌入 =====
        self.zone_embed = nn.Embedding(self.vocab_size, embed_dim)
        self._init_embeddings()

        # ===== 外部特征 =====
        self.register_buffer('zone_lda', torch.zeros(self.vocab_size, n_lda))
        self.register_buffer('zone_affordance', torch.zeros(self.vocab_size, n_affordance))

        # 外部特征投影
        self.external_proj = nn.Sequential(
            nn.Linear(n_lda + n_affordance, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # ===== 家庭位置专用处理 =====
        self.home_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # ===== 工作位置专用处理 =====
        self.work_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 无工作的特殊嵌入（可学习）
        self.no_work_embed = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.no_work_embed, std=0.02)

    def _init_embeddings(self):
        nn.init.normal_(self.zone_embed.weight[:self.config.num_zones], mean=0, std=0.02)
        nn.init.zeros_(self.zone_embed.weight[self.zone_none])
        nn.init.zeros_(self.zone_embed.weight[self.zone_padding])

    def load_external_features(self, lda_matrix, affordance_matrix):
        """加载外部特征"""
        device = self.zone_embed.weight.device
        self.zone_lda[:self.config.num_zones] = torch.from_numpy(lda_matrix).float().to(device)
        self.zone_affordance[:self.config.num_zones] = torch.from_numpy(affordance_matrix).float().to(device)

    def _get_zone_features(self, zone_ids):
        """获取zone的完整特征（嵌入 + 外部特征）"""
        embed = self.zone_embed(zone_ids)
        lda = self.zone_lda[zone_ids]
        aff = self.zone_affordance[zone_ids]
        external = self.external_proj(torch.cat([lda, aff], dim=-1))
        return torch.cat([embed, external], dim=-1)  # [*, embed_dim * 2]

    def get_home_embedding(self, home_zones):
        """
        获取家庭位置嵌入

        Args:
            home_zones: [B] 家庭住址zone ID
        Returns:
            [B, embed_dim] 家庭位置嵌入
        """
        features = self._get_zone_features(home_zones)  # [B, embed_dim * 2]
        return self.home_proj(features)  # [B, embed_dim]

    def get_work_embedding(self, work_zones):
        """
        获取工作位置嵌入

        Args:
            work_zones: [B, M] 成员工作地zone ID (2006表示无工作, 2007表示padding成员)
        Returns:
            [B, M, embed_dim] 工作位置嵌入
        """
        # 识别有效工作位置
        has_work = work_zones < self.zone_none  # [B, M]

        # 获取所有位置的特征
        features = self._get_zone_features(work_zones)  # [B, M, embed_dim * 2]
        work_embed = self.work_proj(features)  # [B, M, embed_dim]

        # 对于无工作的成员，使用特殊嵌入
        no_work = self.no_work_embed.expand_as(work_embed)
        work_embed = torch.where(has_work.unsqueeze(-1), work_embed, no_work)

        return work_embed  # [B, M, embed_dim]


class ActivityZoneEmbedding(nn.Module):
    """
    活动起终点位置嵌入模块

    - origin_zone: 活动出发地
    - destination_zone: 活动目的地

    注意：origin由上一个活动的destination决定（第一个活动的origin是home）
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        num_zones = config.num_zones
        embed_dim = config.zone_embed_dim
        n_lda = config.n_lda_topics
        n_affordance = config.n_affordance_purposes

        self.vocab_size = num_zones + 1
        self.zone_none = num_zones

        # ===== 共享的zone基础嵌入 =====
        self.zone_embed = nn.Embedding(self.vocab_size, embed_dim)
        self._init_embeddings()

        # ===== 外部特征 =====
        self.register_buffer('zone_lda', torch.zeros(self.vocab_size, n_lda))
        self.register_buffer('zone_affordance', torch.zeros(self.vocab_size, n_affordance))

        # 外部特征投影
        self.external_proj = nn.Sequential(
            nn.Linear(n_lda + n_affordance, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # ===== Origin位置处理 =====
        self.origin_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # ===== Destination位置处理 =====
        self.destination_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 无效位置的特殊嵌入（padding活动）
        self.invalid_origin_embed = nn.Parameter(torch.zeros(embed_dim))
        self.invalid_dest_embed = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.invalid_origin_embed, std=0.02)
        nn.init.normal_(self.invalid_dest_embed, std=0.02)

    def _init_embeddings(self):
        nn.init.normal_(self.zone_embed.weight[:self.config.num_zones], mean=0, std=0.02)
        nn.init.zeros_(self.zone_embed.weight[self.zone_none])

    def load_external_features(self, lda_matrix, affordance_matrix):
        """加载外部特征"""
        device = self.zone_embed.weight.device
        self.zone_lda[:self.config.num_zones] = torch.from_numpy(lda_matrix).float().to(device)
        self.zone_affordance[:self.config.num_zones] = torch.from_numpy(affordance_matrix).float().to(device)

    def _get_zone_features(self, zone_ids):
        """获取zone的完整特征"""
        # 处理可能的越界
        safe_ids = zone_ids.clamp(0, self.vocab_size - 1)
        embed = self.zone_embed(safe_ids)
        lda = self.zone_lda[safe_ids]
        aff = self.zone_affordance[safe_ids]
        external = self.external_proj(torch.cat([lda, aff], dim=-1))
        return torch.cat([embed, external], dim=-1)

    def get_origin_embedding(self, origin_zones):
        """
        获取出发地嵌入

        Args:
            origin_zones: [B, M, T] 或 [B, M] 出发地zone ID
        Returns:
            同形状 + embed_dim 的嵌入
        """
        valid_mask = origin_zones < self.zone_none

        features = self._get_zone_features(origin_zones)
        origin_embed = self.origin_proj(features)

        # 无效位置使用特殊嵌入
        invalid_embed = self.invalid_origin_embed.expand_as(origin_embed)
        origin_embed = torch.where(valid_mask.unsqueeze(-1), origin_embed, invalid_embed)

        return origin_embed

    def get_destination_embedding(self, dest_zones):
        """
        获取目的地嵌入

        Args:
            dest_zones: [B, M, T] 或 [B, M] 目的地zone ID
        Returns:
            同形状 + embed_dim 的嵌入
        """
        valid_mask = dest_zones < self.zone_none

        features = self._get_zone_features(dest_zones)
        dest_embed = self.destination_proj(features)

        # 无效位置使用特殊嵌入
        invalid_embed = self.invalid_dest_embed.expand_as(dest_embed)
        dest_embed = torch.where(valid_mask.unsqueeze(-1), dest_embed, invalid_embed)

        return dest_embed

    def forward(self, origin_zones, dest_zones):
        """
        获取起终点嵌入

        Args:
            origin_zones: [B, M, T] 出发地
            dest_zones: [B, M, T] 目的地
        Returns:
            origin_embed: [B, M, T, embed_dim]
            dest_embed: [B, M, T, embed_dim]
        """
        origin_embed = self.get_origin_embedding(origin_zones)
        dest_embed = self.get_destination_embedding(dest_zones)
        return origin_embed, dest_embed


class DistanceAwareModeHead(nn.Module):
    """
    距离感知的方式预测头

    改进：
    1. 支持软加权期望距离
    2. 引入距离-方式先验偏置（从数据统计初始化）
    3. 多特征距离编码
    """

    def __init__(
            self,
            d_model: int,
            num_modes: int = 11,
            num_distance_bins: int = 10,
            dropout: float = 0.1
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_distance_bins = num_distance_bins

        # 距离分箱边界（z-score标准化后）
        self.register_buffer(
            'distance_bins',
            torch.linspace(-1.6, 4, num_distance_bins + 1)
        )

        # ===== 多特征距离编码 =====
        # 输入：标量距离 + 分箱soft embedding
        self.distance_encoder = nn.Sequential(
            nn.Linear(1 + num_distance_bins, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)
        )

        # 融合层
        fusion_dim = d_model + d_model // 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_modes)
        )

        # ===== 思路3：距离-方式先验偏置 =====
        # 形状：(num_distance_bins, num_modes)
        # 初始化为0，可以从数据统计后手动设置
        self.distance_mode_prior = nn.Parameter(
            torch.zeros(num_distance_bins, num_modes)
        )

        # 先验偏置的缩放系数（可学习）
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

    def get_soft_bin_weights(self, distance: torch.Tensor) -> torch.Tensor:
        """
        计算距离属于每个bin的软权重（高斯核）

        Args:
            distance: (...) 标准化后的距离

        Returns:
            weights: (..., num_distance_bins) 软分箱权重
        """
        # 计算每个bin的中心
        bin_centers = (self.distance_bins[:-1] + self.distance_bins[1:]) / 2  # (num_bins,)
        bin_width = self.distance_bins[1] - self.distance_bins[0]

        # 计算到每个中心的距离
        dist_expanded = distance.unsqueeze(-1)  # (..., 1)
        diff = (dist_expanded - bin_centers) / (bin_width / 2)  # (..., num_bins)

        # 高斯核
        weights = torch.exp(-0.5 * diff ** 2)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化

        return weights

    def get_distance_bin_index(self, distance: torch.Tensor) -> torch.Tensor:
        """获取硬分箱索引"""
        bin_idx = torch.bucketize(distance, self.distance_bins[1:-1])
        return bin_idx.clamp(0, self.num_distance_bins - 1)

    def forward(
            self,
            hidden: torch.Tensor,  # (..., d_model)
            distance: torch.Tensor  # (...) 标准化后的距离
    ) -> torch.Tensor:
        """
        Args:
            hidden: decoder输出的隐状态
            distance: z-score标准化后的出行距离

        Returns:
            mode_logits: (..., num_modes)
        """
        # ===== 多特征距离编码 =====
        # 1. 标量距离
        dist_scalar = distance.unsqueeze(-1)  # (..., 1)

        # 2. 软分箱权重
        soft_bin_weights = self.get_soft_bin_weights(distance)  # (..., num_bins)

        # 3. 拼接距离特征
        dist_features = torch.cat([dist_scalar, soft_bin_weights], dim=-1)  # (..., 1+num_bins)
        dist_encoded = self.distance_encoder(dist_features)  # (..., d_model//4)

        # ===== 融合并预测 =====
        fused = torch.cat([hidden, dist_encoded], dim=-1)
        mode_logits = self.fusion(fused)

        # ===== 思路3：加入先验偏置 =====
        # 用软分箱权重加权先验
        # soft_bin_weights: (..., num_bins)
        # distance_mode_prior: (num_bins, num_modes)
        prior_bias = torch.einsum('...b,bm->...m', soft_bin_weights, self.distance_mode_prior)
        mode_logits = mode_logits + self.prior_scale * prior_bias

        return mode_logits

    def init_prior_from_data(self, distance_mode_counts: torch.Tensor):
        """
        从数据统计初始化先验偏置

        Args:
            distance_mode_counts: (num_distance_bins, num_modes)
                                  每个距离区间各方式的出行次数
        """
        # 转换为log概率作为先验
        counts = distance_mode_counts.float() + 1  # 平滑
        probs = counts / counts.sum(dim=-1, keepdim=True)
        log_probs = torch.log(probs)

        # 减去均值，使其成为偏置
        log_probs = log_probs - log_probs.mean(dim=-1, keepdim=True)

        with torch.no_grad():
            self.distance_mode_prior.copy_(log_probs)