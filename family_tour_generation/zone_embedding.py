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
        lda = self.zone_lda[:self.num_zones]                 # [num_zones, n_topics]
        affordance = self.zone_affordance[:self.num_zones]   # [num_zones, n_purposes]

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