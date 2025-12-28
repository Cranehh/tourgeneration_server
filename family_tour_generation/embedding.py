"""
活动嵌入层
将活动的各个属性嵌入到统一的向量空间
"""
import torch
import torch.nn as nn
from config import ModelConfig


class ActivityEmbedding(nn.Module):
    """
    活动嵌入层
    
    输入活动维度: 27 = 2(连续) + 10(目的) + 11(方式) + 2(驾驶状态) + 2(联合出行)
    输出: (*, d_model)
    """
    
    def __init__(self, config: ModelConfig, zone_embedding_module: nn.Module = None):
        super().__init__()
        self.config = config
        self.zone_embedding_module = zone_embedding_module  # ← 新增
        self.use_zone_embedding = zone_embedding_module is not None  # ← 新增
        
        # 连续属性投影 (开始时间z-score, 结束时间z-score)
        self.continuous_proj = nn.Sequential(
            nn.Linear(config.continuous_dim, config.d_emb),
            nn.ReLU(),
            nn.Linear(config.d_emb, config.d_emb)
        )
        
        # 离散属性嵌入
        self.purpose_emb = nn.Embedding(config.num_purposes, config.d_emb)
        self.mode_emb = nn.Embedding(config.num_modes, config.d_emb)
        self.driver_emb = nn.Embedding(config.num_driver_status, config.d_emb)
        self.joint_emb = nn.Embedding(config.num_joint_status, config.d_emb)

        if self.use_zone_embedding:
            fusion_input_dim = 5 * config.d_emb + 2 * config.zone_embed_dim
        else:
            fusion_input_dim = 5 * config.d_emb

        # 融合投影: 5 * d_emb -> d_model
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )
    
    def forward(self, activities: torch.Tensor, origin_zones=None, dest_zones=None) -> torch.Tensor:
        """
        Args:
            activities: (*, 27) 活动特征
                - [:, 0:2]: 连续属性 (开始时间, 结束时间的z-score)
                - [:, 2:12]: 目的 one-hot (10维)
                - [:, 12:23]: 方式 one-hot (11维)
                - [:, 23:25]: 驾驶状态 one-hot (2维)
                - [:, 25:27]: 联合出行 one-hot (2维)
        
        Returns:
            (*, d_model) 活动嵌入
        """
        # 保存原始形状
        original_shape = activities.shape[:-1]
        # activities = activities.reshape(-1, self.config.activity_dim)

        # ===== 现有代码：处理各属性 =====
        # 连续属性
        continuous = activities[..., :2]
        continuous_emb = self.continuous_proj(continuous)

        # 离散属性（从one-hot转为index）
        purpose_idx = activities[..., 2:12].argmax(dim=-1)
        mode_idx = activities[..., 12:23].argmax(dim=-1)
        driver_idx = activities[..., 23:25].argmax(dim=-1)
        joint_idx = activities[..., 25:27].argmax(dim=-1)

        purpose_emb = self.purpose_emb(purpose_idx)
        mode_emb = self.mode_emb(mode_idx)
        driver_emb = self.driver_emb(driver_idx)
        joint_emb = self.joint_emb(joint_idx)

        # ===== 新增：处理起终点嵌入 =====
        if self.use_zone_embedding:
            # 如果没有显式传入，从activities中提取
            if origin_zones is None:
                origin_zones = activities[..., -2].long()
            if dest_zones is None:
                dest_zones = activities[..., -1].long()

            origin_emb = self.zone_embedding_module.get_origin_embedding(origin_zones)
            dest_emb = self.zone_embedding_module.get_destination_embedding(dest_zones)

            # 融合所有嵌入
            combined = torch.cat([
                continuous_emb, purpose_emb, mode_emb, driver_emb, joint_emb,
                origin_emb, dest_emb
            ], dim=-1)
        else:
            combined = torch.cat([
                continuous_emb, purpose_emb, mode_emb, driver_emb, joint_emb
            ], dim=-1)
        
        return self.fusion_proj(combined)
    
    def embed_from_indices(
        self,
        continuous: torch.Tensor,
        purpose_idx: torch.Tensor,
        mode_idx: torch.Tensor,
        driver_idx: torch.Tensor,
        joint_idx: torch.Tensor,
        origin_zones: torch.Tensor = None,  # ← 新增
        dest_zones: torch.Tensor = None  # ← 新增
    ) -> torch.Tensor:
        """
        从预测的索引构建嵌入（用于自回归生成）
        
        Args:
            continuous: (*, 2) 连续属性
            purpose_idx: (*,) 目的索引
            mode_idx: (*,) 方式索引
            driver_idx: (*,) 驾驶状态索引
            joint_idx: (*,) 联合出行索引
        
        Returns:
            (*, d_model) 活动嵌入
        """
        continuous_emb = self.continuous_proj(continuous)
        purpose_emb = self.purpose_emb(purpose_idx)
        mode_emb = self.mode_emb(mode_idx)
        driver_emb = self.driver_emb(driver_idx)
        joint_emb = self.joint_emb(joint_idx)

        if self.use_zone_embedding and origin_zones is not None and dest_zones is not None:
            origin_emb = self.zone_embedding_module.get_origin_embedding(origin_zones)
            dest_emb = self.zone_embedding_module.get_destination_embedding(dest_zones)

            combined = torch.cat([
                continuous_emb, purpose_emb, mode_emb, driver_emb, joint_emb,
                origin_emb, dest_emb
            ], dim=-1)
        else:
            combined = torch.cat([
                continuous_emb, purpose_emb, mode_emb, driver_emb, joint_emb
            ], dim=-1)

        return self.fusion_proj(combined)


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
