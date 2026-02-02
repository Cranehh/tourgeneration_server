"""
配置类定义
"""
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """模型配置"""
    # 输入维度
    family_dim: int = 32          # 家庭特征维度 Ff
    member_dim: int = 48          # 成员特征维度 Fm
    activity_dim: int = 27        # 活动特征维度 Fa (2连续 + 10 + 11 + 2 + 2 one-hot)
    num_family_patterns: int = 20  # 家庭出行模式类别数
    num_individual_patterns: int = 40  # 个人出行模式类别数
    
    # 活动属性配置
    continuous_dim: int = 2       # 连续属性维度 (开始时间z-score, 结束时间z-score)
    num_purposes: int = 10        # 出行目的类别数
    num_modes: int = 11           # 出行方式类别数
    num_driver_status: int = 2    # 驾驶员/乘客
    num_joint_status: int = 2     # 是否联合出行
    
    # 序列配置
    max_members: int = 8          # 最大家庭成员数
    max_activities: int = 6       # 最大活动链长度
    
    # 模型维度
    d_model: int = 256            # 隐藏层维度
    d_emb: int = 64               # 各属性嵌入维度
    num_heads: int = 8            # 注意力头数
    num_decoder_layers: int = 20  # Decoder层数
    d_ff: int = 1024              # FFN中间维度
    dropout: float = 0.1
    
    # PLE配置
    num_inducing_points: int = 16  # Set Transformer诱导点数量

    # ===== 新增：PLE CGC配置 =====
    num_shared_experts: int = 2  # 共享专家数量
    num_task_experts: int = 1  # 每个任务的特定专家数量
    num_cgc_layers: int = 3  # CGC层数量
    
    # 损失权重
    loss_weights: dict = None

    # 使用模式专家
    use_pattern_condition = False
    use_pattern_condition_in_decoder = True

    # ===== 新增：目的地预测配置 =====
    use_destination_prediction: bool = True
    num_zones: int = 2006  # TAZ数量
    zone_embed_dim: int = 64  # zone嵌入维度
    n_lda_topics: int = 12  # LDA主题数
    n_affordance_purposes: int = 5  # affordance维度（与num_purposes一致）

    # 特殊zone ID
    zone_none: int = 2006  # 无效zone
    zone_padding: int = 2007  # padding

    # ===== 新增：距离感知方式预测配置 =====
    use_distance_aware_mode: bool = True  # 是否启用距离感知方式预测

    # ===== 新增：Gumbel-Softmax 配置 =====
    gumbel_temperature: float = 1.0  # Gumbel-Softmax 温度，训练时可逐渐降低
    use_soft_distance: bool = True  # 是否使用期望距离（软加权）

    # Nash Bargaining 配置
    use_nash_bargaining: bool = True
    nash_config: dict = field(default_factory=lambda: {
        'num_zones': 2006,
        'num_modes': 11,
        'alpha_joint': 0.1,
        'alpha_social': 0.3,
        'theta_escort': -0.58,
        'theta_car': -1.0,
        'theta_pt': -0.4,
        'sqp_max_iter': 150,
        'sqp_tol': 1e-4,
    })



    
    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {
                'continuous': 1.0,
                'purpose': 1.0,
                'mode': 1.0,
                'driver': 1,
                'joint': 1,
                'pattern': 0.2,
                'destination': 1.0,  # 新增\
                'coord_joint_consistency': 0.5,
                'coord_vehicle_constraint': 0.5,
                'coord_home_constraint': 0.5,
                'end_token': 1.0
            }


@dataclass 
class TrainConfig:
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    
    # Focal Loss参数
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # 设备
    device: str = 'cuda'
    # ===== 新增：联合活动损失 =====
    lambda_joint: float = 1.0
    lambda_vehicle: float = 2.0
    lambda_home: float = 1.0
    use_coordination_loss: bool = True
    coordination_weight: float = 0.5