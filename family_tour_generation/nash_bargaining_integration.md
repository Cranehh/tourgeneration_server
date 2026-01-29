# Nash Bargaining 优化层集成指南

## 概述

本文档说明如何将 `nash_bargaining_sqp_optnet.py` 中的 Nash Bargaining 优化博弈层集成到 `train_with_ss_rollout.py` 的训练流程中。

### 核心思想

Nash Bargaining 层作为一个**后处理优化层**，在模型完成输出后、计算损失前，对家庭成员的活动决策进行协调优化，使其满足 Nash 均衡条件。

```
神经网络输出 → Nash Bargaining 协调 → 协调后的输出 → 损失函数 → 反向传播
                     ↑
     通过 OptNet 的 KKT 隐函数实现精确梯度反传
```

---

## 训练流程分析

### 当前训练流程 (`train_with_ss_rollout.py`)

```
1. ExposureBiasTrainer.train_step() (exposure_bias.py:600)
   ├── 编码: encoder → member_repr, family_repr
   ├── 模式预测: pattern_predictor → pattern_probs
   ├── 解码: ss_decoder → predictions
   │     └── 返回格式: {
   │           'continuous': [B, M, T, 2],      # 开始/结束时间
   │           'purpose': [B, M, T, 10],        # 目的 logits
   │           'mode': [B, M, T, 11],           # 方式 logits
   │           'driver': [B, M, T, 2],          # 驾驶状态 logits
   │           'joint': [B, M, T, 2],           # 联合出行 logits
   │           'destination': [B, M, T, 2006],  # 目的地 logits
   │           'end': [B, M, T, 2]              # 结束标记 logits
   │         }
   ├── 损失计算: criterion(predictions, targets, ...)
   └── 返回: loss, losses
```

### 集成点位置

**在 `exposure_bias.py` 的 `ExposureBiasTrainer.train_step()` 方法中，第 629-636 行之后，第 641 行之前：**

```python
# 解码 (使用 Scheduled Sampling)
predictions = self.ss_decoder(...)  # 第 629-636 行

# ========== 在这里插入 Nash Bargaining 层 ==========
# predictions = self.nash_layer(predictions, batch, ...)

# 计算损失
loss, losses = criterion(...)  # 第 641-645 行
```

---

## 集成方案

### 方案 A：在 ExposureBiasTrainer 中集成（推荐）

修改 `exposure_bias.py`，在 `ExposureBiasTrainer` 类中添加 Nash Bargaining 层。

#### 步骤 1：导入和初始化

```python
# exposure_bias.py 顶部添加导入
from nash_bargaining_sqp_optnet import HouseholdNashBargainingLayer, UtilityParams

# 在 ExposureBiasTrainer.__init__() 中添加
class ExposureBiasTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        # ... 其他参数 ...
        use_nash_bargaining: bool = False,  # 新增参数
        nash_config: dict = None,           # 新增参数
    ):
        # ... 原有初始化代码 ...

        # 新增: Nash Bargaining 层
        self.use_nash_bargaining = use_nash_bargaining
        if use_nash_bargaining:
            self.nash_layer = HouseholdNashBargainingLayer(
                num_zones=nash_config.get('num_zones', 2006),
                num_modes=nash_config.get('num_modes', 11),
                max_members=config.max_members,
                max_tours=config.max_activities,
                utility_params=UtilityParams(
                    alpha_joint=nash_config.get('alpha_joint', 0.1),
                    alpha_social=nash_config.get('alpha_social', 0.3),
                    theta_escort=nash_config.get('theta_escort', -0.58),
                ),
                sqp_max_iter=nash_config.get('sqp_max_iter', 15),
                sqp_tol=nash_config.get('sqp_tol', 1e-4),
                enabled=True
            )

            # 加载出行时间矩阵（需要预先准备）
            self.travel_time_matrix = None  # 需要从外部加载
```

#### 步骤 2：修改 train_step 方法

```python
def train_step(
    self,
    batch,
    criterion,
    optimizer,
    current_epoch,
    scaler=None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """单步训练"""
    # ... 原有编码和解码代码 ...

    # 解码 (使用 Scheduled Sampling)
    predictions = self.ss_decoder(
        member_repr, family_repr, batch.activities,
        batch.member_mask, batch.activity_mask,
        training=True,
        pattern_outputs=pattern_prob,
        home_zones=batch.home_zones,
        target_destinations=batch.target_destinations
    )

    # ========== 新增: Nash Bargaining 协调 ==========
    if self.use_nash_bargaining and self.nash_layer is not None:
        predictions = self._apply_nash_bargaining(predictions, batch)
    # ========== 新增结束 ==========

    # 计算损失
    pattern_prob.update({...})
    loss, losses = criterion(...)

    return loss, losses

def _apply_nash_bargaining(
    self,
    predictions: Dict[str, torch.Tensor],
    batch
) -> Dict[str, torch.Tensor]:
    """应用 Nash Bargaining 协调层"""

    # 准备输入
    # 注意：Nash 层需要 departure_time 和 duration，但模型输出是 continuous[..., 0] 和 continuous[..., 1]
    departure_time = predictions['continuous'][..., 0]  # [B, M, T]
    duration = predictions['continuous'][..., 1] - predictions['continuous'][..., 0]  # 计算持续时间

    # 确保 duration 为正
    duration = torch.clamp(duration, min=0.01)

    # 准备 member_is_adult（需要从 batch 或 member_attr 中提取）
    # 假设 member_attr 中第 0 维是年龄或类似指标
    member_is_adult = (batch.member_attr[..., 0] > 18).float()  # 需要根据实际数据调整

    # 调用 Nash 层
    nash_output = self.nash_layer(
        departure_time=departure_time,
        duration=duration,
        destination_logits=predictions['destination'],
        mode_logits=predictions['mode'],
        is_joint_logit=predictions['joint'][..., 1] - predictions['joint'][..., 0],  # 转为单值 logit
        is_driver_logit=predictions['driver'][..., 1] - predictions['driver'][..., 0],
        member_mask=batch.member_mask.float(),
        tour_mask=batch.activity_mask.float(),
        num_vehicles=batch.num_vehicles if batch.num_vehicles is not None else torch.ones(batch.batch_size, device=batch.device),
        home_zone=batch.home_zones,
        member_is_adult=member_is_adult,
        travel_time_matrix=self.travel_time_matrix,
    )

    # 将 Nash 输出转换回原格式
    coordinated_predictions = predictions.copy()

    # 更新 continuous（从 departure + duration 恢复）
    coordinated_continuous = torch.stack([
        nash_output.departure_time,
        nash_output.departure_time + nash_output.duration
    ], dim=-1)
    coordinated_predictions['continuous'] = coordinated_continuous

    # 更新 destination (logits)
    coordinated_predictions['destination'] = nash_output.destination_logits

    # 更新 mode (logits)
    coordinated_predictions['mode'] = nash_output.mode_logits

    # 更新 joint (转回 2-class logits)
    joint_logit = nash_output.is_joint_logit
    coordinated_predictions['joint'] = torch.stack([
        -joint_logit / 2, joint_logit / 2
    ], dim=-1)

    # 更新 driver (转回 2-class logits)
    driver_logit = nash_output.is_driver_logit
    coordinated_predictions['driver'] = torch.stack([
        -driver_logit / 2, driver_logit / 2
    ], dim=-1)

    return coordinated_predictions
```

---

### 方案 B：在模型层集成

修改 `model.py` 中的 `FamilyTourGenerator`，将 Nash 层作为模型的一部分。

```python
# model.py

class FamilyTourGenerator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # ... 原有代码 ...

        # 新增: Nash Bargaining 层
        if getattr(config, 'use_nash_bargaining', False):
            self.nash_layer = HouseholdNashBargainingLayer(
                num_zones=2006,
                num_modes=config.num_modes,
                max_members=config.max_members,
                max_tours=config.max_activities,
            )
        else:
            self.nash_layer = None

    def forward(self, batch, teacher_forcing=True):
        # ... 原有编码和解码代码 ...
        predictions = self.decoder(...)

        # Nash 协调
        if self.nash_layer is not None and self.training:
            predictions = self._apply_nash_coordination(predictions, batch)

        return predictions, pattern_probs
```

---

## 数据准备

### 需要额外准备的数据

1. **出行时间矩阵 `travel_time_matrix`**
   - 形状: `[num_zones, num_zones, num_modes]` 或 `[batch, num_zones, num_zones, num_modes]`
   - 含义: 不同方式下从 zone i 到 zone j 的出行时间
   - 加载方式:
   ```python
   travel_time_matrix = np.load('path/to/travel_time_matrix.npy')
   self.travel_time_matrix = torch.from_numpy(travel_time_matrix).to(device)
   ```

2. **成员是否成年 `member_is_adult`**
   - 形状: `[batch, max_members]`
   - 可以从 `member_attr` 中提取年龄信息判断

3. **车辆数 `num_vehicles`**
   - 形状: `[batch]`
   - 如果 `FamilyTourBatch` 中没有，需要添加

---

## 变量映射说明

### 模型输出 → Nash 层输入

| 模型输出 | Nash 层输入 | 转换方式 |
|---------|------------|---------|
| `predictions['continuous'][..., 0]` | `departure_time` | 直接使用 |
| `predictions['continuous'][..., 1] - [..., 0]` | `duration` | 计算差值 |
| `predictions['destination']` | `destination_logits` | 直接使用 |
| `predictions['mode']` | `mode_logits` | 直接使用 |
| `predictions['joint'][..., 1] - [..., 0]` | `is_joint_logit` | 转为单值 logit |
| `predictions['driver'][..., 1] - [..., 0]` | `is_driver_logit` | 转为单值 logit |
| `batch.member_mask` | `member_mask` | `.float()` |
| `batch.activity_mask` | `tour_mask` | `.float()` |

### Nash 层输出 → 模型格式

| Nash 输出 | 模型格式 | 转换方式 |
|----------|---------|---------|
| `nash_output.departure_time` | `continuous[..., 0]` | 直接使用 |
| `nash_output.departure_time + duration` | `continuous[..., 1]` | 相加 |
| `nash_output.destination_logits` | `predictions['destination']` | 直接使用 |
| `nash_output.mode_logits` | `predictions['mode']` | 直接使用 |
| `nash_output.is_joint_logit` | `predictions['joint']` | 转为 2-class logits |
| `nash_output.is_driver_logit` | `predictions['driver']` | 转为 2-class logits |

---

## 配置示例

### 在 `config.py` 中添加配置

```python
@dataclass
class ModelConfig:
    # ... 原有配置 ...

    # Nash Bargaining 配置
    use_nash_bargaining: bool = False
    nash_config: dict = field(default_factory=lambda: {
        'num_zones': 2006,
        'num_modes': 11,
        'alpha_joint': 0.1,
        'alpha_social': 0.3,
        'theta_escort': -0.58,
        'theta_car': -1.0,
        'theta_pt': -0.4,
        'sqp_max_iter': 15,
        'sqp_tol': 1e-4,
    })
```

### 在训练脚本中启用

```python
# train_with_ss_rollout.py 的 main() 函数中

trainer = ScheduledSamplingTrainer(
    model=model,
    model_config=model_config,
    train_config=train_config,
    train_loader=train_loader,
    val_loader=val_loader,
    save_dir='../checkpoints_ss_with_nash',
    eb_strategy='aggressive',
    # 新增参数
    use_nash_bargaining=True,
    nash_config=model_config.nash_config,
)
```

---

## 注意事项

1. **计算开销**：Nash Bargaining 层包含 SQP 迭代，会增加训练时间。建议：
   - 初期训练时禁用（`enabled=False`）
   - 训练稳定后再启用
   - 或者只在部分 epoch 启用

2. **梯度流动**：OptNet 通过 KKT 条件提供精确梯度，但：
   - 如果 `qpth` 库不可用，会退化为近似梯度
   - 确保安装: `pip install qpth`

3. **数值稳定性**：
   - Nash 层内部有 clamp 和 softmax 保护
   - 如果出现 NaN，检查 `travel_time_matrix` 是否有效

4. **验证时的处理**：
   - 验证时建议禁用 Nash 层，或设置 `enabled=False`
   - 因为验证使用 `model.generate()`，不经过 `ExposureBiasTrainer`

---

## 文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `family_tour_generation/exposure_bias.py` | 添加 Nash 层初始化和 `_apply_nash_bargaining` 方法 |
| `family_tour_generation/config.py` | 添加 Nash 相关配置 |
| `family_tour_generation/data.py` | 确保 `FamilyTourBatch` 包含 `num_vehicles` |
| `family_tour_generation/train_with_ss_rollout.py` | 传递 Nash 配置参数 |
| `nash_bargaining_sqp_optnet.py` | 复制到 `family_tour_generation/` 目录 |

---

## 完整代码示例

详见 `exposure_bias_with_nash.py`（需要新建）或直接修改 `exposure_bias.py`。
