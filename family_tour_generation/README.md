# 活动模式蒸馏模型

## 概述

本模块实现了一个知识蒸馏框架，用于训练一个只需人口属性就能预测活动模式的学生模型。

### 核心思路

1. **教师模型**：从完整的活动序列识别模式（能看到"答案"）
2. **学生模型**：从人口属性预测模式（只能看到输入特征）
3. **蒸馏**：让学生模仿教师的表征和输出

### 为什么用蒸馏而非直接训练？

直接训练的问题：
- GMM产生的hard label信息量有限
- 学生只能从0/1标签学习，不知道"模式A和模式B很相似"

蒸馏的优势：
- 教师的中间表征包含丰富的模式理解
- 教师的软输出保留了模式间的相似关系
- 学生可以从这些软信号中学到更多

## 文件结构

```
pattern_distillation/
├── pattern_distillation.py    # 核心模型定义
│   ├── PatternDistillConfig   # 配置类
│   ├── PatternTeacher         # 教师模型
│   ├── PatternStudent         # 学生模型
│   ├── DistillationLoss       # 蒸馏损失
│   ├── PatternDistillDataset  # 数据集
│   ├── TeacherTrainer         # 教师训练器
│   └── PatternDistillTrainer  # 蒸馏训练器
│
├── train_distillation.py      # 训练脚本
│   ├── train_teacher()        # 阶段1：训练教师
│   ├── train_student()        # 阶段2：蒸馏学生
│   └── evaluate_models()      # 评估两个模型
│
├── integration.py             # 主模型集成工具
│   ├── PatternPredictorAdapter      # 学生模型适配器
│   ├── IntegratedPatternPredictor   # 完整集成预测器
│   └── PatternConditionInjector     # 条件注入模块
│
└── README.md                  # 本文档
```

## 使用方法

### 阶段1：训练教师模型

```bash
python train_distillation.py --stage teacher \
    --data_dir ../数据 \
    --teacher_dir ./checkpoints_teacher \
    --teacher_epochs 500 \
    --batch_size 256 \
    --device cuda
```

或在Python中：

```python
from train_distillation import train_teacher

trainer = train_teacher(
    data_dir="../数据",
    save_dir="./checkpoints_teacher",
    num_epochs=50,
    batch_size=128
)
```

**验收标准**：教师模型在验证集上的准确率应该 >90%。如果达不到，说明活动序列→模式的映射本身存在问题。

### 阶段2：蒸馏训练学生

```bash
python train_distillation.py --stage student \
    --data_dir ../数据 \
    --teacher_dir ./checkpoints_teacher \
    --student_dir ./checkpoints_student \
    --student_epochs 500 \
    --temperature 3.0 \
    --device cuda
```

或在Python中：

```python
from train_distillation import train_student

trainer = train_student(
    teacher_checkpoint="./checkpoints_teacher/best_teacher.pt",
    data_dir="../数据",
    save_dir="./checkpoints_student",
    num_epochs=100,
    temperature=3.0
)
```

### 评估模型

```bash
python train_distillation.py --stage eval \
    --teacher_dir ./checkpoints_teacher \
    --student_dir ./checkpoints_student \
    --data_dir ../数据
```

### 集成到主模型

```python
from integration import IntegratedPatternPredictor, PatternConditionInjector

# 创建集成的Pattern Predictor
pattern_predictor = IntegratedPatternPredictor(
    student_checkpoint_path="./checkpoints_student/best_student.pt",
    target_d_model=256,  # 主模型的d_model
    freeze_student=True   # 建议先冻结，后期可以解冻微调
)

# 在主模型中使用
outputs = pattern_predictor(family_attr, member_attr, member_mask)

# 获取可用于Decoder的条件embedding
pattern_condition = outputs['pattern_condition']  # [B, M, d_model]
family_condition = outputs['family_pattern_condition']  # [B, d_model]

# 获取模式概率分布（可选，用于监督）
family_prob = outputs['family_pattern_prob']  # [B, 20]
person_prob = outputs['person_pattern_prob']  # [B, M, 40]
```

## 配置参数说明

### PatternDistillConfig

```python
@dataclass
class PatternDistillConfig:
    # 数据维度
    activity_dim: int = 27          # 活动特征维度
    max_activities: int = 6         # 最大活动数
    max_members: int = 8            # 最大家庭成员数
    family_dim: int = 10            # 家庭属性维度
    member_dim: int = 51            # 成员属性维度
    
    # 模式数量（与GMM一致）
    num_family_patterns: int = 20   # 家庭模式数
    num_person_patterns: int = 40   # 个人模式数
    
    # 模型结构
    d_model: int = 256              # 隐藏层维度
    d_emb: int = 128                # embedding维度
    num_heads: int = 8              # 注意力头数
    num_encoder_layers: int = 3     # Transformer层数
    dropout: float = 0.1
    
    # 蒸馏参数（重要！）
    temperature: float = 3.0        # 蒸馏温度，越高越软化
    alpha_soft: float = 0.7         # 软标签损失权重
    alpha_repr: float = 0.2         # 表征蒸馏权重
    alpha_hard: float = 0.1         # 硬标签损失权重
```

### 蒸馏参数调整建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| temperature | 2.0~4.0 | 初期用高温(3-4)学习模式关系，后期可降低 |
| alpha_soft | 0.5~0.8 | 主要损失，确保学生学会教师的分布 |
| alpha_repr | 0.1~0.3 | 表征对齐，帮助学习中间表示 |
| alpha_hard | 0.1~0.3 | 保证最终分类准确率 |

## 数据格式要求

### 输入数据

| 数据 | 形状 | 说明 |
|------|------|------|
| family_data | [N, 11] | 家庭属性（最后一列是home_zone） |
| member_data | [N, M, 51] | 成员属性 |
| activity_data | [N, M, T, 29] | 活动序列（前27维是特征，后2维是位置） |
| member_mask | [N, M] | 有效成员mask |
| activity_mask | [N, M, T] | 有效活动mask |
| family_pattern | [N, 20] | GMM家庭模式概率分布 |
| person_pattern | [N, M, 40] | GMM个人模式概率分布 |

### 活动特征结构 (27维)

```
[0:2]   时间特征：开始时间, 结束时间（归一化）
[2:12]  目的 one-hot (10类)
[12:23] 方式 one-hot (11类)
[23:25] 驾驶员/乘客 one-hot (2类)
[25:27] 是否联合出行 one-hot (2类)
```

## 模型架构

### 教师模型 (PatternTeacher)

```
活动序列 [B, M, T, 27]
    │
    ▼
ActivityEmbedding (各特征分别embedding后融合)
    │
    ▼
PersonEncoder (Transformer，编码每个成员的活动序列)
    │
    ▼
个人表征 [B, M, d_model] ──────► 个人模式分类头 ──► [B, M, 40]
    │
    ▼
FamilyEncoder (Transformer，成员间交互)
    │
    ▼
家庭表征 [B, d_model] ──────► 家庭模式分类头 ──► [B, 20]
```

### 学生模型 (PatternStudent)

```
家庭属性 [B, 10]          成员属性 [B, M, 51]
    │                         │
    ▼                         ▼
FamilyEncoder              PersonEncoder
    │                         │
    └────► Cross-Attention ◄──┘
                │
                ▼
        MemberInteraction (Transformer)
                │
                ▼
        个人表征 [B, M, d_model] ──► 个人模式分类头
                │
                ▼
        家庭表征 [B, d_model] ──► 家庭模式分类头
```

### 蒸馏损失

```
L_total = α_repr * L_repr + α_soft * L_soft + α_hard * L_hard

L_repr：表征蒸馏 (MSE)
  - 让学生表征接近教师表征

L_soft：软标签蒸馏 (KL散度)
  - 让学生的分类分布接近教师（使用温度缩放）

L_hard：硬标签损失 (Focal Loss)
  - 保证最终分类准确率
```

## 常见问题

### Q: 教师准确率不高怎么办？

A: 如果教师从活动序列识别模式都做不好，说明：
1. GMM聚类的模式定义可能不合理
2. 活动特征编码可能丢失信息
3. 需要增加教师模型容量

### Q: 学生和教师差距太大怎么办？

A: 尝试：
1. 增加蒸馏温度（如T=4或5）
2. 增大alpha_repr，强化表征对齐
3. 增加学生模型容量
4. 使用课程学习：先用高温，逐步降低

### Q: 如何选择output_mode？

A: 
- `embedding`：推荐，直接用中间表征作为Decoder条件
- `distribution`：需要额外的模式原型embedding
- `both`：同时输出，灵活但计算量大

### Q: 集成后Decoder效果变差？

A: 正常现象，因为学生的表征空间与原有不同。解决：
1. 先冻结学生，微调Decoder几个epoch
2. 然后解冻学生，端到端微调（小学习率）

## 引用与参考

- Knowledge Distillation: Hinton et al., 2015
- Focal Loss: Lin et al., 2017
- Transformer: Vaswani et al., 2017
