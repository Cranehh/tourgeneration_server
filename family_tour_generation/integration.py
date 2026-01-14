"""
将蒸馏好的学生模型集成到主生成模型中

使用方法：
1. 训练完成后，使用 PatternPredictorAdapter 加载学生模型
2. 替换主模型中的 Pattern Expert 分支
3. 微调 Decoder 以适应新的 pattern embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PatternPredictorAdapter(nn.Module):
    """
    将蒸馏好的学生模型适配为主模型的 Pattern Predictor
    
    输入：家庭属性 + 成员属性
    输出：
        - pattern_embedding: 可用于 Decoder 条件输入的 embedding
        - family_pattern_prob: 家庭模式概率分布
        - person_pattern_prob: 个人模式概率分布
    """
    
    def __init__(
        self,
        student_checkpoint_path: str,
        output_mode: str = 'embedding',  # 'embedding', 'distribution', 'both'
        freeze_student: bool = True
    ):
        """
        Args:
            student_checkpoint_path: 学生模型检查点路径
            output_mode: 输出模式
                - 'embedding': 输出 pattern embedding（推荐）
                - 'distribution': 输出模式概率分布
                - 'both': 同时输出两者
            freeze_student: 是否冻结学生模型参数
        """
        super().__init__()
        
        self.output_mode = output_mode
        
        # 加载学生模型
        self._load_student(student_checkpoint_path)
        
        # 冻结参数
        if freeze_student:
            for param in self.parameters():
                param.requires_grad = False
                
    def _load_student(self, checkpoint_path: str):
        """加载学生模型组件"""
        
        # 先导入学生模型类
        from pattern_distillation import (
            PatternDistillConfig,
            PatternStudent,
            PersonAttributeEncoder,
            FamilyAttributeEncoder
        )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 获取配置
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            self.config = PatternDistillConfig()
        
        # 创建学生模型
        self.student = PatternStudent(self.config)
        
        # 加载权重
        if 'student_state_dict' in checkpoint:
            self.student.load_state_dict(checkpoint['student_state_dict'])
        else:
            # 如果是分组件导出的，逐个加载
            self.student.person_encoder.load_state_dict(checkpoint['person_encoder'])
            self.student.family_encoder.load_state_dict(checkpoint['family_encoder'])
            self.student.cross_attention.load_state_dict(checkpoint['cross_attention'])
            self.student.member_interaction.load_state_dict(checkpoint['member_interaction'])
            self.student.person_proj.load_state_dict(checkpoint['person_proj'])
            self.student.family_proj.load_state_dict(checkpoint['family_proj'])
            self.student.person_classifier.load_state_dict(checkpoint['person_classifier'])
            self.student.family_classifier.load_state_dict(checkpoint['family_classifier'])
            
        self.student.eval()
        
    def forward(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            family_attr: [B, family_dim] 家庭属性
            member_attr: [B, M, member_dim] 成员属性
            member_mask: [B, M] 有效成员mask
            
        Returns:
            Dict包含（根据output_mode）:
                - person_embedding: [B, M, d_model] 个人模式embedding
                - family_embedding: [B, d_model] 家庭模式embedding
                - person_pattern_prob: [B, M, num_person_patterns] 个人模式概率
                - family_pattern_prob: [B, num_family_patterns] 家庭模式概率
        """
        # 学生前向
        outputs = self.student(family_attr, member_attr, member_mask)
        
        result = {}
        
        if self.output_mode in ['embedding', 'both']:
            result['person_embedding'] = outputs['person_repr']
            result['family_embedding'] = outputs['family_repr']
            
        if self.output_mode in ['distribution', 'both']:
            result['person_pattern_prob'] = F.softmax(outputs['person_logits'], dim=-1)
            result['family_pattern_prob'] = F.softmax(outputs['family_logits'], dim=-1)
            
        return result
    
    def get_pattern_condition(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取可直接用于Decoder的pattern条件
        
        Returns:
            person_condition: [B, M, d_model]
            family_condition: [B, d_model]
        """
        outputs = self.forward(family_attr, member_attr, member_mask)
        
        if 'person_embedding' in outputs:
            return outputs['person_embedding'], outputs['family_embedding']
        else:
            # 如果只有概率分布，需要转换
            # 这里可以用概率加权的模式原型embedding
            raise NotImplementedError(
                "Distribution-only mode requires pattern prototype embeddings"
            )


class IntegratedPatternPredictor(nn.Module):
    """
    带有适配层的完整 Pattern Predictor
    
    可以替换主模型中的 Pattern Expert，同时提供与原有接口兼容的输出
    """
    
    def __init__(
        self,
        student_checkpoint_path: str,
        target_d_model: int = 256,  # 主模型的 d_model
        num_family_patterns: int = 20,
        num_person_patterns: int = 40,
        freeze_student: bool = True
    ):
        super().__init__()
        
        # 加载学生模型
        self.pattern_predictor = PatternPredictorAdapter(
            student_checkpoint_path,
            output_mode='both',
            freeze_student=freeze_student
        )
        
        student_d_model = self.pattern_predictor.config.d_model
        
        # 适配层（如果维度不匹配）
        if student_d_model != target_d_model:
            self.person_adapter = nn.Linear(student_d_model, target_d_model)
            self.family_adapter = nn.Linear(student_d_model, target_d_model)
        else:
            self.person_adapter = nn.Identity()
            self.family_adapter = nn.Identity()
            
        self.target_d_model = target_d_model
        self.num_family_patterns = num_family_patterns
        self.num_person_patterns = num_person_patterns
        
    def forward(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，输出与原有 Pattern Expert 兼容的格式
        
        Returns:
            Dict包含:
                - pattern_condition: [B, M, d_model] 可用于Decoder的条件embedding
                - family_pattern_condition: [B, d_model] 家庭级别条件
                - family_pattern_prob: [B, num_family_patterns]
                - person_pattern_prob: [B, M, num_person_patterns]
        """
        outputs = self.pattern_predictor(family_attr, member_attr, member_mask)
        
        # 适配维度
        person_embedding = self.person_adapter(outputs['person_embedding'])
        family_embedding = self.family_adapter(outputs['family_embedding'])
        
        return {
            'pattern_condition': person_embedding,
            'family_pattern_condition': family_embedding,
            'family_pattern_prob': outputs['family_pattern_prob'],
            'person_pattern_prob': outputs['person_pattern_prob']
        }


def replace_pattern_expert(
    main_model: nn.Module,
    student_checkpoint_path: str,
    freeze_student: bool = True
):
    """
    替换主模型中的 Pattern Expert
    
    Args:
        main_model: 主生成模型
        student_checkpoint_path: 蒸馏好的学生模型路径
        freeze_student: 是否冻结学生参数
        
    Returns:
        修改后的主模型
    """
    # 获取主模型的配置
    # 假设主模型有 encoder.pattern_expert 或类似结构
    # 需要根据实际主模型结构调整
    
    target_d_model = getattr(main_model, 'd_model', 256)
    
    # 创建新的 Pattern Predictor
    new_predictor = IntegratedPatternPredictor(
        student_checkpoint_path=student_checkpoint_path,
        target_d_model=target_d_model,
        freeze_student=freeze_student
    )
    
    # 替换（需要根据主模型结构调整）
    # 例如：main_model.encoder.pattern_expert = new_predictor
    
    print("Pattern Expert replaced with distilled student model")
    print(f"Student parameters frozen: {freeze_student}")
    
    return main_model, new_predictor


class PatternConditionInjector(nn.Module):
    """
    将 pattern condition 注入到 Decoder 中的辅助模块
    
    支持多种注入方式：
    1. 加法融合
    2. 拼接后投影
    3. Cross-attention
    """
    
    def __init__(
        self,
        d_model: int,
        injection_mode: str = 'add'  # 'add', 'concat', 'attention'
    ):
        super().__init__()
        
        self.injection_mode = injection_mode
        
        if injection_mode == 'concat':
            self.proj = nn.Linear(d_model * 2, d_model)
        elif injection_mode == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        pattern_condition: torch.Tensor
    ) -> torch.Tensor:
        """
        将 pattern condition 注入到 hidden states
        
        Args:
            hidden_states: [B, M, T, d_model] 或 [B, T, d_model]
            pattern_condition: [B, M, d_model] 或 [B, d_model]
            
        Returns:
            注入后的 hidden states
        """
        # 扩展 pattern_condition 到与 hidden_states 相同的形状
        if hidden_states.dim() == 4:  # [B, M, T, d]
            B, M, T, D = hidden_states.shape
            if pattern_condition.dim() == 2:  # [B, d]
                pattern_condition = pattern_condition.unsqueeze(1).unsqueeze(2)
                pattern_condition = pattern_condition.expand(B, M, T, D)
            elif pattern_condition.dim() == 3:  # [B, M, d]
                pattern_condition = pattern_condition.unsqueeze(2)
                pattern_condition = pattern_condition.expand(B, M, T, D)
        elif hidden_states.dim() == 3:  # [B, T, d]
            B, T, D = hidden_states.shape
            if pattern_condition.dim() == 2:  # [B, d]
                pattern_condition = pattern_condition.unsqueeze(1)
                pattern_condition = pattern_condition.expand(B, T, D)
        
        if self.injection_mode == 'add':
            return hidden_states + pattern_condition
            
        elif self.injection_mode == 'concat':
            concat = torch.cat([hidden_states, pattern_condition], dim=-1)
            return self.proj(concat)
            
        elif self.injection_mode == 'attention':
            # 将 hidden_states reshape 用于 attention
            original_shape = hidden_states.shape
            hidden_flat = hidden_states.view(-1, original_shape[-2], original_shape[-1])
            pattern_flat = pattern_condition.view(-1, 1, original_shape[-1])
            
            attended, _ = self.attention(
                query=hidden_flat,
                key=pattern_flat,
                value=pattern_flat
            )
            return attended.view(original_shape) + hidden_states  # residual
            
        else:
            raise ValueError(f"Unknown injection mode: {self.injection_mode}")


# ==================== 使用示例 ====================

def demo_integration():
    """演示如何集成蒸馏模型"""
    
    # 假设已有训练好的学生模型
    student_checkpoint = "./checkpoints_student/best_student.pt"
    
    # 创建适配器
    adapter = PatternPredictorAdapter(
        student_checkpoint_path=student_checkpoint,
        output_mode='both',
        freeze_student=True
    )
    
    # 模拟输入
    batch_size = 4
    max_members = 8
    family_dim = 10
    member_dim = 51
    
    family_attr = torch.randn(batch_size, family_dim)
    member_attr = torch.randn(batch_size, max_members, member_dim)
    member_mask = torch.ones(batch_size, max_members, dtype=torch.bool)
    member_mask[:, 5:] = False  # 假设每个家庭只有5个成员
    
    # 前向传播
    with torch.no_grad():
        outputs = adapter(family_attr, member_attr, member_mask)
    
    print("Output keys:", outputs.keys())
    print("Person embedding shape:", outputs.get('person_embedding', 'N/A'))
    print("Family embedding shape:", outputs.get('family_embedding', 'N/A'))
    print("Person pattern prob shape:", outputs.get('person_pattern_prob', 'N/A'))
    print("Family pattern prob shape:", outputs.get('family_pattern_prob', 'N/A'))
    
    return adapter


if __name__ == '__main__':
    # 这里只是演示代码结构，实际运行需要有训练好的检查点
    print("This module provides integration utilities for the distilled pattern predictor.")
    print("See demo_integration() for usage example.")
