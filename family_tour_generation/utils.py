"""
模型加载和推理工具
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import logging

from .config import ModelConfig, TrainConfig
from .model import FamilyTourGenerator, create_model
from .data import FamilyTourBatch

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
    strict: bool = True
) -> Tuple[FamilyTourGenerator, ModelConfig, Dict]:
    """
    从检查点文件加载模型
    
    Args:
        checkpoint_path: 检查点文件路径 (.pt 文件)
        device: 目标设备 ('cuda', 'cpu', 或 None 自动选择)
        strict: 是否严格匹配参数名 (默认True)
    
    Returns:
        model: 加载好参数的模型
        config: 模型配置
        checkpoint_info: 检查点中的其他信息 (epoch, loss等)
    
    Example:
        >>> model, config, info = load_model_from_checkpoint('checkpoints/best_model.pt')
        >>> print(f"Loaded model from epoch {info['epoch']}")
        >>> model.eval()
        >>> predictions = model.generate(family_attr, member_attr, member_mask)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 确定设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # 如果检查点中没有配置，使用默认配置
        logger.warning("No model_config found in checkpoint, using default config")
        config = ModelConfig()
    
    # 创建模型
    model = create_model(config)
    
    # 加载参数
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # 移动到目标设备
    model.to(device)
    
    # 提取其他信息
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', -1),
        'global_step': checkpoint.get('global_step', -1),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'train_config': checkpoint.get('train_config', None)
    }
    
    logger.info(f"Model loaded successfully from epoch {checkpoint_info['epoch']}")
    logger.info(f"Best validation loss: {checkpoint_info['best_val_loss']:.4f}")
    
    return model, config, checkpoint_info


def load_model_state_dict(
    model: FamilyTourGenerator,
    state_dict_path: Union[str, Path],
    device: Optional[str] = None,
    strict: bool = True
) -> FamilyTourGenerator:
    """
    仅加载模型参数 (state_dict)，不包含配置
    适用于只保存了 model.state_dict() 的情况
    
    Args:
        model: 已创建的模型实例
        state_dict_path: state_dict 文件路径
        device: 目标设备
        strict: 是否严格匹配参数名
    
    Returns:
        model: 加载好参数的模型
    
    Example:
        >>> config = ModelConfig(family_dim=32, member_dim=48)
        >>> model = create_model(config)
        >>> model = load_model_state_dict(model, 'model_weights.pt')
    """
    state_dict_path = Path(state_dict_path)
    
    if not state_dict_path.exists():
        raise FileNotFoundError(f"State dict not found: {state_dict_path}")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    state_dict = torch.load(state_dict_path, map_location=device)
    
    # 如果保存的是完整checkpoint，提取state_dict
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    
    logger.info(f"State dict loaded from {state_dict_path}")
    
    return model


def save_model(
    model: FamilyTourGenerator,
    save_path: Union[str, Path],
    config: Optional[ModelConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: int = -1,
    global_step: int = -1,
    best_val_loss: float = float('inf'),
    train_config: Optional[TrainConfig] = None,
    save_only_weights: bool = False
):
    """
    保存模型
    
    Args:
        model: 模型实例
        save_path: 保存路径
        config: 模型配置 (推荐保存)
        optimizer: 优化器 (用于继续训练)
        scheduler: 学习率调度器
        epoch: 当前epoch
        global_step: 全局步数
        best_val_loss: 最佳验证损失
        train_config: 训练配置
        save_only_weights: 是否只保存权重
    
    Example:
        >>> save_model(model, 'checkpoints/model.pt', config=config, epoch=10)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_only_weights:
        # 只保存模型权重
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model weights saved to {save_path}")
    else:
        # 保存完整检查点
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': config,
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'train_config': train_config
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f"Full checkpoint saved to {save_path}")


class ModelLoader:
    """
    模型加载器类
    提供更方便的模型加载和管理接口
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: 默认设备
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.checkpoint_info = None
    
    def load(
        self, 
        checkpoint_path: Union[str, Path],
        strict: bool = True
    ) -> 'ModelLoader':
        """
        加载模型
        
        Args:
            checkpoint_path: 检查点路径
            strict: 是否严格匹配
        
        Returns:
            self (支持链式调用)
        """
        self.model, self.config, self.checkpoint_info = load_model_from_checkpoint(
            checkpoint_path, self.device, strict
        )
        return self
    
    def eval(self) -> 'ModelLoader':
        """设置为评估模式"""
        if self.model is not None:
            self.model.eval()
        return self
    
    def train(self) -> 'ModelLoader':
        """设置为训练模式"""
        if self.model is not None:
            self.model.train()
        return self
    
    def to(self, device: str) -> 'ModelLoader':
        """移动到指定设备"""
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self
    
    @torch.no_grad()
    def generate(
        self,
        family_attr: torch.Tensor,
        member_attr: torch.Tensor,
        member_mask: torch.BoolTensor,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        生成活动链
        
        Args:
            family_attr: (B, Ff) 家庭属性
            member_attr: (B, max_members, Fm) 成员属性
            member_mask: (B, max_members) 有效成员掩码
            max_length: 最大生成长度
        
        Returns:
            生成的活动链
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self.model.eval()
        
        # 移动到设备
        family_attr = family_attr.to(self.device)
        member_attr = member_attr.to(self.device)
        member_mask = member_mask.to(self.device)
        
        return self.model.generate(family_attr, member_attr, member_mask, max_length)
    
    @torch.no_grad()
    def predict(self, batch: FamilyTourBatch) -> Dict[str, torch.Tensor]:
        """
        对批次数据进行预测 (teacher forcing模式)
        
        Args:
            batch: 输入批次
        
        Returns:
            预测结果
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self.model.eval()
        batch = batch.to(self.device)
        
        return self.model(batch, teacher_forcing=True)
    
    def get_model(self) -> FamilyTourGenerator:
        """获取模型实例"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model
    
    def get_config(self) -> ModelConfig:
        """获取模型配置"""
        if self.config is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.config
    
    def summary(self) -> str:
        """获取模型摘要"""
        if self.model is None:
            return "No model loaded"
        
        params = self.model.count_parameters()
        info = self.checkpoint_info or {}
        
        summary_str = f"""
Model Summary:
==============
Device: {self.device}
Total Parameters: {params['total']:,}
  - Encoder: {params['encoder']:,}
  - Decoder: {params['decoder']:,}

Checkpoint Info:
  - Epoch: {info.get('epoch', 'N/A')}
  - Global Step: {info.get('global_step', 'N/A')}
  - Best Val Loss: {info.get('best_val_loss', 'N/A')}

Model Config:
  - d_model: {self.config.d_model}
  - num_heads: {self.config.num_heads}
  - num_decoder_layers: {self.config.num_decoder_layers}
  - max_members: {self.config.max_members}
  - max_activities: {self.config.max_activities}
"""
        return summary_str


# 便捷函数
def quick_load(checkpoint_path: Union[str, Path], device: Optional[str] = None) -> FamilyTourGenerator:
    """
    快速加载模型 (一行代码)
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        加载好的模型 (评估模式)
    
    Example:
        >>> model = quick_load('checkpoints/best_model.pt')
        >>> predictions = model.generate(family_attr, member_attr, member_mask)
    """
    model, _, _ = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()
    return model


# 使用示例
if __name__ == '__main__':
    # 示例1: 使用函数加载
    print("=" * 50)
    print("Example 1: Load using function")
    print("=" * 50)
    
    # model, config, info = load_model_from_checkpoint('checkpoints/best_model.pt')
    # model.eval()
    
    # 示例2: 使用类加载
    print("\n" + "=" * 50)
    print("Example 2: Load using ModelLoader class")
    print("=" * 50)
    
    # loader = ModelLoader(device='cuda')
    # loader.load('checkpoints/best_model.pt').eval()
    # print(loader.summary())
    
    # 生成
    # predictions = loader.generate(family_attr, member_attr, member_mask)
    
    # 示例3: 快速加载
    print("\n" + "=" * 50)
    print("Example 3: Quick load")
    print("=" * 50)
    
    # model = quick_load('checkpoints/best_model.pt')
    
    # 示例4: 只加载权重
    print("\n" + "=" * 50)
    print("Example 4: Load only weights")
    print("=" * 50)
    
    # config = ModelConfig(family_dim=32, member_dim=48)
    # model = create_model(config)
    # model = load_model_state_dict(model, 'model_weights.pt')
    
    print("\nAll examples are commented out. Uncomment to run with actual checkpoint files.")