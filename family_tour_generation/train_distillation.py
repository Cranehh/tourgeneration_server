"""
模式蒸馏训练脚本

使用方法：
1. 先运行 train_teacher() 训练教师模型
2. 确认教师模型准确率达标后，运行 train_student() 蒸馏训练学生
3. 使用 export_student_for_integration() 导出学生模型供主模型集成
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from pattern_distillation import (
    PatternDistillConfig,
    PatternTeacher,
    PatternStudent,
    PatternDistillDataset,
    TeacherTrainer,
    PatternDistillTrainer,
    create_teacher,
    create_student,
    load_teacher,
    load_student,
    collate_fn
)


def load_data(data_dir: str = "../数据"):
    """加载训练和验证数据"""
    
    print("Loading data...")
    
    # 训练集
    data = np.load(f'{data_dir}/family_sample_improved_cluster_train.npy')
    family_data_train = np.concatenate([data[:, :10], data[:, -1:]], axis=1)  # [N, 11]
    member_data_train = np.load(f'{data_dir}/family_member_sample_improved_cluster_train.npy')  # [N, M, 51]
    activity_data_train = np.load(f'{data_dir}/family_activity_train.npy')  # [N, M, T, 29]
    member_mask_train = member_data_train[:, :, -2] != 0  # 使用valid flag
    activity_mask_train = activity_data_train[:, :, :, :27].sum(axis=-1) != 0
    
    family_pattern_train = np.load(f'{data_dir}/family_pattern_train.npy')  # [N, 20]
    person_pattern_train = np.load(f'{data_dir}/person_pattern_train.npy')  # [N, M, 40]
    
    # 验证集
    data = np.load(f'{data_dir}/family_sample_improved_cluster_test.npy')
    family_data_test = np.concatenate([data[:, :10], data[:, -1:]], axis=1)
    member_data_test = np.load(f'{data_dir}/family_member_sample_improved_cluster_test.npy')
    activity_data_test = np.load(f'{data_dir}/family_activity_test.npy')
    member_mask_test = member_data_test[:, :, -2] != 0
    activity_mask_test = activity_data_test[:, :, :, :27].sum(axis=-1) != 0
    
    family_pattern_test = np.load(f'{data_dir}/family_pattern_test.npy')
    person_pattern_test = np.load(f'{data_dir}/person_pattern_test.npy')
    
    print(f"Train samples: {len(family_data_train)}")
    print(f"Val samples: {len(family_data_test)}")
    print(f"Family pattern shape: {family_pattern_train.shape}")
    print(f"Person pattern shape: {person_pattern_train.shape}")
    
    return {
        'train': {
            'family_data': family_data_train,
            'member_data': member_data_train,
            'activity_data': activity_data_train,
            'member_mask': member_mask_train,
            'activity_mask': activity_mask_train,
            'family_pattern': family_pattern_train,
            'person_pattern': person_pattern_train
        },
        'val': {
            'family_data': family_data_test,
            'member_data': member_data_test,
            'activity_data': activity_data_test,
            'member_mask': member_mask_test,
            'activity_mask': activity_mask_test,
            'family_pattern': family_pattern_test,
            'person_pattern': person_pattern_test
        }
    }


def create_dataloaders(data: dict, config: PatternDistillConfig):
    """创建数据加载器"""
    
    train_dataset = PatternDistillDataset(
        family_data=data['train']['family_data'],
        member_data=data['train']['member_data'],
        activity_data=data['train']['activity_data'],
        member_mask=data['train']['member_mask'],
        activity_mask=data['train']['activity_mask'],
        family_pattern=data['train']['family_pattern'],
        person_pattern=data['train']['person_pattern']
    )
    
    val_dataset = PatternDistillDataset(
        family_data=data['val']['family_data'],
        member_data=data['val']['member_data'],
        activity_data=data['val']['activity_data'],
        member_mask=data['val']['member_mask'],
        activity_mask=data['val']['activity_mask'],
        family_pattern=data['val']['family_pattern'],
        person_pattern=data['val']['person_pattern']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_teacher(
    data_dir: str = "../数据",
    save_dir: str = "./checkpoints_teacher",
    num_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """
    阶段1：训练教师模型
    
    教师模型从活动序列识别模式，应该能达到很高的准确率（>90%）
    """
    print("=" * 60)
    print("Stage 1: Training Teacher Model")
    print("=" * 60)
    
    # 配置
    config = PatternDistillConfig(
        num_family_patterns=20,
        num_person_patterns=40,
        d_model=256,
        num_heads=8,
        num_encoder_layers=4,
        dropout=0.1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device
    )
    
    # 加载数据
    data = load_data(data_dir)
    train_loader, val_loader = create_dataloaders(data, config)
    
    # 创建教师模型
    teacher = create_teacher(config)
    print(f"Teacher model parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # 训练
    trainer = TeacherTrainer(
        config=config,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir
    )
    
    trainer.train(num_epochs)
    
    print("\nTeacher training completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Model saved to: {save_dir}/best_teacher.pt")
    
    return trainer


def train_student(
    teacher_checkpoint: str,
    data_dir: str = "../数据",
    save_dir: str = "./checkpoints_student",
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    temperature: float = 3.0,
    alpha_soft: float = 0.7,
    alpha_repr: float = 0.2,
    alpha_hard: float = 0.1,
    device: str = 'cuda'
):
    """
    阶段2：蒸馏训练学生模型
    
    使用训练好的教师模型指导学生学习
    """
    print("=" * 60)
    print("Stage 2: Distillation Training Student Model")
    print("=" * 60)
    
    # 配置
    config = PatternDistillConfig(
        num_family_patterns=20,
        num_person_patterns=40,
        d_model=256,
        num_heads=8,
        num_encoder_layers=4,
        dropout=0.1,
        temperature=temperature,
        alpha_soft=alpha_soft,
        alpha_repr=alpha_repr,
        alpha_hard=alpha_hard,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device
    )
    
    # 加载数据
    data = load_data(data_dir)
    train_loader, val_loader = create_dataloaders(data, config)
    
    # 加载教师模型
    print(f"Loading teacher from: {teacher_checkpoint}")
    teacher = load_teacher(teacher_checkpoint, config)
    print(f"Teacher loaded successfully")
    
    # 创建学生模型
    student = create_student(config)
    print(f"Student model parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # 蒸馏训练
    trainer = PatternDistillTrainer(
        config=config,
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir
    )
    
    trainer.train(num_epochs)
    
    print("\nDistillation training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}/best_student.pt")
    
    return trainer


def export_student_for_integration(
    student_checkpoint: str,
    output_path: str = "./student_for_integration.pt"
):
    """
    导出学生模型供主模型集成
    
    只导出学生的编码器部分（用于生成模式embedding）
    """
    print("Exporting student model for integration...")
    
    checkpoint = torch.load(student_checkpoint, map_location='cpu')
    config = checkpoint.get('config', PatternDistillConfig())
    
    student = create_student(config)
    student.load_state_dict(checkpoint['student_state_dict'])
    student.eval()
    
    # 导出必要的组件
    export_dict = {
        'config': config,
        'person_encoder': student.person_encoder.state_dict(),
        'family_encoder': student.family_encoder.state_dict(),
        'cross_attention': student.cross_attention.state_dict(),
        'member_interaction': student.member_interaction.state_dict(),
        'person_proj': student.person_proj.state_dict(),
        'family_proj': student.family_proj.state_dict(),
        'person_classifier': student.person_classifier.state_dict(),
        'family_classifier': student.family_classifier.state_dict(),
        # 完整的学生模型状态
        'student_state_dict': checkpoint['student_state_dict']
    }
    
    torch.save(export_dict, output_path)
    print(f"Student model exported to: {output_path}")
    
    return export_dict


def evaluate_models(
    teacher_checkpoint: str,
    student_checkpoint: str,
    data_dir: str = "../数据",
    device: str = 'cuda'
):
    """评估教师和学生模型"""
    
    print("=" * 60)
    print("Evaluating Models")
    print("=" * 60)
    
    config = PatternDistillConfig(device=device)
    
    # 加载数据
    data = load_data(data_dir)
    _, val_loader = create_dataloaders(data, config)
    
    # 加载模型
    teacher = load_teacher(teacher_checkpoint, config)
    student = load_student(student_checkpoint, config)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.eval()
    
    # 评估
    teacher_person_correct = 0
    teacher_family_correct = 0
    student_person_correct = 0
    student_family_correct = 0
    person_total = 0
    family_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 教师预测
            teacher_out = teacher(
                batch['activities'],
                batch['member_mask'],
                batch['activity_mask']
            )
            
            # 学生预测
            student_out = student(
                batch['family_attr'],
                batch['member_attr'],
                batch['member_mask']
            )
            
            # 教师准确率
            t_person_pred = teacher_out['person_logits'].argmax(dim=-1)
            t_family_pred = teacher_out['family_logits'].argmax(dim=-1)
            
            # 学生准确率
            s_person_pred = student_out['person_logits'].argmax(dim=-1)
            s_family_pred = student_out['family_logits'].argmax(dim=-1)
            
            # 目标
            person_target = batch['person_pattern_label']
            family_target = batch['family_pattern_label']
            mask = batch['member_mask']
            
            # 累计
            teacher_person_correct += ((t_person_pred == person_target) & mask).sum().item()
            teacher_family_correct += (t_family_pred == family_target).sum().item()
            student_person_correct += ((s_person_pred == person_target) & mask).sum().item()
            student_family_correct += (s_family_pred == family_target).sum().item()
            
            person_total += mask.sum().item()
            family_total += len(family_target)
    
    print("\nResults:")
    print("-" * 40)
    print(f"Teacher Person Accuracy:  {teacher_person_correct / person_total:.4f}")
    print(f"Teacher Family Accuracy:  {teacher_family_correct / family_total:.4f}")
    print(f"Student Person Accuracy:  {student_person_correct / person_total:.4f}")
    print(f"Student Family Accuracy:  {student_family_correct / family_total:.4f}")
    print("-" * 40)
    
    # 学生相对于教师的性能保持率
    person_retention = (student_person_correct / person_total) / (teacher_person_correct / person_total + 1e-8)
    family_retention = (student_family_correct / family_total) / (teacher_family_correct / family_total + 1e-8)
    print(f"Person Accuracy Retention: {person_retention:.2%}")
    print(f"Family Accuracy Retention: {family_retention:.2%}")


def main():
    """主函数：完整的两阶段训练流程"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Pattern Distillation Training')
    parser.add_argument('--stage', type=str, choices=['teacher', 'student', 'both', 'eval'],
                        default='student', help='Training stage')
    parser.add_argument('--data_dir', type=str, default='../数据', help='Data directory')
    parser.add_argument('--teacher_dir', type=str, default='./checkpoints_teacher',
                        help='Teacher checkpoint directory')
    parser.add_argument('--student_dir', type=str, default='./checkpoints_student',
                        help='Student checkpoint directory')
    parser.add_argument('--teacher_epochs', type=int, default=50, help='Teacher training epochs')
    parser.add_argument('--student_epochs', type=int, default=100, help='Student training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    if args.stage in ['teacher', 'both']:
        # 阶段1：训练教师
        train_teacher(
            data_dir=args.data_dir,
            save_dir=args.teacher_dir,
            num_epochs=args.teacher_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device
        )
    
    if args.stage in ['student', 'both']:
        # 阶段2：蒸馏训练学生
        teacher_checkpoint = f"{args.teacher_dir}/best_teacher.pt"
        train_student(
            teacher_checkpoint=teacher_checkpoint,
            data_dir=args.data_dir,
            save_dir=args.student_dir,
            num_epochs=args.student_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            temperature=args.temperature,
            device=args.device
        )
        
        # 导出学生模型
        student_checkpoint = f"{args.student_dir}/best_student.pt"
        export_student_for_integration(
            student_checkpoint=student_checkpoint,
            output_path=f"{args.student_dir}/student_for_integration.pt"
        )
    
    if args.stage == 'eval':
        # 评估
        teacher_checkpoint = f"{args.teacher_dir}/best_teacher.pt"
        student_checkpoint = f"{args.student_dir}/best_student.pt"
        evaluate_models(
            teacher_checkpoint=teacher_checkpoint,
            student_checkpoint=student_checkpoint,
            data_dir=args.data_dir,
            device=args.device
        )


if __name__ == '__main__':
    main()
