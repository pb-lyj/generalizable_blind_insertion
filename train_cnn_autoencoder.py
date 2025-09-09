"""
CNN自编码器训练脚本 - 简化版本
使用CNN编码解码器进行触觉力数据重建训练
"""

import os
import sys
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# 设置代理（如果需要代理才能访问外网）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["WANDB_HTTP_TIMEOUT"] = "60"

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tactile_dataset import create_train_test_tactile_datasets
from cnn_autoencoder import TactileCNNAutoencoder, compute_cnn_autoencoder_losses

from ae_utils import save_comparison_images


def train_cnn_autoencoder(config):
    """
    训练CNN自编码器 - 简化版本
    Args:
        config: 配置字典
    """
    print("🚀 开始CNN自编码器训练...")
    
    # 登录wandb
    try:
        wandb.login()
        print("✅ wandb登录成功")
    except Exception as e:
        print(f"⚠️  wandb登录警告: {e}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output']['output_dir'], f"cnn_autoencoder_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 初始化 wandb
    run = wandb.init(
        project=config.get('wandb', {}).get('project', 'tactile-cnn-autoencoder'),
        name = config.get('wandb', {}).get('name', f"run_{timestamp}"),
        config=config,
        dir=output_dir,
        tags=['cnn-autoencoder', 'tactile', 'reconstruction'] + [timestamp],
        notes='CNN autoencoder training for tactile force reconstruction'
    )
    
    print("=" * 60)
    print("CNN Autoencoder Training")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['lr']}")
    print("=" * 60)
    
    
    # 创建数据集
    train_dataset, test_dataset, _ = create_train_test_tactile_datasets(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        start_frame=config['data']['start_frame'],
        normalize_method=config['data']['normalize_method']
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        pin_memory=True
    )

    # 创建模型
    model = TactileCNNAutoencoder(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim']
    ).cuda()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 训练循环
    best_loss = float('inf')

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        total_metrics = {}
        total_samples = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
            inputs = batch['image'].cuda()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss, metrics = compute_cnn_autoencoder_losses(
                inputs, outputs, config['loss'], dataset=train_dataset
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累积损失和指标
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value * batch_size
        
        # 计算平均损失和指标
        avg_metrics = {k: v/total_samples for k, v in total_metrics.items()}
        avg_loss = avg_metrics['total_loss']  # 直接从metrics获取总损失
        
        # 学习率调度
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到 wandb - 直接使用metrics中的所有损失
        wandb_log = {
            'train/learning_rate': current_lr,
        }
        for key, value in avg_metrics.items():
            wandb_log[f'train/{key}'] = value
        
        run.log(wandb_log, step=epoch)
        
        # 打印训练信息
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"  Learning Rate: {current_lr:.6e}")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.6f}")
        print("-" * 50)
        
        # 验证阶段 - 由eval_every控制频率
        if epoch % config['training']['eval_every'] == 0:
            print("🔍 开始验证阶段...")
            val_metrics = evaluate_model(model, test_loader, config['loss'])
            
            # 记录验证指标到 wandb
            val_wandb_log = {}
            for key, value in val_metrics.items():
                val_wandb_log[f'val/{key}'] = value
            
            run.log(val_wandb_log, step=epoch)
            
            # 打印验证信息
            print(f"验证结果:")
            for key, value in val_metrics.items():
                print(f"  val_{key}: {value:.6f}")
            print("-" * 50)
        
        # 每10个epoch可视化重建结果
        if epoch % 10 == 0:
            epoch_dir = os.path.join(visualization_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            print(f"正在生成第{epoch}轮的重建可视化...")
            visualize_reconstruction(model, train_loader, epoch_dir)
        
        # 保存模型检查点
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存最佳模型
            best_model_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': config
            }, best_model_path)
            wandb.save(best_model_path)
            print(f"💾 保存最佳模型 (Loss: {best_loss:.6f})")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': avg_loss,
        'config': config
    }, final_model_path)
    wandb.save(final_model_path)
    
    # 最终可视化重建结果
    print("正在生成最终的重建可视化...")
    final_viz_dir = os.path.join(visualization_dir, "final")
    os.makedirs(final_viz_dir, exist_ok=True)
    visualize_reconstruction(model, train_loader, final_viz_dir)
    
    # 记录训练总结
    run.log({
        'final_loss': avg_loss,
        'best_loss': best_loss,
        'total_epochs': epoch,
        'total_params': sum(p.numel() for p in model.parameters()),
        'dataset_size': len(train_dataset)
    })
    
    print("✅ CNN自编码器训练完成!")
    return model, best_loss


def evaluate_model(model, dataloader, loss_config):
    """
    评估模型在验证集上的性能
    Args:
        model: 要评估的模型
        dataloader: 验证数据加载器
        loss_config: 损失配置
    Returns:
        dict: 包含各种损失指标的字典
    """
    model.eval()
    total_metrics = {}
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].cuda()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss, metrics = compute_cnn_autoencoder_losses(
                inputs, outputs, loss_config, dataset=dataloader.dataset
            )
            
            # 累积指标
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value * batch_size
    
    # 计算平均指标
    avg_metrics = {k: v/total_samples for k, v in total_metrics.items()}
    return avg_metrics


def visualize_reconstruction(model, dataloader, output_dir, max_batches=None):
    """可视化重建结果 - 新版本，使用对比图"""
    model.eval()
    
    # 计算总样本数和需要绘制的样本数（总验证集的  分之一）
    total_samples = len(dataloader.dataset)
    target_samples = max(1, total_samples // 400)  # 至少绘制1个样本
    
    if max_batches is None:
        batch_size = dataloader.batch_size
        max_batches = max(1, target_samples // batch_size)  # 计算需要的批次数
    
    print(f"📊 可视化统计: 总样本={total_samples}, 目标样本={target_samples}, 最大批次={max_batches}")
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            inputs = batch['image'].cuda()
            outputs = model(inputs)
            reconstructions = outputs['reconstructed']
            
            # 保存对比图
            save_comparison_images(
                inputs.cpu(),
                reconstructions.cpu(),
                output_dir,
                prefix=f"comparison_batch_{batch_idx}"
            )
            
            sample_count += inputs.size(0)
            
        print(f"✅ 已生成 {sample_count} 个样本的重建对比图")


def main(config):
    """主训练函数 - 简化版本"""
    print("🎯 CNN自编码器训练开始")
    print("🔧连接wandb...")

    return train_cnn_autoencoder(config)


if __name__ == '__main__':
    # 简化配置
    config = {
        'data': {
            'data_root': 'data25.7_aligned',
            'categories': [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ],
            'start_frame': 0,
            'normalize_method': 'zscore'
        },
        'wandb': {
            'project': 'tactile-latent-autoencoder',
            'name': 'cnn_ae_base_run'
        },
        'model': {
            'in_channels': 3,
            'latent_dim': 128
        },
        'loss': {
            'l2_lambda': 0.001,
            'use_resultant_loss': False,  # 启用合力和合力矩损失
            'force_lambda': 0.1,         # 合力损失权重
            'moment_lambda': 0.05        # 合力矩损失权重（通常比合力小一些）
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'eval_every': 1
        },
        'output': {
            'output_dir': "ae_checkpoints"
        }
    }
    
    main(config)
