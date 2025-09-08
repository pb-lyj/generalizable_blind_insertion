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


def save_comparison_images(original_images, reconstructed_images, save_path, prefix="comparison"):
    """
    保存原始图像和重建图像的对比图
    original_images: shape (batch_size, 3, H, W)，原始图像
    reconstructed_images: shape (batch_size, 3, H, W)，重建图像
    """
    # 处理输入格式：如果是tensor则转换为numpy
    if isinstance(original_images, torch.Tensor):
        original_images = original_images.detach().cpu().numpy()
    if isinstance(reconstructed_images, torch.Tensor):
        reconstructed_images = reconstructed_images.detach().cpu().numpy()
    
    batch_size = original_images.shape[0]
    
    for i in range(batch_size):
        # 创建左右对比布局
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        
        # 提取原始图像的每个通道
        orig_x = original_images[i, 0]  # X方向力
        orig_y = original_images[i, 1]  # Y方向力
        orig_z = original_images[i, 2]  # Z方向力
        
        # 提取重建图像的每个通道
        recon_x = reconstructed_images[i, 0]  # X方向力
        recon_y = reconstructed_images[i, 1]  # Y方向力
        recon_z = reconstructed_images[i, 2]  # Z方向力
        
        # 计算统一的颜色范围（使用原始图像和重建图像的最大绝对值）
        max_abs_value = max(
            np.max(np.abs(orig_x)), np.max(np.abs(orig_y)), np.max(np.abs(orig_z)),
            np.max(np.abs(recon_x)), np.max(np.abs(recon_y)), np.max(np.abs(recon_z))
        )
        vmin, vmax = -max_abs_value, max_abs_value
        
        # 第一行：原始图像的三个通道
        axes[0, 0].imshow(orig_x, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Original X Force', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Width')
        axes[0, 0].set_ylabel('Height')
        
        axes[0, 1].imshow(orig_y, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Original Y Force', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Width')
        axes[0, 1].set_ylabel('Height')
        
        im_orig_z = axes[0, 2].imshow(orig_z, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('Original Z Force', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Width')
        axes[0, 2].set_ylabel('Height')
        
        # 第二行：重建图像的三个通道
        axes[1, 0].imshow(recon_x, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Reconstructed X Force', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Width')
        axes[1, 0].set_ylabel('Height')
        
        axes[1, 1].imshow(recon_y, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Reconstructed Y Force', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Width')
        axes[1, 1].set_ylabel('Height')
        
        im_recon_z = axes[1, 2].imshow(recon_z, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[1, 2].set_title('Reconstructed Z Force', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Width')
        axes[1, 2].set_ylabel('Height')
        
        # 差异图：第一行右侧三个图
        diff_x = orig_x - recon_x
        diff_y = orig_y - recon_y
        diff_z = orig_z - recon_z
        
        # 计算差异图的颜色范围
        max_diff = max(np.max(np.abs(diff_x)), np.max(np.abs(diff_y)), np.max(np.abs(diff_z)))
        diff_vmin, diff_vmax = -max_diff, max_diff
        
        axes[0, 3].imshow(diff_x, cmap='RdBu_r', interpolation='nearest', vmin=diff_vmin, vmax=diff_vmax)
        axes[0, 3].set_title('Difference X (Orig-Recon)', fontsize=12, fontweight='bold')
        axes[0, 3].set_xlabel('Width')
        axes[0, 3].set_ylabel('Height')
        
        axes[0, 4].imshow(diff_y, cmap='RdBu_r', interpolation='nearest', vmin=diff_vmin, vmax=diff_vmax)
        axes[0, 4].set_title('Difference Y (Orig-Recon)', fontsize=12, fontweight='bold')
        axes[0, 4].set_xlabel('Width')
        axes[0, 4].set_ylabel('Height')
        
        im_diff_z = axes[0, 5].imshow(diff_z, cmap='RdBu_r', interpolation='nearest', vmin=diff_vmin, vmax=diff_vmax)
        axes[0, 5].set_title('Difference Z (Orig-Recon)', fontsize=12, fontweight='bold')
        axes[0, 5].set_xlabel('Width')
        axes[0, 5].set_ylabel('Height')
        
        # 计算重建误差指标
        mse_x = np.mean((orig_x - recon_x) ** 2)
        mse_y = np.mean((orig_y - recon_y) ** 2)
        mse_z = np.mean((orig_z - recon_z) ** 2)
        total_mse = np.mean((original_images[i] - reconstructed_images[i]) ** 2)
        
        # 在第二行右侧添加文本信息
        axes[1, 3].axis('off')
        axes[1, 4].axis('off')
        axes[1, 5].axis('off')
        
        # 合并右侧三个子图显示误差信息
        info_text = f"""Reconstruction Metrics:
        
Total MSE: {total_mse:.6f}

Channel-wise MSE:
• X Force: {mse_x:.6f}
• Y Force: {mse_y:.6f}  
• Z Force: {mse_z:.6f}

Data Ranges:
• Original: [{np.min(original_images[i]):.4f}, {np.max(original_images[i]):.4f}]
• Reconstructed: [{np.min(reconstructed_images[i]):.4f}, {np.max(reconstructed_images[i]):.4f}]
• Max Difference: {max_diff:.4f}"""
        
        fig.text(0.72, 0.25, info_text, fontsize=11, verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 添加颜色条
        plt.colorbar(im_orig_z, ax=axes[0, 2], fraction=0.046, pad=0.04)
        plt.colorbar(im_recon_z, ax=axes[1, 2], fraction=0.046, pad=0.04)
        plt.colorbar(im_diff_z, ax=axes[0, 5], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(save_path, f"{prefix}_{i}.png")
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        plt.close()


def save_physicalXYZ_images(images, save_path, prefix="tactile"):
    """
    保存物理意义的XYZ触觉力图像
    images: shape (batch_size, 3, H, W)，3个通道分别表示X, Y, Z方向的力
    """
    # 处理输入格式：如果是tensor则转换为numpy，如果已经是numpy则保持不变
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        # 创建2行布局：第一行1个大图，第二行3个小图
        fig = plt.figure(figsize=(15, 10))
        
        # 提取每个通道的数据
        x_force = images[i, 0]  # X方向力
        y_force = images[i, 1]  # Y方向力
        z_force = images[i, 2]  # Z方向力
        
        # 第一行：三通道合并可视化（大图） - 放在正中间
        ax_combined = plt.subplot2grid((2, 3), (0, 1), fig=fig)
        
        # XY方向用箭头表示，Z用背景红色深浅表示
        ax_combined.imshow(z_force, cmap='Reds', alpha=0.7, interpolation='nearest')
        
        # 创建箭头网格
        H, W = x_force.shape
        step = max(1, min(H, W) // 10)  # 箭头间隔，避免过于密集
        y_indices, x_indices = np.meshgrid(
            np.arange(0, H, step),
            np.arange(0, W, step),
            indexing='ij'
        )
        
        # 下采样力数据用于箭头显示
        x_arrows = x_force[::step, ::step]
        y_arrows = y_force[::step, ::step]
        
        # 绘制箭头
        ax_combined.quiver(x_indices, y_indices, x_arrows, y_arrows, 
                          color='blue', alpha=0.8, scale_units='xy', scale=1,
                          width=0.003, headwidth=3, headlength=5)
        
        ax_combined.set_title('Combined XYZ Force Visualization (XY=arrows, Z=background)', fontsize=14, fontweight='bold')
        ax_combined.set_xlabel('Width')
        ax_combined.set_ylabel('Height')
        
        # 第二行：三个单独通道，设置统一的颜色范围
        # 计算所有力数据的最大绝对值，用于统一颜色范围
        max_abs_value = max(np.max(np.abs(x_force)), np.max(np.abs(y_force)), np.max(np.abs(z_force)))
        vmin, vmax = -max_abs_value, max_abs_value
        
        # X方向力 - 使用 RdBu_r 颜色映射（正值红色，负值蓝色）
        ax_x = plt.subplot2grid((2, 3), (1, 0), fig=fig)
        im1 = ax_x.imshow(x_force, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax_x.set_title('X Direction Force')
        ax_x.set_xlabel('Width')
        ax_x.set_ylabel('Height')
        plt.colorbar(im1, ax=ax_x, fraction=0.046, pad=0.04)
        
        # Y方向力 - 使用 RdBu_r 颜色映射（正值红色，负值蓝色）
        ax_y = plt.subplot2grid((2, 3), (1, 1), fig=fig)
        im2 = ax_y.imshow(y_force, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax_y.set_title('Y Direction Force')
        ax_y.set_xlabel('Width')
        ax_y.set_ylabel('Height')
        plt.colorbar(im2, ax=ax_y, fraction=0.046, pad=0.04)
        
        # Z方向力 - 使用 RdBu_r 颜色映射（正值红色，负值蓝色）
        ax_z = plt.subplot2grid((2, 3), (1, 2), fig=fig)
        im3 = ax_z.imshow(z_force, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax_z.set_title('Z Direction Force')
        ax_z.set_xlabel('Width')
        ax_z.set_ylabel('Height')
        plt.colorbar(im3, ax=ax_z, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(save_path, f"{prefix}_physical_xyz_{i}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()

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
    train_dataset, _, _ = create_train_test_tactile_datasets(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        start_frame=config['data']['start_frame'],
        normalize_method=config['data']['normalize_method']
    )
    loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers'], 
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
    patience_counter = 0
    patience = config['training']['patience']

    try:
        for epoch in range(1, config['training']['epochs'] + 1):
            model.train()
            total_loss = 0
            total_metrics = {}
            total_samples = 0
            
            for batch in tqdm(loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
                inputs = batch['image'].cuda()
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                loss, metrics = compute_cnn_autoencoder_losses(
                    inputs, outputs, config['loss']
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
            avg_loss = total_loss / total_samples
            avg_metrics = {k: v/total_samples for k, v in total_metrics.items()}
            
            # 学习率调度
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录到 wandb
            wandb_log = {
                'train/total_loss': avg_loss,
                'train/learning_rate': current_lr,
            }
            for key, value in avg_metrics.items():
                wandb_log[f'train/{key}'] = value
            
            run.log(wandb_log, step=epoch)
            
            # 打印训练信息
            print(f"Epoch {epoch}/{config['training']['epochs']}")
            print(f"  Total Loss: {avg_loss:.6f}")
            print(f"  Learning Rate: {current_lr:.6e}")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.6f}")
            print("-" * 50)
            
            # 每10个epoch可视化重建结果
            if epoch % 10 == 0:
                epoch_dir = os.path.join(visualization_dir, f"epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                print(f"正在生成第{epoch}轮的重建可视化...")
                visualize_reconstruction(model, loader, epoch_dir)
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
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
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"⏰ 早停：{patience} 个epoch没有改善")
                run.log({'early_stopping': True, 'stopped_epoch': epoch})
                break
        
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
        visualize_reconstruction(model, loader, final_viz_dir)
        
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
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        raise
    finally:
        run.finish()


def visualize_reconstruction(model, dataloader, output_dir, max_batches=None):
    """可视化重建结果 - 新版本，使用对比图"""
    model.eval()
    
    # 计算总样本数和需要绘制的样本数（总验证集的十分之一）
    total_samples = len(dataloader.dataset)
    target_samples = max(1, total_samples // 10)  # 至少绘制1个样本
    
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
    print(f"📊 配置摘要:")
    print(f"   数据根目录: {config['data']['data_root']}")
    print(f"   批次大小: {config['training']['batch_size']}")
    print(f"   学习率: {config['training']['lr']}")
    print(f"   训练轮数: {config['training']['epochs']}")
    
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
            'num_workers': 4,
            'normalize_method': 'zscore'
        },
        'model': {
            'in_channels': 3,
            'latent_dim': 128
        },
        'loss': {
            'l2_lambda': 0.001
        },
        'training': {
            'batch_size': 32,
            'epochs': 30,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'patience': 15
        },
        'wandb': {
            'project': "tactile-latent-autoencoder"
        },
        'output': {
            'output_dir': "ae_checkpoints"
        }
    }
    
    main(config)
