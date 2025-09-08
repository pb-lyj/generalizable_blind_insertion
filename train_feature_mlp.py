"""
Feature-MLP策略模型训练脚本 - 序列版本
输入: forces_l[3,20,20] + forces_r[3,20,20] -> CNN特征[256] + current_action[3] = 259维
输出: action_nextstep[3] = 3维
"""
import os
import sys
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 设置代理（如果需要代理才能访问外网）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["WANDB_HTTP_TIMEOUT"] = "60"

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 项目根路径
project_root = os.path.abspath(os.path.dirname(__file__))

from PointPairDataset import create_classic_datasets
from feature_mlp import create_tactile_policy_feature_mlp, compute_feature_mlp_losses, prepare_feature_mlp_input_from_flexible_dataset


def train_feature_mlp_policy(config):
    """
    训练Feature-MLP策略模型 - 序列版本
    """
    print("🚀 开始Feature-MLP策略训练...")
    
    # 登录wandb
    try:
        wandb.login()
        print("✅ wandb登录成功")
    except Exception as e:
        print(f"⚠️  wandb登录警告: {e}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output']['output_dir'], f"feature_mlp_policy_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 wandb
    run = wandb.init(
        project=config.get('wandb', {}).get('project', 'tactile-action-learn'),
        name = config.get('wandb', {}).get('name'),
        config=config,
        dir=output_dir,
        tags=['feature-mlp-policy', 'sequence'] + [timestamp],
        notes='Feature-MLP policy training with sequence prediction'
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    print("=" * 60)
    print("Feature-MLP Policy Training (Sequence)")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['lr']}")
    print("=" * 60)
    print("Model Configuration:")
    print(config['model'])
    print("=" * 60)
    
    # 创建数据集
    print("📂 加载数据集...")
    # 注意：Feature-MLP需要forces数据，所以必须设置use_forces=True
    from PointPairDataset import PointPairDataset
    
    # 创建训练数据集
    train_dataset = PointPairDataset(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        is_train=True,
        use_resultant=False,  
        use_forces=True,      
        normalization_config=config['data'].get('normalization_config', None),
        prediction_step=config['data'].get('prediction_step', 1)  # 支持预测步长
    )
    
    # 创建测试数据集
    test_dataset = PointPairDataset(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        is_train=False,
        use_resultant=False,  
        use_forces=True,      
        normalization_config=train_dataset.normalization_config,
        prediction_step=config['data'].get('prediction_step', 1)  # 支持预测步长
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers= 4, 
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers= 4, 
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 测试集: {len(test_dataset)} 样本")

    # 创建模型
    print("🏗️ 创建模型...")
    model = create_tactile_policy_feature_mlp(config['model']).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,  # patience frrom 10 to 2
        threshold=1e-4,    # 改善阈值
        cooldown=1,        # 降lr后冷却1个epoch
        min_lr=1e-6,       # 最低lr
    )


    # 训练循环
    best_loss = float('inf')
    best_model_path = None
    
    try:
        for epoch in range(config['training']['epochs']):
            print(f"\n🔄 Epoch {epoch + 1}/{config['training']['epochs']}")
            
            # 训练阶段
            train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, train_dataset)
            
            # 记录训练指标
            run.log({
                'train/loss': train_loss,
                'train/l1_error': train_metrics.get('l1_error', 0),
                'train/mse_error': train_metrics.get('mse_error', 0),
                'train/rmse_error': train_metrics.get('rmse_error', 0),
                'train/real_l1_error(mm)': train_metrics.get('real_l1_error(mm)', 0),
                'train/real_l1_error_max(mm)': train_metrics.get('real_l1_error_max(mm)', 0),
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)

            # 验证阶段
            if (epoch + 1) % config['training'].get('eval_every', 1) == 0:
                test_loss, test_metrics = evaluate(model, test_loader, device, test_dataset)
                
                print(f"📊 验证结果:")
                print(f"   Loss: {test_loss:.6f}")
                print(f"   Real L1 Error: {test_metrics.get('real_l1_error(mm)', 0):.2f} mm")
                print(f"   Real L1 Max: {test_metrics.get('real_l1_error_max(mm)', 0):.2f} mm")
                
                # 记录验证指标
                run.log({
                    'val/loss': test_loss,
                    'val/l1_error': test_metrics.get('l1_error', 0),
                    'val/mse_error': test_metrics.get('mse_error', 0),
                    'val/rmse_error': test_metrics.get('rmse_error', 0),
                    'val/real_l1_error(mm)': test_metrics.get('real_l1_error(mm)', 0),
                    'val/real_l1_error_max(mm)': test_metrics.get('real_l1_error_max(mm)', 0),
                }, step=epoch)
                
                # 保存最佳模型
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'config': config
                    }, best_model_path)
                    print(f"💾 保存最佳模型: {best_model_path}")
                
                # 学习率调度
                scheduler.step(test_loss)
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'config': config
        }, final_model_path)
        
        # 保存到 wandb
        wandb.save(final_model_path)
        if best_model_path:
            wandb.save(best_model_path)
        
        print("✅ Feature-MLP策略模型训练完成!")
        return model, best_loss, best_model_path
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        raise
    finally:
        run.finish()


def train_epoch(model, train_loader, optimizer, device, dataset=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_metrics = {}
    total_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        # 准备输入数据
        mlp_inputs = prepare_feature_mlp_input_from_flexible_dataset(batch)
        for key in mlp_inputs:
            if isinstance(mlp_inputs[key], torch.Tensor):
                mlp_inputs[key] = mlp_inputs[key].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        forces_l = mlp_inputs['forces_l']  # (B, 3, 20, 20)
        forces_r = mlp_inputs['forces_r']  # (B, 3, 20, 20)
        current_action = mlp_inputs['current_action']  # (B, 3)
        outputs = model(forces_l, forces_r, current_action)
        
        # 计算损失
        loss, metrics = compute_feature_mlp_losses(mlp_inputs, outputs, dataset=dataset)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累加统计
        batch_size = forces_l.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 累加指标
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value * batch_size
    
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {key: value / max(total_samples, 1) 
                   for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def evaluate(model, test_loader, device, dataset=None):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 准备输入数据
            mlp_inputs = prepare_feature_mlp_input_from_flexible_dataset(batch)
            for key in mlp_inputs:
                if isinstance(mlp_inputs[key], torch.Tensor):
                    mlp_inputs[key] = mlp_inputs[key].to(device)
            
            # 前向传播
            forces_l = mlp_inputs['forces_l']  # (B, 3, 20, 20)
            forces_r = mlp_inputs['forces_r']  # (B, 3, 20, 20)
            current_action = mlp_inputs['current_action']  # (B, 3)
            outputs = model(forces_l, forces_r, current_action)
            
            # 计算损失
            loss, metrics = compute_feature_mlp_losses(mlp_inputs, outputs, dataset=dataset)
            
            # 累加统计
            batch_size = forces_l.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 累加指标
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value * batch_size
    
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {key: value / max(total_samples, 1) 
                   for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def main(config):
    """主函数"""
    print("🎯 Feature-MLP策略训练开始 (序列版)")
    print(f"📊 配置摘要: Waiting for wandb")
    train_feature_mlp_policy(config)


if __name__ == '__main__':
    # 默认配置
    config = {
        'data': {
            'data_root': 'data25.7_aligned',
            'categories': [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma",
            ],
            'prediction_step': 5,  # 预测步长：t时刻预测t+n时刻目标
            'normalization_config': {
                'actions': {'method': 'zscore', 'params': None},
                'resultants': {'method': 'zscore', 'params': None},
                'forces': {'method': 'zscore', 'params': None}
            },
        },
        'model': {
            'feature_dim': 128,                    # 单手特征维度
            'action_dim': 3,                       # 输出动作维度
            'hidden_dims': [320, 256, 128, 64],     # MLP隐藏层：4层，与resultant_mlp对齐
                                                    # [384, 256, 128, 128] 平衡扩张
                                                    # [320, 256, 128, 64] 保守扩张，减少过拟合风险
            'dropout_rate': 0.1,                   # Dropout率
            'pretrained_encoder_path': 'cnnae_crt_128.pt',  # 预训练编码器路径
            'action_embed_dim' : 64                     # 动作嵌入维度
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'eval_every': 1
        },
        'wandb': {
            'project': "tactile-action-learn-test",
            'name': 'feature-mlp-policy-combined-scheduler-2'
        },
        'output': {
            'output_dir': 'checkpoints'
        }
    }
    
    # 检查路径
    data_path = os.path.join(project_root, config['data']['data_root'])
    config['data']['data_root'] = data_path
    config['output']['output_dir'] = os.path.join(project_root, config['output']['output_dir'])
    
    # 检查预训练模型路径
    pretrained_path = config['model']['pretrained_encoder_path']
    if pretrained_path and not os.path.isabs(pretrained_path):
        config['model']['pretrained_encoder_path'] = os.path.join(project_root, pretrained_path)
    
    if os.path.exists(data_path):
        print(f"✅ 数据路径存在: {data_path}")
        main(config)
    else:
        print(f"❌ 数据路径不存在: {data_path}")
