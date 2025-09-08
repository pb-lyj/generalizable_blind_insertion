"""
Feature-MLP模型 - 序列版本
输入: forces_l[3,20,20] + forces_r[3,20,20] -> CNN特征提取 -> 合并特征[256] + current_action[3] = 259维
输出: action_nextstep(end_XYZ)[3] = 3维
注意: 预测t+1时刻的动作，基于t时刻的触觉状态和动作，不跨越轨迹边界
t取值范围为1~L-1，其中L为轨迹长度
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

try:
    from cnn_autoencoder import TactileCNNAutoencoder
except ImportError:
    print("警告: 无法导入 TactileCNNAutoencoder，将使用简化特征提取")
    TactileCNNAutoencoder = None


class TactilePolicyFeatureMLP(nn.Module):
    """
    触觉策略Feature-MLP模型 - 时序预测版本
    基于预训练触觉特征的序列预测
    
    架构：
    1. 预训练CNN编码器提取左右手触觉特征 (128维 × 2 = 256维)
    2. 拼接当前动作 (256 + 3 = 259维)
    3. 通过MLP预测下一时刻动作 (259 → hidden → 3)
    """
    
    def __init__(self, 
                 feature_dim=128,           # 双手特征维度
                 action_dim=3,              # 输出动作维度 (dx, dy, dz)
                 hidden_dims=[512, 256, 128, 64],  # 隐藏层维度：259→512→256→128→64→3
                 dropout_rate=0.25,         # 提高Dropout到0.25
                 pretrained_encoder_path=None,
                 action_embed_dim=64,       # 动作嵌入维度
                 ):
        """
        Args:
            feature_dim: 单手触觉特征维度
            action_dim: 输出动作维度
            hidden_dims: MLP隐藏层维度列表
            dropout_rate: Dropout比率
            pretrained_encoder_path: 预训练编码器权重路径
        """
        super(TactilePolicyFeatureMLP, self).__init__()
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # 加载预训练的触觉特征提取器
        if TactileCNNAutoencoder is not None:
            self.tactile_encoder = TactileCNNAutoencoder(
                in_channels=3, 
                latent_dim=feature_dim
            )
            
            # 加载预训练权重
            if pretrained_encoder_path is not None and os.path.exists(pretrained_encoder_path):
                print(f"加载预训练触觉编码器: {pretrained_encoder_path}")
                checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
                
                # 检查checkpoint格式，提取模型状态字典
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        # 标准训练checkpoint格式
                        model_state = checkpoint['model_state_dict']
                        print("📦 检测到训练checkpoint格式，提取model_state_dict")
                    elif 'state_dict' in checkpoint:
                        # 另一种常见格式
                        model_state = checkpoint['state_dict']
                        print("📦 检测到state_dict格式")
                    else:
                        # 直接的状态字典
                        model_state = checkpoint
                        print("📦 检测到直接状态字典格式")
                else:
                    model_state = checkpoint
                
                # 加载状态字典
                self.tactile_encoder.load_state_dict(model_state, strict=True)
                print("✅ 成功加载预训练权重")
                
                # 打印checkpoint信息
                if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                    print(f"📊 预训练模型信息: epoch {checkpoint['epoch']}")
                    
            else:
                print("⚠️  预训练权重路径无效，使用随机初始化")
            
            # 冻结特征提取器参数
            for param in self.tactile_encoder.parameters():
                param.requires_grad = False
            print("🔒 特征提取器参数已冻结")
        else:
            print("❌ 无法导入CNN编码器，将使用随机特征")
            self.tactile_encoder = None
        
        # 构建动作input扩张
        self.action_embed_dim = action_embed_dim  # 可调：32/64/128
        self.action_encoder = nn.Sequential(
            nn.Linear(3, self.action_embed_dim),
            nn.LayerNorm(self.action_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 构建MLP网络
        # 输入维度: 左右手特征连接 + 当前动作 = feature_dim * 2 + 3
        # input_dim = feature_dim * 2 + 3  # 256 + 3 = 259
        input_dim = feature_dim * 2 + self.action_embed_dim  # 256 + 64 = 320
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"� 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
    def _initialize_weights(self):
        """初始化MLP权重"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, forces_l, forces_r, current_action):
        """
        前向传播
        
        Args:
            forces_l: 左手触觉力数据 (B, 3, 20, 20)
            forces_r: 右手触觉力数据 (B, 3, 20, 20)
            current_action: t时刻动作张量 (B, 3)
            
        Returns:
            next_action: t+1时刻动作 (B, 3)
        """
        batch_size = forces_l.size(0)
        
        if self.tactile_encoder is not None:
            # 使用预训练编码器提取特征
            # 注意：编码器参数已冻结，但我们需要保持特征的梯度流
            features_l = self.tactile_encoder.encoder(forces_l)  # (B, feature_dim)
            features_r = self.tactile_encoder.encoder(forces_r)  # (B, feature_dim)
        else:
            # 如果没有编码器，使用简单的全局平均池化作为特征
            features_l = torch.mean(forces_l.view(batch_size, -1), dim=1, keepdim=True)
            features_r = torch.mean(forces_r.view(batch_size, -1), dim=1, keepdim=True)
            # 扩展到指定的特征维度 - 确保保持梯度
            features_l = features_l.expand(-1, self.feature_dim).contiguous()
            features_r = features_r.expand(-1, self.feature_dim).contiguous()
        
        # # 连接左右手特征和当前动作
        # # combined_features: [features_l, features_r, current_action] = [128, 128, 3] = 259维
        # combined_features = torch.cat([features_l, features_r, current_action], dim=1)  # (B, 259)
        
        # # MLP预测下一时刻动作
        # next_action = self.mlp(combined_features)  # (B, action_dim)
        
        
        # 在 forward 里，拼接前先编码动作：
        action_feat = self.action_encoder(current_action)      # (B, 64)
        combined_features = torch.cat([features_l, features_r, action_feat], dim=1)
        next_action = self.mlp(combined_features)           # (B, action_dim)
        
        return next_action


def compute_feature_mlp_losses(inputs, outputs, dataset=None):
    """
    计算Feature-MLP损失 - 时序预测版本
    
    Args:
        inputs: 输入数据字典，包含 'target_next_action'
        outputs: 模型输出张量 (t+1时刻预测动作)
        dataset: 数据集对象，用于反归一化计算真实损失
        
    Returns:
        loss: 总损失
        metrics: 损失分解字典
    """
    predicted_next_action = outputs
    target_next_action = inputs['target_next_action']

    
    # 计算损失
    l1_loss = F.l1_loss(predicted_next_action, target_next_action, reduction='none').mean(dim=-1)  # (B,)
    mse_loss = F.mse_loss(predicted_next_action, target_next_action, reduction='none').mean(dim=-1) # (B,)
    
    # 对批次求平均得到标量损失
    total_loss = 0.5 * l1_loss.mean() + 0.5 * mse_loss.mean()
    
    # 计算未加权损失用于记录
    l1_loss_scalar = l1_loss.mean()
    mse_loss_scalar = mse_loss.mean()
    
    # 评估指标
    with torch.no_grad():
        rmse_loss = torch.sqrt(mse_loss_scalar)
        
        # 计算真实损失（反归一化后的L1损失）
        real_l1_loss = 0.0
        real_l1_loss_max = 0.0
        if dataset is not None and hasattr(dataset, 'denormalize_data'):
            try:
                # 反归一化预测值和目标值
                pred_denorm = dataset.denormalize_data(predicted_next_action.detach().cpu().numpy(), 'actions')
                target_denorm = dataset.denormalize_data(target_next_action.detach().cpu().numpy(), 'actions')
                
                # 计算逐样本的真实L1损失
                sample_real_l1_losses = np.mean(np.abs(pred_denorm - target_denorm), axis=1)  # (B,)
                
                # 计算平均值和最大值
                real_l1_loss = np.mean(sample_real_l1_losses)
                real_l1_loss_max = np.max(sample_real_l1_losses)
            except Exception as e:
                print(f"⚠️  计算真实损失失败: {e}")
                real_l1_loss = 0.0
                real_l1_loss_max = 0.0
        
    metrics = {
        'total_loss': total_loss.item(),
        'l1_error': l1_loss_scalar.item(),
        'mse_error': mse_loss_scalar.item(),
        'rmse_error': rmse_loss.item(),
        'real_l1_error(mm)': real_l1_loss * 1000,  # 真实损失（反归一化后）
        'real_l1_error_max(mm)': real_l1_loss_max * 1000,  # 每个batch中的最大真实损失
    }
    
    return total_loss, metrics


def prepare_feature_mlp_input_from_flexible_dataset(batch_data):
    """
    从FlexiblePolicyDataset批次中准备Feature-MLP模型的输入（时序预测版本）
    
    Args:
        batch_data: 来自FlexiblePolicyDataset的批次数据
    
    Returns:
        dict: Feature-MLP模型的输入字典，包含t时刻输入和t+1时刻目标
    """
    # 获取触觉力数据 (t时刻)
    forces_l = batch_data['forces_l']  # (B, 3, 20, 20)
    forces_r = batch_data['forces_r']  # (B, 3, 20, 20)
    
    # t时刻的动作 - 在序列模式下使用 'current_action'，普通模式下使用 'action'
    if 'current_action' in batch_data:
        current_action = batch_data['current_action'][:, :3]  # (B, 3) 序列模式
    else:
        current_action = batch_data['action'][:, :3]  # (B, 3) 普通模式
    
    # t+1时刻的动作（目标）- 在序列模式下才有这个键
    if 'next_action' in batch_data:
        target_next_action = batch_data['next_action'][:, :3]  # (B, 3) 序列模式
    else:
        target_next_action = batch_data['action'][:, :3]  # (B, 3) 普通模式下用当前action作为目标
    
    return {
        'forces_l': forces_l,
        'forces_r': forces_r,
        'current_action': current_action,
        'target_next_action': target_next_action
    }


def create_tactile_policy_feature_mlp(config):
    """创建触觉策略Feature-MLP模型"""
    return TactilePolicyFeatureMLP(
        feature_dim=config.get('feature_dim', 128),
        action_dim=config.get('action_dim', 3),
        hidden_dims=config.get('hidden_dims', [256, 128]),
        dropout_rate=config.get('dropout_rate', 0.1),
        pretrained_encoder_path=config.get('pretrained_encoder_path', None),
        action_embed_dim=config.get('action_embed_dim', 64)
    )


if __name__ == '__main__':
    # 简单测试
    config = {
        'feature_dim': 128,
        'action_dim': 3,
        'hidden_dims': [512, 256, 128, 64],  # 4层隐藏层，与resultant_mlp对齐
        'dropout_rate': 0.1,
        'pretrained_encoder_path': None,
        'action_embed_dim': 64
    }
    
    model = create_tactile_policy_feature_mlp(config)
    
    # 测试时序预测输入
    forces_l = torch.randn(4, 3, 20, 20)
    forces_r = torch.randn(4, 3, 20, 20)
    current_action = torch.randn(4, 3)
    output = model(forces_l, forces_r, current_action)
    
    print(f"输入触觉力l形状: {forces_l.shape}")
    print(f"输入触觉力r形状: {forces_r.shape}")
    print(f"当前动作形状: {current_action.shape}")
    print(f"输出下一动作形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试损失计算
    inputs = {'target_next_action': torch.randn_like(output)}
    loss, metrics = compute_feature_mlp_losses(inputs, output)
    
    print(f"总损失: {loss.item():.4f}")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    print("✅ Feature-MLP模型测试完成！")