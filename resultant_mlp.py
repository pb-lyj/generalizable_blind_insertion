"""
MLP策略模型 - 序列版本
输入: resultant_force[6] + resultant_moment[6] + current_action[3] = 15维
输出: action_nextstep(end_XYZ)[3] = 3维
注意: 预测t+1时刻的动作，基于t时刻的状态和动作，不跨越轨迹边界
t取值范围为1~L-1，其中L为轨迹长度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TactilePolicyMLP(nn.Module):
    """
    触觉策略MLP模型 - 时序预测版本
    """
    def __init__(self, input_dim=15, output_dim=3, hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 构建MLP网络
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        
        # 中间隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, resultant_force, resultant_moment, current_action):
        """
        前向传播
        
        Args:
            resultant_force: t时刻合力张量 (B, 6)
            resultant_moment: t时刻合力矩张量 (B, 6)
            current_action: t时刻动作张量 (B, 3)
        
        Returns:
            next_action: t+1时刻动作 (B, 3)
        """
        # 拼接输入特征: [force(6), moment(6), action(3)] = 15维
        x = torch.cat([resultant_force, resultant_moment, current_action], dim=-1)  # (B, 15)
        
        # 通过MLP网络
        next_action = self.mlp(x)
        
        return next_action


def compute_mlp_policy_losses(inputs, outputs, dataset=None):
    """
    计算MLP策略损失 - 时序预测版本
    
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


def prepare_mlp_input_from_flexible_dataset(batch_data):
    """
    从FlexiblePolicyDataset批次中准备MLP模型的输入（时序预测版本）
    
    Args:
        batch_data: 来自FlexiblePolicyDataset的批次数据
    
    Returns:
        dict: MLP模型的输入字典，包含t时刻输入和t+1时刻目标
    """
    # 合并左右手的合力和合力矩 (t时刻)
    resultant_force = torch.cat([
        batch_data['resultant_force_l'], 
        batch_data['resultant_force_r']
    ], dim=-1)  # (B, 6)
    
    resultant_moment = torch.cat([
        batch_data['resultant_moment_l'], 
        batch_data['resultant_moment_r']
    ], dim=-1)  # (B, 6)
    
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
        'resultant_force': resultant_force,
        'resultant_moment': resultant_moment,
        'current_action': current_action,
        'target_next_action': target_next_action
    }


def create_tactile_policy_mlp(config):
    """创建触觉策略MLP模型"""
    return TactilePolicyMLP(
        input_dim=config.get('input_dim', 15),  # 默认15维：force(6) + moment(6) + action(3)
        output_dim=config.get('output_dim', 3),
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    # 简单测试
    config = {
        'input_dim': 15,  # force(6) + moment(6) + action(3)
        'output_dim': 3,
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.1
    }
    
    model = create_tactile_policy_mlp(config)
    
    # 测试时序预测输入
    resultant_force = torch.randn(4, 6)
    resultant_moment = torch.randn(4, 6)
    current_action = torch.randn(4, 3)
    output = model(resultant_force, resultant_moment, current_action)
    
    print(f"输入合力形状: {resultant_force.shape}")
    print(f"输入合力矩形状: {resultant_moment.shape}")
    print(f"当前动作形状: {current_action.shape}")
    print(f"输出下一动作形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试损失计算
    inputs = {'target_next_action': torch.randn_like(output)}
    loss, metrics = compute_mlp_policy_losses(inputs, output)
    
    print(f"总损失: {loss.item():.4f}")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
