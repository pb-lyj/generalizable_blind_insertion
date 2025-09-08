"""
CNN自编码器模型 - 用于触觉力数据重建
输入形状: (3, 20, 20) - 三通道触觉力图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return F.relu(out)


class TactileCNNEncoder(nn.Module):
    """触觉CNN编码器"""
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 初始卷积 (3, 20, 20) -> (64, 20, 20)
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第一个残差块组 (64, 20, 20) -> (64, 10, 10)
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, stride=2),
            
            # 第二个残差块组 (64, 10, 10) -> (128, 5, 5)
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            
            # 第三个残差块组 (128, 5, 5) -> (256, 3, 3)
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            
            # 自适应池化到固定大小
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        
        # 全连接层映射到潜在空间
        self.fc = nn.Linear(256 * 5 * 5, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TactileCNNDecoder(nn.Module):
    """触觉CNN解码器"""
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        
        # 从潜在向量到特征图
        self.fc = nn.Linear(latent_dim, 256 * 5 * 5)
        
        self.decoder = nn.Sequential(
            # 从 (256, 5, 5) 上采样到 (128, 10, 10)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 从 (128, 10, 10) 上采样到 (64, 20, 20)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 输出层 (64, 20, 20) -> (3, 20, 20)
            nn.Conv2d(64, out_channels, 3, padding=1),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 5, 5)
        x = self.decoder(x)
        return x


class TactileCNNAutoencoder(nn.Module):
    """触觉CNN自编码器"""
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.encoder = TactileCNNEncoder(in_channels, latent_dim)
        self.decoder = TactileCNNDecoder(latent_dim, in_channels)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return {
            'reconstructed': reconstructed,
            'latent': latent
        }
    
    def encode(self, x):
        """编码输入到潜在空间"""
        return self.encoder(x)
    
    def decode(self, z):
        """从潜在空间解码"""
        return self.decoder(z)


def compute_cnn_autoencoder_losses(inputs, outputs, config):
    """
    计算CNN自编码器损失
    
    Args:
        inputs: 输入数据 (B, 3, 20, 20)
        outputs: 模型输出字典
        config: 损失配置
    
    Returns:
        loss: 总损失
        metrics: 损失分解字典
    """
    reconstructed = outputs['reconstructed']
    latent = outputs['latent']
    
    # 重建损失
    recon_loss = F.mse_loss(reconstructed, inputs)
    
    # L2正则化损失
    l2_loss = torch.norm(latent, p=2, dim=1).mean()
    
    # 总损失
    total_loss = (recon_loss + 
                  config.get('l2_lambda', 0.001) * l2_loss)
    
    metrics = {
        'recon_loss': recon_loss.item(),
        'l2_loss': l2_loss.item(),
        'total_loss': total_loss.item()
    }
    
    return total_loss, metrics


# 便利函数
def create_tactile_cnn_autoencoder(config):
    """创建触觉CNN自编码器"""
    return TactileCNNAutoencoder(
        in_channels=config.get('in_channels', 3),
        latent_dim=config.get('latent_dim', 128)
    )


if __name__ == '__main__':
    # 测试模型
    model = TactileCNNAutoencoder(latent_dim=128)
    
    # 测试输入
    x = torch.randn(4, 3, 20, 20)
    
    # 前向传播
    outputs = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"重建形状: {outputs['reconstructed'].shape}")
    print(f"潜在向量形状: {outputs['latent'].shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
