"""
CNN Autoencoder for tactile force data reconstruction
Input shape: (3, 20, 20) - 3-channel tactile force images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block"""
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
    """Tactile CNN Encoder"""
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, stride=2),
            
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        
        self.fc = nn.Linear(256 * 5 * 5, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TactileCNNDecoder(nn.Module):
    """Tactile CNN Decoder"""
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 256 * 5 * 5)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, out_channels, 3, padding=1),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 5, 5)
        x = self.decoder(x)
        return x


class TactileCNNAutoencoder(nn.Module):
    """Tactile CNN Autoencoder"""
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
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)


def compute_resultant_force_and_moment(force_maps):
    """
    Compute resultant force and moment from tactile force maps.
    
    Args:
        force_maps: (B, 3, H, W) tactile force maps, 3 channels for X, Y, Z forces
    
    Returns:
        resultant_force: (B, 3) resultant force [Fx, Fy, Fz]
        resultant_moment: (B, 3) resultant moment [Mx, My, Mz]
    """
    B, C, H, W = force_maps.shape
    
    fx = force_maps[:, 0, :, :]
    fy = force_maps[:, 1, :, :]
    fz = force_maps[:, 2, :, :]
    
    resultant_fx = torch.sum(fx, dim=(1, 2))
    resultant_fy = torch.sum(fy, dim=(1, 2))
    resultant_fz = torch.sum(fz, dim=(1, 2))
    
    resultant_force = torch.stack([resultant_fx, resultant_fy, resultant_fz], dim=1)
    
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=force_maps.device),
        torch.arange(W, dtype=torch.float32, device=force_maps.device),
        indexing='ij'
    )
    
    center_x, center_y = W / 2.0, H / 2.0
    x_coords = x_coords - center_x
    y_coords = y_coords - center_y
    
    x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
    y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
    
    # M = r × F (z=0 for 2D sensor surface)
    mx = y_coords * fz
    my = -x_coords * fz
    mz = x_coords * fy - y_coords * fx
    
    resultant_mx = torch.sum(mx, dim=(1, 2))
    resultant_my = torch.sum(my, dim=(1, 2))
    resultant_mz = torch.sum(mz, dim=(1, 2))
    
    resultant_moment = torch.stack([resultant_mx, resultant_my, resultant_mz], dim=1)
    
    return resultant_force, resultant_moment


def compute_cnn_autoencoder_losses(inputs, outputs, config, dataset=None):
    """
    Compute CNN autoencoder losses.
    
    Args:
        inputs: Input data (B, 3, 20, 20)
        outputs: Model output dictionary
        config: Loss configuration
        dataset: Dataset object for denormalization to compute physical losses
    
    Returns:
        loss: Total loss
        metrics: Loss breakdown dictionary
    """
    reconstructed = outputs['reconstructed']
    latent = outputs['latent']
    
    # 1. Reconstruction loss (pixel-wise MSE)
    recon_loss = F.mse_loss(reconstructed, inputs)
    
    # 2. L2 regularization loss
    l2_loss = torch.norm(latent, p=2, dim=1).mean()
    
    force_loss = torch.tensor(0.0, device=inputs.device)
    moment_loss = torch.tensor(0.0, device=inputs.device)
    real_force_loss = [0.0, 0.0, 0.0, 0.0]
    real_moment_loss = [0.0, 0.0, 0.0, 0.0]
    
    orig_force, orig_moment = compute_resultant_force_and_moment(inputs)
    recon_force, recon_moment = compute_resultant_force_and_moment(reconstructed)
    
    force_loss = F.mse_loss(recon_force, orig_force)
    moment_loss = F.mse_loss(recon_moment, orig_moment)
    
    # Compute physical unit losses for monitoring
    if dataset is not None and hasattr(dataset, 'denormalize_data'):
        try:
            inputs_denorm = dataset.denormalize_data(inputs.detach().cpu().numpy())
            reconstructed_denorm = dataset.denormalize_data(reconstructed.detach().cpu().numpy())
            inputs_denorm_tensor = torch.from_numpy(inputs_denorm).to(inputs.device)
            reconstructed_denorm_tensor = torch.from_numpy(reconstructed_denorm).to(inputs.device)
            
            orig_force_real, orig_moment_real = compute_resultant_force_and_moment(inputs_denorm_tensor)
            recon_force_real, recon_moment_real = compute_resultant_force_and_moment(reconstructed_denorm_tensor)
            
            # 计算各轴向的MSE损失
            force_diff = recon_force_real - orig_force_real  # (B, 3)
            moment_diff = recon_moment_real - orig_moment_real  # (B, 3)
            
            # 计算三轴分量的MSE
            force_mse_per_axis = torch.mean(force_diff ** 2, dim=0)  # (3,) [fx, fy, fz]
            moment_mse_per_axis = torch.mean(moment_diff ** 2, dim=0)  # (3,) [mx, my, mz]
            
            # 计算总体MSE
            force_mse_total = torch.mean(force_diff ** 2)  # 标量
            moment_mse_total = torch.mean(moment_diff ** 2)  # 标量
            
            # 转换为列表格式 [x, y, z, total]
            real_force_loss = [
                force_mse_per_axis[0].item(),  # fx
                force_mse_per_axis[1].item(),  # fy
                force_mse_per_axis[2].item(),  # fz
                force_mse_total.item()         # total
            ]
            
            real_moment_loss = [
                moment_mse_per_axis[0].item(),  # mx
                moment_mse_per_axis[1].item(),  # my
                moment_mse_per_axis[2].item(),  # mz
                moment_mse_total.item()         # total
            ]
            
        except Exception as e:
            print(f"⚠️  计算真实物理损失失败: {e}")
            real_force_loss = [0.0, 0.0, 0.0, 0.0]
            real_moment_loss = [0.0, 0.0, 0.0, 0.0]
        
    # 4. 总损失
    if config.get('use_resultant_loss', False):
        total_loss = (recon_loss + 
                config.get('l2_lambda', 0.001) * l2_loss +
                config.get('force_lambda', 1e-5) * force_loss +
                config.get('moment_lambda', 5e-7) * moment_loss)
    else:
        total_loss = (recon_loss + 
                      config.get('l2_lambda', 0.001) * l2_loss)
    
    metrics = {
        'recon_loss(mse)': recon_loss.item(),
        'l2_loss': l2_loss.item(),
        'force_loss': force_loss.item(),
        'moment_loss': moment_loss.item(),
        'real_force_loss_x(N)': real_force_loss[0],
        'real_force_loss_y(N)': real_force_loss[1],
        'real_force_loss_z(N)': real_force_loss[2],
        'real_force_loss_total(N)': real_force_loss[3],
        'real_moment_loss_x(N*pixel)': real_moment_loss[0],
        'real_moment_loss_y(N*pixel)': real_moment_loss[1],
        'real_moment_loss_z(N*pixel)': real_moment_loss[2],
        'real_moment_loss_total(N*pixel)': real_moment_loss[3],
        'total_loss': total_loss.item()
    }
    
    return total_loss, metrics


def create_tactile_cnn_autoencoder(config):
    """Create tactile CNN autoencoder"""
    return TactileCNNAutoencoder(
        in_channels=config.get('in_channels', 3),
        latent_dim=config.get('latent_dim', 128)
    )


if __name__ == '__main__':
    model = TactileCNNAutoencoder(latent_dim=128)
    x = torch.randn(4, 3, 20, 20)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
    print(f"Latent vector shape: {outputs['latent'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
