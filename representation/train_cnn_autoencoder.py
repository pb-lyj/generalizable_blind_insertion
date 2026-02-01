"""
CNN Autoencoder Training Script
Trains a CNN encoder-decoder for tactile force reconstruction.
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

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["WANDB_HTTP_TIMEOUT"] = "60"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from DatasetTactile import create_train_test_tactile_datasets
from cnn_autoencoder import TactileCNNAutoencoder, compute_cnn_autoencoder_losses

from ae_utils import save_comparison_images


def train_cnn_autoencoder(config):
    """
    Train CNN autoencoder.
    Args:
        config: Configuration dictionary
    """
    print("🚀 Starting CNN autoencoder training...")
    
    try:
        wandb.login()
        print("✅ wandb logged in")
    except Exception as e:
        print(f"⚠️  wandb login warning: {e}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output']['output_dir'], f"cnn_autoencoder_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
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

    model = TactileCNNAutoencoder(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim']
    ).cuda()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_loss = float('inf')

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        total_metrics = {}
        total_samples = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
            inputs = batch['image'].cuda()
            
            outputs = model(inputs)
            
            loss, metrics = compute_cnn_autoencoder_losses(
                inputs, outputs, config['loss'], dataset=train_dataset
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value * batch_size
        
        avg_metrics = {k: v/total_samples for k, v in total_metrics.items()}
        avg_loss = avg_metrics['total_loss']
        
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb_log = {
            'train/learning_rate': current_lr,
        }
        for key, value in avg_metrics.items():
            wandb_log[f'train/{key}'] = value
        
        run.log(wandb_log, step=epoch)
        
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"  Learning Rate: {current_lr:.6e}")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.6f}")
        print("-" * 50)
        
        if epoch % config['training']['eval_every'] == 0:
            print("🔍 Starting validation...")
            val_metrics = evaluate_model(model, test_loader, config['loss'])
            
            val_wandb_log = {}
            for key, value in val_metrics.items():
                val_wandb_log[f'val/{key}'] = value
            
            run.log(val_wandb_log, step=epoch)
            
            print(f"Validation results:")
            for key, value in val_metrics.items():
                print(f"  val_{key}: {value:.6f}")
            print("-" * 50)
        
        if epoch % 10 == 0:
            epoch_dir = os.path.join(visualization_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            print(f"Generating reconstruction visualization for epoch {epoch}...")
            visualize_reconstruction(model, train_loader, epoch_dir)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': config
            }, best_model_path)
            wandb.save(best_model_path)
            print(f"💾 Saved best model (Loss: {best_loss:.6f})")
    
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': avg_loss,
        'config': config
    }, final_model_path)
    wandb.save(final_model_path)
    
    print("Generating final reconstruction visualization...")
    final_viz_dir = os.path.join(visualization_dir, "final")
    os.makedirs(final_viz_dir, exist_ok=True)
    visualize_reconstruction(model, train_loader, final_viz_dir)
    
    run.log({
        'final_loss': avg_loss,
        'best_loss': best_loss,
        'total_epochs': epoch,
        'total_params': sum(p.numel() for p in model.parameters()),
        'dataset_size': len(train_dataset)
    })
    
    print("✅ CNN autoencoder training complete!")
    return model, best_loss


def evaluate_model(model, dataloader, loss_config):
    """
    Evaluate model performance on validation set.
    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        loss_config: Loss configuration
    Returns:
        dict: Dictionary containing various loss metrics
    """
    model.eval()
    total_metrics = {}
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].cuda()
            
            outputs = model(inputs)
            
            loss, metrics = compute_cnn_autoencoder_losses(
                inputs, outputs, loss_config, dataset=dataloader.dataset
            )
            
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value * batch_size
    
    avg_metrics = {k: v/total_samples for k, v in total_metrics.items()}
    return avg_metrics


def visualize_reconstruction(model, dataloader, output_dir, max_batches=None):
    """Visualize reconstruction results."""
    model.eval()
    
    total_samples = len(dataloader.dataset)
    target_samples = max(1, total_samples // 400)
    
    if max_batches is None:
        batch_size = dataloader.batch_size
        max_batches = max(1, target_samples // batch_size)
    
    print(f"📊 Visualization stats: total_samples={total_samples}, target_samples={target_samples}, max_batches={max_batches}")
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            inputs = batch['image'].cuda()
            outputs = model(inputs)
            reconstructions = outputs['reconstructed']
            
            save_comparison_images(
                inputs.cpu(),
                reconstructions.cpu(),
                output_dir,
                prefix=f"comparison_batch_{batch_idx}"
            )
            
            sample_count += inputs.size(0)
            
        print(f"✅ Generated {sample_count} reconstruction comparison images")


def main(config):
    """Main training function."""
    print("🎯 Starting CNN autoencoder training")
    print("🔧 Connecting to wandb...")

    return train_cnn_autoencoder(config)


if __name__ == '__main__':
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
            'use_resultant_loss': False,
            'force_lambda': 0.1,
            'moment_lambda': 0.05
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
