import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_comparison_images(original_images, reconstructed_images, save_path, prefix="comparison"):
    """
    Save comparison images of original and reconstructed tactile force data.
    Args:
        original_images: shape (batch_size, 3, H, W)
        reconstructed_images: shape (batch_size, 3, H, W)
    """
    if isinstance(original_images, torch.Tensor):
        original_images = original_images.detach().cpu().numpy()
    if isinstance(reconstructed_images, torch.Tensor):
        reconstructed_images = reconstructed_images.detach().cpu().numpy()
    
    batch_size = original_images.shape[0]
    
    for i in range(batch_size):
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        
        orig_x = original_images[i, 0]
        orig_y = original_images[i, 1]
        orig_z = original_images[i, 2]
        
        recon_x = reconstructed_images[i, 0]
        recon_y = reconstructed_images[i, 1]
        recon_z = reconstructed_images[i, 2]
        
        # Compute unified color range
        max_abs_value = max(
            np.max(np.abs(orig_x)), np.max(np.abs(orig_y)), np.max(np.abs(orig_z)),
            np.max(np.abs(recon_x)), np.max(np.abs(recon_y)), np.max(np.abs(recon_z))
        )
        vmin, vmax = -max_abs_value, max_abs_value
        
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
        
        diff_x = orig_x - recon_x
        diff_y = orig_y - recon_y
        diff_z = orig_z - recon_z
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
        
        mse_x = np.mean((orig_x - recon_x) ** 2)
        mse_y = np.mean((orig_y - recon_y) ** 2)
        mse_z = np.mean((orig_z - recon_z) ** 2)
        total_mse = np.mean((original_images[i] - reconstructed_images[i]) ** 2)
        
        axes[1, 3].axis('off')
        axes[1, 4].axis('off')
        axes[1, 5].axis('off')
        
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
        
        plt.colorbar(im_orig_z, ax=axes[0, 2], fraction=0.046, pad=0.04)
        plt.colorbar(im_recon_z, ax=axes[1, 2], fraction=0.046, pad=0.04)
        plt.colorbar(im_diff_z, ax=axes[0, 5], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        save_file = os.path.join(save_path, f"{prefix}_{i}.png")
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        plt.close()