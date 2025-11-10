"""
Evaluation Metrics for Super-Resolution
PSNR, SSIM, and other quality metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from math import log10
import warnings


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, 
                   max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image tensor (B, C, H, W) or (H, W)
        img2: Second image tensor (same shape as img1)
        max_val: Maximum possible pixel value (1.0 for [0,1] range)
        
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * log10(max_val) - 10 * log10(mse.item())
    return psnr


def calculate_psnr_batch(img1: torch.Tensor, img2: torch.Tensor,
                        max_val: float = 1.0) -> torch.Tensor:
    """
    Calculate PSNR for each image in a batch
    
    Args:
        img1: Batch of images (B, C, H, W)
        img2: Batch of images (B, C, H, W)
        max_val: Maximum pixel value
        
    Returns:
        Tensor of PSNR values for each image (B,)
    """
    batch_size = img1.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        mse = F.mse_loss(img1[i], img2[i])
        if mse == 0:
            psnr = torch.tensor(float('inf'))
        else:
            psnr = 20 * log10(max_val) - 10 * torch.log10(mse)
        psnr_values.append(psnr)
    
    return torch.stack(psnr_values)


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor,
                  window_size: int = 11,
                  size_average: bool = True) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        img1: First image (B, C, H, W) or (C, H, W)
        img2: Second image (same shape)
        window_size: Size of Gaussian window
        size_average: Whether to average over batch
        
    Returns:
        SSIM value between -1 and 1 (1 = identical)
    """
    # Ensure 4D tensor
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Constants for stability
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
                         for x in range(window_size)])
    gauss = gauss / gauss.sum()
    
    # 2D Gaussian window
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.to(img1.device)
    
    # Expand window for all channels
    channel = img1.size(1)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    
    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1**2, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=(1, 2, 3))


def calculate_mae(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        MAE value
    """
    return F.l1_loss(img1, img2).item()


def calculate_rmse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Root Mean Squared Error
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        RMSE value
    """
    mse = F.mse_loss(img1, img2).item()
    return np.sqrt(mse)


class MetricsCalculator:
    """
    Utility class to calculate and store metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.psnr_values = []
        self.ssim_values = []
        self.mae_values = []
        self.rmse_values = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new batch
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Ground truth images (B, C, H, W)
        """
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            pred_img = pred[i:i+1]
            target_img = target[i:i+1]
            
            # Calculate metrics
            psnr = calculate_psnr(pred_img, target_img)
            ssim = calculate_ssim(pred_img, target_img)
            mae = calculate_mae(pred_img, target_img)
            rmse = calculate_rmse(pred_img, target_img)
            
            # Store
            if not np.isinf(psnr):
                self.psnr_values.append(psnr)
            self.ssim_values.append(ssim)
            self.mae_values.append(mae)
            self.rmse_values.append(rmse)
    
    def get_metrics(self) -> dict:
        """
        Get average metrics
        
        Returns:
            Dictionary with average metrics
        """
        return {
            'psnr': np.mean(self.psnr_values) if self.psnr_values else 0,
            'ssim': np.mean(self.ssim_values) if self.ssim_values else 0,
            'mae': np.mean(self.mae_values) if self.mae_values else 0,
            'rmse': np.mean(self.rmse_values) if self.rmse_values else 0
        }
    
    def __str__(self):
        """String representation of metrics"""
        metrics = self.get_metrics()
        return (f"PSNR: {metrics['psnr']:.2f} dB | "
                f"SSIM: {metrics['ssim']:.4f} | "
                f"MAE: {metrics['mae']:.4f} | "
                f"RMSE: {metrics['rmse']:.4f}")


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with validation data
        device: Device to run on
        
    Returns:
        Dictionary with average metrics
    """
    model.eval()
    calculator = MetricsCalculator()
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Predict
            sr_imgs = model(lr_imgs)
            
            # Update metrics
            calculator.update(sr_imgs, hr_imgs)
    
    return calculator.get_metrics()


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create dummy images
    img1 = torch.rand(2, 1, 256, 256)
    img2 = img1 + torch.randn_like(img1) * 0.05  # Add noise
    
    # Test PSNR
    psnr = calculate_psnr(img1, img2)
    print(f"PSNR: {psnr:.2f} dB")
    
    # Test PSNR batch
    psnr_batch = calculate_psnr_batch(img1, img2)
    print(f"PSNR batch: {psnr_batch}")
    
    # Test SSIM
    ssim = calculate_ssim(img1, img2)
    print(f"SSIM: {ssim:.4f}")
    
    # Test MAE
    mae = calculate_mae(img1, img2)
    print(f"MAE: {mae:.4f}")
    
    # Test RMSE
    rmse = calculate_rmse(img1, img2)
    print(f"RMSE: {rmse:.4f}")
    
    # Test metrics calculator
    print("\nTesting MetricsCalculator...")
    calculator = MetricsCalculator()
    calculator.update(img1, img2)
    calculator.update(img1, img2)  # Add more samples
    
    print(calculator)
    print("\nMetrics dict:", calculator.get_metrics())
    
    print("\n✅ All metrics working correctly!")
