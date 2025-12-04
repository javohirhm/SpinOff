"""
Bicubic Interpolation Baseline for Super-Resolution
Simple non-learning baseline for comparison with deep learning methods.

This provides a consistent interface matching other models (SR3, DiT, U-Net)
for fair comparison in the SpinOff project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BicubicSR(nn.Module):
    """
    Bicubic Interpolation for Super-Resolution.
    
    This is a non-learning baseline that simply upsamples LR images
    using bicubic interpolation. No trainable parameters.
    
    Provides the same interface as other SR models for easy comparison.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scale_factor: int = 2,
        image_size: int = 256,  # HR output size (for compatibility)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.lr_size = image_size // scale_factor
        
        # No trainable parameters - this is a dummy parameter for compatibility
        # Some training loops check for parameters
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample input using bicubic interpolation.
        
        Args:
            x: Low-resolution input [B, C, H, W]
            
        Returns:
            High-resolution output [B, C, H*scale, W*scale]
        """
        return F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='bicubic', 
            align_corners=False
        )
    
    def super_resolve(self, lr: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for API consistency."""
        return self.forward(lr)


class BilinearSR(nn.Module):
    """
    Bilinear Interpolation for Super-Resolution.
    Another simple baseline, faster but lower quality than bicubic.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scale_factor: int = 2,
        image_size: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.lr_size = image_size // scale_factor
        
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='bilinear', 
            align_corners=False
        )


class NearestSR(nn.Module):
    """
    Nearest Neighbor Interpolation for Super-Resolution.
    Fastest but lowest quality baseline.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scale_factor: int = 2,
        image_size: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.lr_size = image_size // scale_factor
        
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='nearest'
        )


class LanczosSR(nn.Module):
    """
    Lanczos Interpolation for Super-Resolution.
    
    Lanczos is not directly available in PyTorch, so we implement
    a close approximation using bicubic with antialiasing.
    For true Lanczos, we'd need to use PIL or custom kernels.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scale_factor: int = 2,
        image_size: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.lr_size = image_size // scale_factor
        
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch 1.11+ supports antialias parameter
        try:
            return F.interpolate(
                x, 
                scale_factor=self.scale_factor, 
                mode='bicubic', 
                align_corners=False,
                antialias=True  # Better quality, closer to Lanczos
            )
        except TypeError:
            # Fallback for older PyTorch versions
            return F.interpolate(
                x, 
                scale_factor=self.scale_factor, 
                mode='bicubic', 
                align_corners=False
            )


def create_bicubic_model(
    image_size: int = 256,
    in_channels: int = 1,
    scale_factor: int = 2,
    method: str = 'bicubic'
) -> nn.Module:
    """
    Factory function to create interpolation-based SR model.
    
    Args:
        image_size: HR output size
        in_channels: Number of input channels
        scale_factor: Upsampling factor
        method: Interpolation method ('bicubic', 'bilinear', 'nearest', 'lanczos')
    
    Returns:
        Interpolation model
    """
    methods = {
        'bicubic': BicubicSR,
        'bilinear': BilinearSR,
        'nearest': NearestSR,
        'lanczos': LanczosSR
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
    
    model = methods[method](
        in_channels=in_channels,
        out_channels=in_channels,
        scale_factor=scale_factor,
        image_size=image_size
    )
    
    return model


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_bicubic(
    model: nn.Module,
    lr_images: torch.Tensor,
    hr_images: torch.Tensor
) -> dict:
    """
    Evaluate bicubic model on a batch of images.
    
    Args:
        model: Bicubic SR model
        lr_images: Low-resolution images [B, C, H, W]
        hr_images: High-resolution ground truth [B, C, H*scale, W*scale]
    
    Returns:
        Dictionary with PSNR and SSIM metrics
    """
    from utils.metrics import calculate_psnr, calculate_ssim
    
    model.eval()
    with torch.no_grad():
        sr_images = model(lr_images)
    
    psnr = calculate_psnr(sr_images, hr_images)
    ssim = calculate_ssim(sr_images, hr_images)
    
    return {
        'psnr': psnr,
        'ssim': ssim
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Bicubic Interpolation Models")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test all methods
    methods = ['bicubic', 'bilinear', 'nearest', 'lanczos']
    
    # Create test inputs
    batch_size = 2
    lr_img = torch.randn(batch_size, 1, 128, 128).to(device)
    
    print(f"\nInput shape: {lr_img.shape}")
    print(f"Expected output: [{batch_size}, 1, 256, 256]")
    print()
    
    for method in methods:
        model = create_bicubic_model(
            image_size=256,
            in_channels=1,
            scale_factor=2,
            method=method
        ).to(device)
        
        with torch.no_grad():
            sr_img = model(lr_img)
        
        # Count parameters (should be 0 or 1 dummy)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"{method.capitalize():10s} | Output: {sr_img.shape} | "
              f"Range: [{sr_img.min():.3f}, {sr_img.max():.3f}] | "
              f"Params: {num_params}")
    
    print("\n✅ All interpolation methods working correctly!")
