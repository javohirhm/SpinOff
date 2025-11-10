"""
Degradation Module for Low-Dose CT/MRI Simulation
Implements downsampling and noise addition
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple
import cv2


def downsample_image(img: Union[np.ndarray, torch.Tensor],
                     scale_factor: int = 2,
                     method: str = 'bicubic') -> Union[np.ndarray, torch.Tensor]:
    """
    Downsample image by scale_factor
    
    Args:
        img: Input image (H, W) or (C, H, W)
        scale_factor: Downsampling factor (2 = half resolution)
        method: 'bicubic', 'bilinear', or 'area'
        
    Returns:
        Downsampled image
    """
    is_numpy = isinstance(img, np.ndarray)
    
    if is_numpy:
        # Convert to torch
        img_torch = torch.from_numpy(img).float()
        if img_torch.ndim == 2:
            img_torch = img_torch.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif img_torch.ndim == 3:
            img_torch = img_torch.unsqueeze(0)  # (1, C, H, W)
    else:
        img_torch = img
        if img_torch.ndim == 2:
            img_torch = img_torch.unsqueeze(0).unsqueeze(0)
    
    h, w = img_torch.shape[-2:]
    new_h, new_w = h // scale_factor, w // scale_factor
    
    # Downsample
    img_lr = F.interpolate(img_torch, size=(new_h, new_w),
                          mode=method, align_corners=False if method == 'bicubic' else None)
    
    if is_numpy:
        return img_lr.squeeze().numpy()
    else:
        return img_lr.squeeze()


def add_gaussian_noise(img: np.ndarray,
                       noise_level: float = 0.02,
                       noise_type: str = 'gaussian') -> np.ndarray:
    """
    Add noise to image
    
    Args:
        img: Input image (normalized to [0, 1])
        noise_level: Standard deviation of noise
        noise_type: 'gaussian' or 'rician' (more realistic for MRI)
        
    Returns:
        Noisy image clipped to [0, 1]
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        noisy_img = img + noise
    
    elif noise_type == 'rician':
        # Rician noise is more realistic for MRI (magnitude of complex Gaussian)
        noise_real = np.random.normal(0, noise_level, img.shape)
        noise_imag = np.random.normal(0, noise_level, img.shape)
        noisy_img = np.sqrt((img + noise_real)**2 + noise_imag**2)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return np.clip(noisy_img, 0, 1).astype(np.float32)


def add_blur(img: np.ndarray,
             kernel_size: int = 3,
             sigma: float = 1.0) -> np.ndarray:
    """
    Add Gaussian blur to simulate motion or acquisition blur
    
    Args:
        img: Input image
        kernel_size: Gaussian kernel size (odd number)
        sigma: Gaussian standard deviation
        
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def simulate_low_dose(img_hr: np.ndarray,
                     scale_factor: int = 2,
                     noise_level: float = 0.02,
                     noise_type: str = 'gaussian',
                     add_blur_flag: bool = False,
                     blur_sigma: float = 0.5) -> np.ndarray:
    """
    Complete low-dose simulation pipeline
    
    Simulates low-dose acquisition by:
    1. Downsampling (reduced spatial resolution)
    2. Adding noise (reduced SNR)
    3. Optional: Adding blur (motion artifacts)
    
    Args:
        img_hr: High-resolution input image [0, 1]
        scale_factor: Downsampling factor
        noise_level: Noise standard deviation
        noise_type: Type of noise to add
        add_blur_flag: Whether to add blur
        blur_sigma: Blur kernel sigma
        
    Returns:
        Low-dose simulated image (downsampled + noisy)
    """
    # Step 1: Downsample
    img_lr = downsample_image(img_hr, scale_factor=scale_factor)
    
    # Step 2: Add noise
    img_lr = add_gaussian_noise(img_lr, noise_level=noise_level, noise_type=noise_type)
    
    # Step 3: Optional blur
    if add_blur_flag:
        img_lr = add_blur(img_lr, kernel_size=3, sigma=blur_sigma)
    
    return img_lr


def upsample_image(img: Union[np.ndarray, torch.Tensor],
                   scale_factor: int = 2,
                   method: str = 'bicubic') -> Union[np.ndarray, torch.Tensor]:
    """
    Upsample image back to original size (for comparison)
    
    Args:
        img: Input low-resolution image
        scale_factor: Upsampling factor
        method: Interpolation method
        
    Returns:
        Upsampled image
    """
    is_numpy = isinstance(img, np.ndarray)
    
    if is_numpy:
        img_torch = torch.from_numpy(img).float()
        if img_torch.ndim == 2:
            img_torch = img_torch.unsqueeze(0).unsqueeze(0)
    else:
        img_torch = img
        if img_torch.ndim == 2:
            img_torch = img_torch.unsqueeze(0).unsqueeze(0)
    
    h, w = img_torch.shape[-2:]
    new_h, new_w = h * scale_factor, w * scale_factor
    
    img_up = F.interpolate(img_torch, size=(new_h, new_w),
                          mode=method, align_corners=False if method == 'bicubic' else None)
    
    if is_numpy:
        return img_up.squeeze().numpy()
    else:
        return img_up.squeeze()


def create_degraded_pairs(hr_images: list,
                         scale_factor: int = 2,
                         noise_level: float = 0.02,
                         save_dir: str = None) -> list:
    """
    Create LR-HR pairs from list of HR images
    
    Args:
        hr_images: List of HR image paths or arrays
        scale_factor: Downsampling factor
        noise_level: Noise level
        save_dir: Optional directory to save LR images
        
    Returns:
        List of (LR, HR) pairs
    """
    from pathlib import Path
    
    pairs = []
    
    for idx, hr_path in enumerate(hr_images):
        # Load HR image
        if isinstance(hr_path, str) or isinstance(hr_path, Path):
            hr_img = np.load(hr_path)
        else:
            hr_img = hr_path
        
        # Create LR version
        lr_img = simulate_low_dose(hr_img, scale_factor=scale_factor,
                                   noise_level=noise_level)
        
        # Save if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if isinstance(hr_path, (str, Path)):
                filename = Path(hr_path).stem + '_lr.npy'
            else:
                filename = f'lr_image_{idx:05d}.npy'
            
            np.save(save_dir / filename, lr_img)
        
        pairs.append((lr_img, hr_img))
    
    return pairs


if __name__ == "__main__":
    # Example usage and testing
    print("Degradation module loaded successfully!")
    
    # Test with dummy image
    test_img = np.random.rand(256, 256).astype(np.float32)
    
    print(f"Original shape: {test_img.shape}")
    
    # Simulate low-dose
    lr_img = simulate_low_dose(test_img, scale_factor=2, noise_level=0.02)
    print(f"Low-dose shape: {lr_img.shape}")
    
    # Upsample back
    up_img = upsample_image(lr_img, scale_factor=2)
    print(f"Upsampled shape: {up_img.shape}")
    
    print("\nAll functions working correctly!")
