"""
PyTorch Dataset for MRI Super-Resolution
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import random

from .degradation import simulate_low_dose


class MRIDataset(Dataset):
    """
    Dataset for MRI super-resolution
    
    Can work in two modes:
    1. Pre-computed LR images (faster, requires storage)
    2. On-the-fly LR generation (slower, saves storage)
    """
    
    def __init__(self,
                 hr_paths: List[str],
                 lr_paths: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 create_lr_on_fly: bool = False,
                 scale_factor: int = 2,
                 noise_level: float = 0.02,
                 noise_type: str = 'gaussian'):
        """
        Args:
            hr_paths: List of paths to high-resolution images
            lr_paths: List of paths to low-resolution images (if pre-computed)
            transform: Optional transform to apply to both LR and HR
            create_lr_on_fly: If True, generate LR from HR during loading
            scale_factor: Downsampling factor for LR generation
            noise_level: Noise level for LR generation
            noise_type: Type of noise ('gaussian' or 'rician')
        """
        self.hr_paths = hr_paths
        self.lr_paths = lr_paths
        self.transform = transform
        self.create_lr_on_fly = create_lr_on_fly
        self.scale_factor = scale_factor
        self.noise_level = noise_level
        self.noise_type = noise_type
        
        if not create_lr_on_fly and lr_paths is None:
            raise ValueError("Must provide lr_paths if not creating LR on-the-fly")
        
        if lr_paths is not None and len(lr_paths) != len(hr_paths):
            raise ValueError("Number of LR and HR images must match")
    
    def __len__(self) -> int:
        return len(self.hr_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (lr_image, hr_image): Both as torch.Tensor of shape (1, H, W)
        """
        # Load HR image
        hr_img = np.load(self.hr_paths[idx]).astype(np.float32)
        
        # Get or create LR image
        if self.create_lr_on_fly:
            lr_img = simulate_low_dose(hr_img, 
                                      scale_factor=self.scale_factor,
                                      noise_level=self.noise_level,
                                      noise_type=self.noise_type)
        else:
            lr_img = np.load(self.lr_paths[idx]).astype(np.float32)
        
        # Apply transforms (augmentation)
        if self.transform:
            # Apply same transform to both images
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            np.random.seed(seed)
            lr_img = self.transform(lr_img)
            
            random.seed(seed)
            np.random.seed(seed)
            hr_img = self.transform(hr_img)
        
        # Convert to tensors with channel dimension
        lr_tensor = torch.from_numpy(lr_img).unsqueeze(0)  # (1, H, W)
        hr_tensor = torch.from_numpy(hr_img).unsqueeze(0)  # (1, H, W)
        
        return lr_tensor, hr_tensor


class PatchDataset(Dataset):
    """
    Dataset that extracts random patches from images
    Useful for training on high-resolution images
    """
    
    def __init__(self,
                 hr_paths: List[str],
                 patch_size: int = 128,
                 patches_per_image: int = 4,
                 scale_factor: int = 2,
                 noise_level: float = 0.02,
                 transform: Optional[Callable] = None):
        """
        Args:
            hr_paths: List of paths to HR images
            patch_size: Size of patches to extract from HR
            patches_per_image: Number of patches per image per epoch
            scale_factor: SR scale factor
            noise_level: Degradation noise level
            transform: Optional augmentation
        """
        self.hr_paths = hr_paths
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.scale_factor = scale_factor
        self.noise_level = noise_level
        self.transform = transform
        self.lr_patch_size = patch_size // scale_factor
    
    def __len__(self) -> int:
        return len(self.hr_paths) * self.patches_per_image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine which image and which patch
        img_idx = idx // self.patches_per_image
        
        # Load HR image
        hr_img = np.load(self.hr_paths[img_idx]).astype(np.float32)
        
        # Extract random patch from HR
        h, w = hr_img.shape
        if h < self.patch_size or w < self.patch_size:
            # Pad if image is too small
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            hr_img = np.pad(hr_img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = hr_img.shape
        
        # Random crop
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        hr_patch = hr_img[top:top+self.patch_size, left:left+self.patch_size]
        
        # Create LR patch
        lr_patch = simulate_low_dose(hr_patch,
                                     scale_factor=self.scale_factor,
                                     noise_level=self.noise_level)
        
        # Apply transforms
        if self.transform:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            lr_patch = self.transform(lr_patch)
            random.seed(seed)
            hr_patch = self.transform(hr_patch)
        
        # Convert to tensors
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)
        
        return lr_tensor, hr_tensor


def get_dataloaders(train_hr_paths: List[str],
                   val_hr_paths: List[str],
                   batch_size: int = 8,
                   num_workers: int = 2,
                   scale_factor: int = 2,
                   noise_level: float = 0.02,
                   train_transform: Optional[Callable] = None,
                   use_patches: bool = False,
                   patch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_hr_paths: Training HR image paths
        val_hr_paths: Validation HR image paths
        batch_size: Batch size
        num_workers: Number of data loading workers
        scale_factor: Super-resolution scale factor
        noise_level: Degradation noise level
        train_transform: Augmentation for training
        use_patches: Whether to use patch-based training
        patch_size: Patch size if using patches
        
    Returns:
        (train_loader, val_loader)
    """
    if use_patches:
        train_dataset = PatchDataset(
            train_hr_paths,
            patch_size=patch_size,
            patches_per_image=4,
            scale_factor=scale_factor,
            noise_level=noise_level,
            transform=train_transform
        )
    else:
        train_dataset = MRIDataset(
            train_hr_paths,
            create_lr_on_fly=True,
            scale_factor=scale_factor,
            noise_level=noise_level,
            transform=train_transform
        )
    
    # Validation without augmentation
    val_dataset = MRIDataset(
        val_hr_paths,
        create_lr_on_fly=True,
        scale_factor=scale_factor,
        noise_level=noise_level,
        transform=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    print("Dataset module loaded successfully!")
    
    # Test with dummy data
    import tempfile
    import os
    
    # Create temp directory with dummy images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some dummy images
        for i in range(5):
            dummy_img = np.random.rand(256, 256).astype(np.float32)
            np.save(os.path.join(tmpdir, f'test_{i}.npy'), dummy_img)
        
        # Get paths
        paths = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
        
        # Create dataset
        dataset = MRIDataset(paths, create_lr_on_fly=True, scale_factor=2)
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading
        lr, hr = dataset[0]
        print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
        
        # Test dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        for batch_lr, batch_hr in loader:
            print(f"Batch LR: {batch_lr.shape}, Batch HR: {batch_hr.shape}")
            break
        
        print("\nDataset working correctly!")
