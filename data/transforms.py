"""
Data Augmentation Transforms for Medical Images
"""

import numpy as np
import random
from typing import Callable, List
import cv2


class RandomHorizontalFlip:
    """Randomly flip image horizontally"""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flipping
        """
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return np.fliplr(img).copy()
        return img


class RandomVerticalFlip:
    """Randomly flip image vertically"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return np.flipud(img).copy()
        return img


class RandomRotation:
    """Randomly rotate image by 90, 180, or 270 degrees"""
    
    def __init__(self, angles: List[int] = [0, 90, 180, 270]):
        """
        Args:
            angles: List of possible rotation angles
        """
        self.angles = angles
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        angle = random.choice(self.angles)
        
        if angle == 0:
            return img
        elif angle == 90:
            return np.rot90(img, k=1).copy()
        elif angle == 180:
            return np.rot90(img, k=2).copy()
        elif angle == 270:
            return np.rot90(img, k=3).copy()
        else:
            # Arbitrary angle rotation
            h, w = img.shape
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, matrix, (w, h), 
                                    borderMode=cv2.BORDER_REFLECT)
            return rotated


class RandomRotation90:
    """Randomly rotate by multiples of 90 degrees (faster)"""
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = random.randint(0, 3)  # 0, 90, 180, 270 degrees
        return np.rot90(img, k=k).copy()


class RandomCrop:
    """Randomly crop image to specified size"""
    
    def __init__(self, crop_size: int):
        self.crop_size = crop_size
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        
        if h < self.crop_size or w < self.crop_size:
            # Pad if image is smaller than crop size
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape
        
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        
        return img[top:top+self.crop_size, left:left+self.crop_size].copy()


class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            img = transform(img)
        return img


class ToTensor:
    """Convert numpy array to torch tensor and add channel dimension"""
    
    def __call__(self, img: np.ndarray):
        import torch
        # Add channel dimension if needed
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # (1, H, W)
        return torch.from_numpy(img.copy()).float()


# Predefined augmentation pipelines

def get_train_transforms(rotation: bool = True,
                         flip: bool = True) -> Compose:
    """
    Get standard training augmentation pipeline
    
    Args:
        rotation: Whether to include rotation
        flip: Whether to include flips
        
    Returns:
        Composed transform
    """
    transforms = []
    
    if flip:
        transforms.append(RandomHorizontalFlip(p=0.5))
        transforms.append(RandomVerticalFlip(p=0.5))
    
    if rotation:
        transforms.append(RandomRotation90())
    
    return Compose(transforms)


def get_val_transforms() -> Compose:
    """
    Get validation transforms (no augmentation)
    
    Returns:
        Identity transform (no-op)
    """
    return Compose([])


# Medical image specific augmentations

class RandomIntensityScale:
    """
    Randomly scale intensities (simulate contrast variation)
    Safe for normalized images in [0, 1]
    """
    
    def __init__(self, scale_range: tuple = (0.9, 1.1)):
        self.scale_range = scale_range
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        scale = random.uniform(*self.scale_range)
        return np.clip(img * scale, 0, 1).astype(np.float32)


class RandomIntensityShift:
    """
    Randomly shift intensities (simulate brightness variation)
    Safe for normalized images in [0, 1]
    """
    
    def __init__(self, shift_range: tuple = (-0.1, 0.1)):
        self.shift_range = shift_range
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        shift = random.uniform(*self.shift_range)
        return np.clip(img + shift, 0, 1).astype(np.float32)


def get_medical_train_transforms(include_intensity: bool = False) -> Compose:
    """
    Medical imaging specific augmentation pipeline
    
    Args:
        include_intensity: Whether to include intensity augmentations
        
    Returns:
        Composed transform
    """
    transforms = [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90()
    ]
    
    if include_intensity:
        transforms.extend([
            RandomIntensityScale(scale_range=(0.95, 1.05)),
            RandomIntensityShift(shift_range=(-0.05, 0.05))
        ])
    
    return Compose(transforms)


if __name__ == "__main__":
    # Test transforms
    print("Testing transforms...")
    
    # Create dummy image
    img = np.random.rand(256, 256).astype(np.float32)
    
    # Test individual transforms
    flip_h = RandomHorizontalFlip(p=1.0)
    flipped = flip_h(img)
    print(f"Horizontal flip: {flipped.shape}")
    
    flip_v = RandomVerticalFlip(p=1.0)
    flipped = flip_v(img)
    print(f"Vertical flip: {flipped.shape}")
    
    rotate = RandomRotation90()
    rotated = rotate(img)
    print(f"Rotation: {rotated.shape}")
    
    # Test composed transforms
    train_transform = get_train_transforms()
    augmented = train_transform(img)
    print(f"Augmented: {augmented.shape}")
    
    # Test medical transforms
    medical_transform = get_medical_train_transforms(include_intensity=True)
    augmented = medical_transform(img)
    print(f"Medical augmented: {augmented.shape}")
    print(f"Value range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    print("\nAll transforms working correctly!")
