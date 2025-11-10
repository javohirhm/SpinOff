"""
Data Preprocessing Module for IXI MRI Dataset
Handles NIfTI loading, normalization, and slice extraction
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_nifti(file_path: str) -> np.ndarray:
    """
    Load NIfTI file and return numpy array
    
    Args:
        file_path: Path to .nii or .nii.gz file
        
    Returns:
        3D numpy array of MRI volume
    """
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def normalize_intensity(img: np.ndarray, 
                        method: str = 'minmax',
                        clip_percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Normalize image intensities
    
    Args:
        img: Input image array
        method: 'minmax' or 'zscore'
        clip_percentiles: Optional (low, high) percentiles for clipping outliers
        
    Returns:
        Normalized image in range [0, 1]
    """
    # Optional: Clip extreme values
    if clip_percentiles:
        low_perc, high_perc = clip_percentiles
        low_val = np.percentile(img, low_perc)
        high_val = np.percentile(img, high_perc)
        img = np.clip(img, low_val, high_val)
    
    if method == 'minmax':
        img_min, img_max = img.min(), img.max()
        if img_max - img_min < 1e-8:
            return np.zeros_like(img)
        return (img - img_min) / (img_max - img_min)
    
    elif method == 'zscore':
        mean, std = img.mean(), img.std()
        if std < 1e-8:
            return np.zeros_like(img)
        normalized = (img - mean) / std
        # Clip to [-3, 3] and scale to [0, 1]
        normalized = np.clip(normalized, -3, 3)
        return (normalized + 3) / 6
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_slices(volume: np.ndarray, 
                   axis: int = 2,
                   skip_empty: bool = True,
                   empty_threshold: float = 0.01) -> List[np.ndarray]:
    """
    Extract 2D slices from 3D volume
    
    Args:
        volume: 3D numpy array (H, W, D)
        axis: Axis along which to slice (0, 1, or 2)
        skip_empty: Whether to skip mostly empty slices
        empty_threshold: Minimum mean intensity to keep slice
        
    Returns:
        List of 2D slices
    """
    slices = []
    
    for i in range(volume.shape[axis]):
        if axis == 0:
            slice_2d = volume[i, :, :]
        elif axis == 1:
            slice_2d = volume[:, i, :]
        else:  # axis == 2
            slice_2d = volume[:, :, i]
        
        # Skip empty or nearly empty slices
        if skip_empty and slice_2d.mean() < empty_threshold:
            continue
            
        slices.append(slice_2d)
    
    return slices


def resize_slice(slice_2d: np.ndarray, 
                 target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Resize slice to target size using interpolation
    
    Args:
        slice_2d: Input 2D slice
        target_size: (height, width)
        
    Returns:
        Resized slice
    """
    from scipy.ndimage import zoom
    
    h, w = slice_2d.shape
    target_h, target_w = target_size
    
    zoom_factors = (target_h / h, target_w / w)
    resized = zoom(slice_2d, zoom_factors, order=1)  # Bilinear interpolation
    
    return resized


def preprocess_pipeline(nifti_path: str,
                       output_dir: str,
                       subject_id: str,
                       normalize_method: str = 'minmax',
                       target_size: Tuple[int, int] = (256, 256),
                       axis: int = 2) -> int:
    """
    Complete preprocessing pipeline for one subject
    
    Args:
        nifti_path: Path to input NIfTI file
        output_dir: Directory to save processed slices
        subject_id: Subject identifier for naming
        normalize_method: Normalization method
        target_size: Target slice size
        axis: Slicing axis
        
    Returns:
        Number of slices saved
    """
    logger.info(f"Processing {subject_id}...")
    
    # Load volume
    volume = load_nifti(nifti_path)
    
    # Normalize
    volume = normalize_intensity(volume, method=normalize_method, 
                                 clip_percentiles=(1, 99))
    
    # Extract slices
    slices = extract_slices(volume, axis=axis, skip_empty=True)
    
    # Save slices
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for idx, slice_2d in enumerate(slices):
        # Resize to target size
        slice_2d = resize_slice(slice_2d, target_size)
        
        # Save
        filename = f"{subject_id}_slice{idx:03d}.npy"
        np.save(output_dir / filename, slice_2d.astype(np.float32))
        saved_count += 1
    
    logger.info(f"Saved {saved_count} slices for {subject_id}")
    return saved_count


def create_splits(data_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42) -> dict:
    """
    Create train/val/test splits from processed data
    
    Args:
        data_dir: Directory with processed .npy files
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists of file paths
    """
    np.random.seed(seed)
    
    data_dir = Path(data_dir)
    all_files = sorted(list(data_dir.glob("*.npy")))
    
    # Group by subject
    subjects = {}
    for file_path in all_files:
        subject_id = file_path.stem.split('_slice')[0]
        if subject_id not in subjects:
            subjects[subject_id] = []
        subjects[subject_id].append(file_path)
    
    # Shuffle subjects
    subject_ids = list(subjects.keys())
    np.random.shuffle(subject_ids)
    
    # Split
    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    
    train_subjects = subject_ids[:n_train]
    val_subjects = subject_ids[n_train:n_train + n_val]
    test_subjects = subject_ids[n_train + n_val:]
    
    splits = {
        'train': [f for s in train_subjects for f in subjects[s]],
        'val': [f for s in val_subjects for f in subjects[s]],
        'test': [f for s in test_subjects for f in subjects[s]]
    }
    
    logger.info(f"Split: {len(splits['train'])} train, "
                f"{len(splits['val'])} val, {len(splits['test'])} test")
    
    return splits


if __name__ == "__main__":
    # Example usage
    print("Preprocessing module loaded successfully!")
    print("Use preprocess_pipeline() to process individual subjects")
    print("Use create_splits() to create train/val/test splits")
