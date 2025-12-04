"""
Evaluation script for Bicubic Interpolation Baseline.

Since bicubic has no trainable parameters, this script simply
evaluates the method on the dataset and saves results for comparison.

This follows the same structure as train_sr3.py, train_dit.py, etc.
for consistency in the SpinOff project.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add repo root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import MRIDataset
from models.bicubic import create_bicubic_model
from utils.metrics import calculate_psnr, calculate_ssim


# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SpinOff Bicubic Baseline Evaluation")

    p.add_argument("--splits", required=True, help="Path to splits JSON file")
    p.add_argument("--save_dir", type=str, default="./results/bicubic", 
                   help="Directory to save results")
    p.add_argument("--device", type=str, 
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # Data arguments
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    p.add_argument("--create_lr_on_fly", action="store_true", 
                   help="Create LR images on-the-fly")
    p.add_argument("--scale", type=int, default=2, help="Downsampling scale factor")
    p.add_argument("--noise_level", type=float, default=0.02, 
                   help="Noise level for degradation")

    # Model arguments
    p.add_argument("--image_size", type=int, default=256, help="High-res image size")
    p.add_argument("--method", type=str, default="bicubic",
                   choices=["bicubic", "bilinear", "nearest", "lanczos"],
                   help="Interpolation method")
    
    # Evaluation arguments
    p.add_argument("--eval_train", action="store_true", 
                   help="Also evaluate on training set")
    p.add_argument("--save_samples", type=int, default=10,
                   help="Number of sample images to save")

    return p.parse_args()


# ---------------------------------------------------------------
def make_dataloaders(splits_file, batch_size, create_lr_on_fly, scale_factor, noise_level):
    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_paths = splits.get("train", [])
    val_paths = splits.get("val", [])
    test_paths = splits.get("test", [])

    def create_loader(paths, shuffle=False):
        if not paths:
            return None
        ds = MRIDataset(
            hr_paths=paths,
            lr_paths=None,
            transform=None,
            create_lr_on_fly=create_lr_on_fly,
            scale_factor=scale_factor,
            noise_level=noise_level
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=2, pin_memory=True)

    return {
        'train': create_loader(train_paths),
        'val': create_loader(val_paths),
        'test': create_loader(test_paths)
    }


# ---------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating"):
    """
    Evaluate bicubic model on a dataloader.
    
    Returns:
        Dictionary with mean PSNR, SSIM, and lists of per-sample metrics
    """
    model.eval()
    
    all_psnr = []
    all_ssim = []
    
    for lr_img, hr_img in tqdm(loader, desc=desc, leave=False):
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        
        # Super-resolve using bicubic
        sr_img = model(lr_img)
        
        # Ensure same size (should already be, but safety check)
        if sr_img.shape != hr_img.shape:
            sr_img = F.interpolate(sr_img, size=hr_img.shape[-2:], 
                                   mode='bicubic', align_corners=False)
        
        # Calculate metrics for each image in batch
        batch_psnr = calculate_psnr(sr_img, hr_img)
        batch_ssim = calculate_ssim(sr_img, hr_img)
        
        all_psnr.append(batch_psnr)
        all_ssim.append(batch_ssim)
    
    # Aggregate
    mean_psnr = float(torch.tensor(all_psnr).mean())
    mean_ssim = float(torch.tensor(all_ssim).mean())
    std_psnr = float(torch.tensor(all_psnr).std())
    std_ssim = float(torch.tensor(all_ssim).std())
    
    return {
        'psnr_mean': mean_psnr,
        'psnr_std': std_psnr,
        'ssim_mean': mean_ssim,
        'ssim_std': std_ssim,
        'psnr_all': [float(x) for x in all_psnr],
        'ssim_all': [float(x) for x in all_ssim],
        'num_samples': len(all_psnr) * loader.batch_size
    }


# ---------------------------------------------------------------
@torch.no_grad()
def save_sample_images(model, loader, device, save_dir, num_samples=10):
    """Save sample LR, SR, HR comparison images."""
    import numpy as np
    from PIL import Image
    
    samples_dir = os.path.join(save_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    model.eval()
    saved = 0
    
    for lr_img, hr_img in loader:
        if saved >= num_samples:
            break
            
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        sr_img = model(lr_img)
        
        # Save each image in batch
        for i in range(min(lr_img.shape[0], num_samples - saved)):
            # Convert to numpy
            lr_np = lr_img[i, 0].cpu().numpy()
            sr_np = sr_img[i, 0].cpu().numpy()
            hr_np = hr_img[i, 0].cpu().numpy()
            
            # Normalize to 0-255
            def to_uint8(img):
                img = np.clip(img, 0, 1)
                return (img * 255).astype(np.uint8)
            
            # Upsample LR for visual comparison
            lr_up = F.interpolate(
                lr_img[i:i+1], scale_factor=model.scale_factor, 
                mode='bicubic', align_corners=False
            )[0, 0].cpu().numpy()
            
            # Create comparison image (LR | SR | HR)
            comparison = np.concatenate([
                to_uint8(lr_up),
                to_uint8(sr_np),
                to_uint8(hr_np)
            ], axis=1)
            
            # Save
            img = Image.fromarray(comparison, mode='L')
            img.save(os.path.join(samples_dir, f"sample_{saved:03d}.png"))
            
            saved += 1
            if saved >= num_samples:
                break
    
    print(f"✅ Saved {saved} sample images to {samples_dir}")


# ---------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    print("=" * 60)
    print("SpinOff Bicubic Baseline Evaluation")
    print("=" * 60)
    print(f"📂 Using device: {device}")
    print(f"🔹 Loading splits: {args.splits}")
    print(f"🔹 HR image size: {args.image_size}")
    print(f"🔹 Scale factor: {args.scale}x")
    print(f"🔹 Method: {args.method}")
    print("=" * 60)

    # Create data loaders
    loaders = make_dataloaders(
        splits_file=args.splits,
        batch_size=args.batch_size,
        create_lr_on_fly=args.create_lr_on_fly,
        scale_factor=args.scale,
        noise_level=args.noise_level
    )
    
    for split, loader in loaders.items():
        if loader:
            print(f"{split.capitalize()} batches: {len(loader)}")

    # Create model
    model = create_bicubic_model(
        image_size=args.image_size,
        in_channels=1,
        scale_factor=args.scale,
        method=args.method
    ).to(device)
    
    print(f"\n✅ Using {args.method} interpolation (no trainable parameters)")

    # Results dictionary
    results = {
        'method': args.method,
        'scale_factor': args.scale,
        'image_size': args.image_size,
        'noise_level': args.noise_level,
        'timestamp': datetime.now().isoformat(),
        'metrics': {}
    }

    # Evaluate on each split
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    for split, loader in loaders.items():
        if loader is None:
            continue
        if split == 'train' and not args.eval_train:
            continue
        
        print(f"\nEvaluating {split} set...")
        metrics = evaluate(model, loader, device, desc=f"Eval {split}")
        results['metrics'][split] = metrics
        
        print(f"  PSNR: {metrics['psnr_mean']:.4f} ± {metrics['psnr_std']:.4f} dB")
        print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")

    # Save sample images
    if args.save_samples > 0 and loaders['val']:
        print(f"\nSaving {args.save_samples} sample images...")
        save_sample_images(model, loaders['val'], device, args.save_dir, args.save_samples)

    # Save results
    results_path = os.path.join(args.save_dir, "bicubic_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {results_path}")

    # Also save a summary text file
    summary_path = os.path.join(args.save_dir, "bicubic_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SpinOff Bicubic Baseline Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Scale Factor: {args.scale}x\n")
        f.write(f"Image Size: {args.image_size}\n")
        f.write(f"Noise Level: {args.noise_level}\n")
        f.write(f"Timestamp: {results['timestamp']}\n\n")
        
        for split, metrics in results['metrics'].items():
            f.write(f"{split.upper()} Results:\n")
            f.write(f"  PSNR: {metrics['psnr_mean']:.4f} ± {metrics['psnr_std']:.4f} dB\n")
            f.write(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}\n")
            f.write(f"  Samples: {metrics['num_samples']}\n\n")
    
    print(f"✅ Summary saved to: {summary_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("📊 Final Summary")
    print("=" * 60)
    
    if 'val' in results['metrics']:
        val = results['metrics']['val']
        print(f"Validation PSNR: {val['psnr_mean']:.4f} dB")
        print(f"Validation SSIM: {val['ssim_mean']:.4f}")
    
    if 'test' in results['metrics']:
        test = results['metrics']['test']
        print(f"Test PSNR: {test['psnr_mean']:.4f} dB")
        print(f"Test SSIM: {test['ssim_mean']:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
