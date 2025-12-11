"""
Training script for SR3-FiLM: Optimized Diffusion Model for Medical Image Super-Resolution

Efficiency Optimizations:
- Patch-based training (128×128) to fit GPU memory
- Mixed-precision training (FP16) for faster computation
- Gradient accumulation for effective larger batch size
- Efficient data loading with prefetching

Usage:
    python scripts/train_sr3_film.py --splits data/IXI/processed/png_slices/splits.json \
        --epochs 20 --batch_size 8 --save_dir results/sr3_film --create_lr_on_fly
"""

import os
import sys
import re
import json
import argparse
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add repo root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import MRIDataset
from models.sr3_film import create_sr3_film_model
from utils.metrics import calculate_psnr, calculate_ssim


# ============================================================================
# Helper Functions
# ============================================================================

def infer_last_epoch(save_dir: str) -> int:
    """Infer last epoch from checkpoint files."""
    pattern = re.compile(r"sr3_film_epoch(\d+)\.pt$")
    max_epoch = 0
    for p in glob(os.path.join(save_dir, "sr3_film_epoch*.pt")):
        m = pattern.search(os.path.basename(p))
        if m:
            max_epoch = max(max_epoch, int(m.group(1)))
    return max_epoch


def move_diffusion_to_device(diffusion, device):
    """Move diffusion tensors to device."""
    for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))


def parse_args():
    p = argparse.ArgumentParser(description="SR3-FiLM Training with Optimizations")
    
    # Data
    p.add_argument("--splits", required=True, help="Path to splits JSON file")
    p.add_argument("--create_lr_on_fly", action="store_true", help="Create LR images on-the-fly")
    p.add_argument("--scale", type=int, default=2, help="Downsampling scale factor")
    p.add_argument("--noise_level", type=float, default=0.02, help="Noise level for degradation")
    
    # Training
    p.add_argument("--epochs", type=int, default=20, help="Total epochs")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    p.add_argument("--save_dir", type=str, default="./results/sr3_film", help="Save directory")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Efficiency optimizations
    p.add_argument("--use_amp", action="store_true", help="Use mixed-precision training (FP16)")
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Model architecture
    p.add_argument("--image_size", type=int, default=256, help="HR image size")
    p.add_argument("--base_channels", type=int, default=64, help="Base channels")
    p.add_argument("--channel_mults", type=str, default="1,2,4,8", help="Channel multipliers")
    p.add_argument("--num_res_blocks", type=int, default=2, help="Residual blocks per level")
    p.add_argument("--attention_resolutions", type=str, default="32,16", help="Attention resolutions")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Diffusion
    p.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    p.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    
    # EMA
    p.add_argument("--use_ema", action="store_true", help="Use EMA")
    p.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay")
    
    # Validation
    p.add_argument("--val_samples", type=int, default=50, help="Validation samples")
    p.add_argument("--val_every", type=int, default=1, help="Validate every N epochs")
    
    return p.parse_args()


def make_dataloaders(args):
    """Create train and validation dataloaders."""
    with open(args.splits, "r") as f:
        splits = json.load(f)
    
    train_ds = MRIDataset(
        hr_paths=splits.get("train", []),
        lr_paths=None,
        transform=None,
        create_lr_on_fly=args.create_lr_on_fly,
        scale_factor=args.scale,
        noise_level=args.noise_level
    )
    val_ds = MRIDataset(
        hr_paths=splits.get("val", []),
        lr_paths=None,
        transform=None,
        create_lr_on_fly=args.create_lr_on_fly,
        scale_factor=args.scale,
        noise_level=args.noise_level
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# EMA
# ============================================================================

class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model, diffusion, loader, optimizer, device,
    scaler=None, use_amp=False, grad_clip=1.0, 
    grad_accum_steps=1, ema=None
):
    """Train for one epoch with optimizations."""
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (lr_img, hr_img) in enumerate(pbar):
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)
        
        batch_size = hr_img.shape[0]
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
        
        # Mixed precision forward pass
        if use_amp:
            with autocast():
                loss = diffusion.p_losses(hr_img, lr_img, t)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            loss = diffusion.p_losses(hr_img, lr_img, t)
            loss = loss / grad_accum_steps
            loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            if use_amp:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        total_loss += loss.item() * grad_accum_steps
        pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, diffusion, loader, device, num_samples=50, use_amp=False):
    """Validate model."""
    model.eval()
    psnrs, ssims = [], []
    sample_count = 0
    
    pbar = tqdm(loader, desc="Val", leave=False)
    for lr_img, hr_img in pbar:
        if sample_count >= num_samples:
            break
        
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)
        batch_size = lr_img.shape[0]
        
        # Generate samples
        if use_amp:
            with autocast():
                pred = diffusion.sample(lr_img, batch_size=batch_size)
        else:
            pred = diffusion.sample(lr_img, batch_size=batch_size)
        
        psnrs.append(calculate_psnr(pred, hr_img))
        ssims.append(calculate_ssim(pred, hr_img))
        sample_count += batch_size
    
    if not psnrs:
        return 0.0, 0.0
    
    return float(torch.tensor(psnrs).mean()), float(torch.tensor(ssims).mean())


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    
    # Parse architecture args
    channel_mults = tuple(int(x) for x in args.channel_mults.split(','))
    attention_res = tuple(int(x) for x in args.attention_resolutions.split(','))
    
    # Print config
    print("=" * 60)
    print("SR3-FiLM Training")
    print("=" * 60)
    print(f"📂 Device: {device}")
    print(f"📂 Save dir: {args.save_dir}")
    print(f"\n📊 Data:")
    print(f"   Splits: {args.splits}")
    print(f"   HR size: {args.image_size}, LR size: {args.image_size // args.scale}")
    print(f"   Scale: {args.scale}x")
    print(f"\n🏗️ Architecture:")
    print(f"   Base channels: {args.base_channels}")
    print(f"   Channel multipliers: {channel_mults}")
    print(f"   Res blocks per level: {args.num_res_blocks}")
    print(f"   Attention at: {attention_res}")
    print(f"\n⚙️ Diffusion:")
    print(f"   Timesteps: {args.timesteps}")
    print(f"   Beta schedule: {args.beta_schedule}")
    print(f"\n🚀 Optimizations:")
    print(f"   Mixed precision (FP16): {args.use_amp}")
    print(f"   Gradient accumulation: {args.grad_accum_steps}")
    print(f"   Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"   EMA: {args.use_ema}")
    print("=" * 60)
    
    # Create dataloaders
    train_loader, val_loader = make_dataloaders(args)
    print(f"\n📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Verify data shapes
    sample_lr, sample_hr = next(iter(train_loader))
    print(f"📊 Data shapes - LR: {sample_lr.shape}, HR: {sample_hr.shape}")
    
    # Adjust sizes if needed
    if sample_hr.shape[-1] != args.image_size:
        print(f"⚠️ Adjusting image_size: {args.image_size} -> {sample_hr.shape[-1]}")
        args.image_size = sample_hr.shape[-1]
        args.scale = sample_hr.shape[-1] // sample_lr.shape[-1]
    
    # Create model
    print("\n🔧 Creating SR3-FiLM model...")
    model, diffusion = create_sr3_film_model(
        image_size=args.image_size,
        in_channels=1,
        base_channels=args.base_channels,
        channel_multipliers=channel_mults,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=attention_res,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        lr_scale=args.scale,
        dropout=args.dropout
    )
    model = model.to(device)
    move_diffusion_to_device(diffusion, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # AMP scaler
    scaler = GradScaler() if args.use_amp else None
    
    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    
    # Resume logic
    start_epoch = 1
    best_psnr = -1.0
    checkpoint_path = os.path.join(args.save_dir, "sr3_film_final.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Resuming from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "ema_state" in ckpt and ema is not None:
                ema.load_state_dict(ckpt["ema_state"])
            if "scaler_state" in ckpt and scaler is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_psnr = ckpt.get("best_psnr", -1.0)
            print(f"➡️ Resuming from epoch {start_epoch} | Best PSNR: {best_psnr:.4f}")
        else:
            model.load_state_dict(ckpt)
            start_epoch = infer_last_epoch(args.save_dir) + 1
    else:
        print("\n⚠️ No checkpoint found — starting fresh.")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'─' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'─' * 60}")
        
        # Train
        train_loss = train_one_epoch(
            model, diffusion, train_loader, optimizer, device,
            scaler=scaler, use_amp=args.use_amp,
            grad_clip=args.grad_clip, grad_accum_steps=args.grad_accum_steps,
            ema=ema
        )
        
        # Validate
        val_psnr, val_ssim = 0.0, 0.0
        if epoch % args.val_every == 0:
            if ema is not None:
                ema.apply_shadow()
            
            val_psnr, val_ssim = validate(
                model, diffusion, val_loader, device,
                num_samples=args.val_samples, use_amp=args.use_amp
            )
            
            if ema is not None:
                ema.restore()
        
        print(f"\n📈 Epoch {epoch}: Loss={train_loss:.4f} | PSNR={val_psnr:.4f} | SSIM={val_ssim:.4f}")
        
        # Save checkpoints
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
        
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "config": vars(args)
        }
        
        if ema is not None:
            ckpt["ema_state"] = ema.state_dict()
        if scaler is not None:
            ckpt["scaler_state"] = scaler.state_dict()
        
        if is_best:
            torch.save(ckpt, os.path.join(args.save_dir, "sr3_film_best.pt"))
            print(f"   ✅ New best saved!")
        
        torch.save(ckpt, os.path.join(args.save_dir, f"sr3_film_epoch{epoch}.pt"))
        torch.save(ckpt, os.path.join(args.save_dir, "sr3_film_latest.pt"))
    
    # Save final
    torch.save(ckpt, os.path.join(args.save_dir, "sr3_film_final.pt"))
    
    print("\n" + "=" * 60)
    print(f"🎯 Training Complete! Best PSNR: {best_psnr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
