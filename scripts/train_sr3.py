"""
Resume-capable trainer for SpinOff SR3 Super-Resolution.
Fully compatible with your training workflow and checkpoint system.
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
from torch.utils.data import DataLoader

# Add repo root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import MRIDataset
from models.sr3 import create_sr3_model
from utils.metrics import calculate_psnr, calculate_ssim


# ---------------------------------------------------------------
# Helper: infer last epoch from sr3_epochN.pt files
# ---------------------------------------------------------------
def infer_last_epoch(save_dir):
    pattern = re.compile(r"sr3_epoch(\d+)\.pt$")
    max_epoch = 0
    for p in glob(os.path.join(save_dir, "sr3_epoch*.pt")):
        m = pattern.search(os.path.basename(p))
        if m:
            n = int(m.group(1))
            max_epoch = max(max_epoch, n)
    return max_epoch


# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SpinOff SR3 Trainer with Resume Support")

    p.add_argument("--splits", required=True, help="Path to splits JSON file")
    p.add_argument("--epochs", type=int, default=20, help="Total epochs to train")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size (smaller for diffusion)")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--save_dir", type=str, default="./results/sr3", help="Directory to save checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Data arguments
    p.add_argument("--create_lr_on_fly", action="store_true", help="Create LR images on-the-fly")
    p.add_argument("--scale", type=int, default=2, help="Downsampling scale factor")
    p.add_argument("--noise_level", type=float, default=0.02, help="Noise level for degradation")

    # SR3 specific arguments
    p.add_argument("--image_size", type=int, default=128, help="High-res image size")
    p.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion steps")
    p.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--base_channels", type=int, default=64, help="Base channels in U-Net")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    p.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    p.add_argument("--use_ema", action="store_true", help="Use exponential moving average")

    return p.parse_args()


# ---------------------------------------------------------------
def make_dataloaders(splits_file, batch_size, create_lr_on_fly, scale_factor, noise_level):
    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_paths = splits.get("train", [])
    val_paths = splits.get("val", [])

    train_ds = MRIDataset(
        hr_paths=train_paths,
        lr_paths=None,
        transform=None,
        create_lr_on_fly=create_lr_on_fly,
        scale_factor=scale_factor,
        noise_level=noise_level
    )
    val_ds = MRIDataset(
        hr_paths=val_paths,
        lr_paths=None,
        transform=None,
        create_lr_on_fly=create_lr_on_fly,
        scale_factor=scale_factor,
        noise_level=noise_level
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ---------------------------------------------------------------
# EMA helper class
# ---------------------------------------------------------------
class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# ---------------------------------------------------------------
def train_one_epoch(model, diffusion, loader, optimizer, device, grad_clip=1.0, ema=None):
    """Train SR3 for one epoch"""
    model.train()
    total_loss = 0.0
    
    for lr_img, hr_img in tqdm(loader, desc="Train", leave=False):
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        
        # Sample random timesteps
        batch_size = hr_img.shape[0]
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
        
        # Calculate loss
        loss = diffusion.p_losses(hr_img, lr_img, t)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


# ---------------------------------------------------------------
@torch.no_grad()
def validate(model, diffusion, loader, device, num_samples=50):
    """
    Validate SR3 model.
    Note: Full sampling is slow, so we limit validation samples.
    """
    model.eval()
    psnrs, ssims = [], []
    
    sample_count = 0
    for lr_img, hr_img in tqdm(loader, desc="Val", leave=False):
        if sample_count >= num_samples:
            break
        
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        batch_size = lr_img.shape[0]
        
        # Generate samples (this is slow!)
        pred = diffusion.sample(lr_img, batch_size=batch_size)
        
        # Calculate metrics
        psnrs.append(calculate_psnr(pred, hr_img))
        ssims.append(calculate_ssim(pred, hr_img))
        
        sample_count += batch_size
    
    if len(psnrs) == 0:
        return 0.0, 0.0
    
    psnr_mean = float(torch.tensor(psnrs).mean())
    ssim_mean = float(torch.tensor(ssims).mean())
    return psnr_mean, ssim_mean


# ---------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    print("=" * 60)
    print("SpinOff SR3 Training")
    print("=" * 60)
    print(f"📂 Using device: {device}")
    print(f"🔹 Loading splits: {args.splits}")
    print(f"🔹 Image size: {args.image_size}")
    print(f"🔹 Timesteps: {args.timesteps}")
    print(f"🔹 Beta schedule: {args.beta_schedule}")
    print(f"🔹 Base channels: {args.base_channels}")
    print(f"🔹 Use EMA: {args.use_ema}")
    print("=" * 60)

    # Create data loaders
    train_loader, val_loader = make_dataloaders(
        splits_file=args.splits,
        batch_size=args.batch_size,
        create_lr_on_fly=args.create_lr_on_fly,
        scale_factor=args.scale,
        noise_level=args.noise_level
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create SR3 model and diffusion process
    model, diffusion = create_sr3_model(
        image_size=args.image_size,
        in_channels=1,
        base_channels=args.base_channels,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule
    )
    model = model.to(device)
    
    # Move diffusion parameters to device
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    diffusion.posterior_variance = diffusion.posterior_variance.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    # ===== Resume logic (exactly matching train_unet.py) =====
    start_epoch = 1
    best_psnr = -1.0
    checkpoint_path = os.path.join(args.save_dir, "sr3_final.pt")

    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        if "model_state" in ckpt:
            # New format - full checkpoint
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "ema_state" in ckpt and ema is not None:
                ema.load_state_dict(ckpt["ema_state"])
            start_epoch = ckpt["epoch"] + 1
            best_psnr = ckpt.get("best_psnr", -1.0)
            print(f"➡️ Resuming from epoch {start_epoch} | Best PSNR: {best_psnr:.4f}")

        else:
            # Old format – weights only
            model.load_state_dict(ckpt)
            inferred = infer_last_epoch(args.save_dir)
            if inferred > 0:
                start_epoch = inferred + 1
                print(f"⚙️ Found epoch files → resume from epoch {start_epoch}")
            else:
                start_epoch = 3
                print("⚙️ No epoch files → default resume from epoch 3")

    else:
        print("⚠️ No checkpoint found — starting fresh.")

    # ======================================================
    # Training Loop
    # ======================================================
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss = train_one_epoch(
            model, diffusion, train_loader, optimizer, device, 
            grad_clip=args.grad_clip, ema=ema
        )
        
        # Validate (with EMA if enabled)
        if ema is not None:
            ema.apply_shadow()
        
        val_psnr, val_ssim = validate(model, diffusion, val_loader, device)
        
        if ema is not None:
            ema.restore()

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val PSNR: {val_psnr:.4f} | Val SSIM: {val_ssim:.4f}")

        # Update best_psnr BEFORE saving
        if val_psnr > best_psnr:
            best_psnr = val_psnr

        # Prepare checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "config": {
                "image_size": args.image_size,
                "timesteps": args.timesteps,
                "beta_schedule": args.beta_schedule,
                "base_channels": args.base_channels,
                "scale_factor": args.scale,
                "noise_level": args.noise_level
            }
        }
        
        # Add EMA state if used
        if ema is not None:
            ckpt["ema_state"] = ema.state_dict()

        # Save best
        if val_psnr >= best_psnr:
            best_path = os.path.join(args.save_dir, "sr3_best.pt")
            torch.save(ckpt, best_path)
            print(f"✅ New best saved: {best_path}")

        # Save epoch checkpoint
        epoch_path = os.path.join(args.save_dir, f"sr3_epoch{epoch}.pt")
        torch.save(ckpt, epoch_path)
        
        # Save latest
        latest_path = os.path.join(args.save_dir, "sr3_latest.pt")
        torch.save(ckpt, latest_path)

    # Save final checkpoint
    final_path = os.path.join(args.save_dir, "sr3_final.pt")
    torch.save(ckpt, final_path)
    print(f"\n{'=' * 60}")
    print(f"🎯 Training finished. Final saved to: {final_path}")
    print(f"Best val PSNR: {best_psnr:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
