"""
Train U-Net baseline for Diffusion Super-Resolution (SpinOff)

This script matches the exact interfaces in your repository:
- data/dataset.py -> MRIDataset(hr_paths, lr_paths=None, transform=None, create_lr_on_fly=False, scale_factor=2, noise_level=0.02, ...)
- models/unet.py -> class UNet(...)
- utils/metrics.py -> functions calculate_psnr, calculate_ssim

Usage:
cd /content/SpinOff
!python scripts/train_unet.py \
  --splits "/content/drive/MyDrive/SpinOff_Project/data/IXI/processed/png_slices/splits.json" \
  --epochs 2 --batch_size 4 \
  --save_dir "/content/drive/MyDrive/SpinOff_Project/results/unet_test" \
  --create_lr_on_fly
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

# Ensure repo root (one level up from scripts/) is visible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Repo imports (aligned to your structure)
from data.dataset import MRIDataset
from models.unet import UNet
from utils.metrics import calculate_psnr, calculate_ssim


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def infer_last_epoch_from_dir(save_dir):
    """
    Look for files named like 'unet_epoch{n}.pt' in save_dir and return the max n.
    If none found, return 0.
    """
    pattern = re.compile(r'unet_epoch(\d+)\.pt$')
    max_epoch = 0
    for path in glob(os.path.join(save_dir, "unet_epoch*.pt")):
        fname = os.path.basename(path)
        m = pattern.search(fname)
        if m:
            try:
                n = int(m.group(1))
                if n > max_epoch:
                    max_epoch = n
            except:
                pass
    return max_epoch


# ===============================================================
# Argument parsing
# ===============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net (SpinOff repo - compatible version)")
    p.add_argument("--splits", required=True,
                   help="Path to splits.json produced by preprocessing (contains full file paths).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_dir", type=str, default="./results/unet")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--create_lr_on_fly", action="store_true",
                   help="If set, dataset will generate LR on the fly. If not set, assumes LR are provided.")
    p.add_argument("--scale", type=int, default=2, help="SR scale factor (default=2)")
    p.add_argument("--noise_level", type=float, default=0.02,
                   help="Noise level for LR simulation when creating on-the-fly")
    return p.parse_args()


# ===============================================================
# Dataloaders
# ===============================================================
def make_dataloaders(splits_file, batch_size, create_lr_on_fly, scale_factor, noise_level):
    # Load split lists
    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_paths = splits.get("train", [])
    val_paths = splits.get("val", [])

    if not train_paths:
        raise RuntimeError(f"No train paths found in splits file: {splits_file}")

    # MRIDataset(hr_paths, lr_paths=None, transform=None, create_lr_on_fly=False, scale_factor=2, noise_level=0.02)
    train_ds = MRIDataset(hr_paths=train_paths,
                          lr_paths=None,
                          transform=None,
                          create_lr_on_fly=create_lr_on_fly,
                          scale_factor=scale_factor,
                          noise_level=noise_level)
    val_ds = MRIDataset(hr_paths=val_paths,
                        lr_paths=None,
                        transform=None,
                        create_lr_on_fly=create_lr_on_fly,
                        scale_factor=scale_factor,
                        noise_level=noise_level)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ===============================================================
# Training and validation
# ===============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for lr, hr in tqdm(loader, desc="Train", leave=False):
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        loss = criterion(pred, hr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for lr, hr in tqdm(loader, desc="Val", leave=False):
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            # calculate_psnr and calculate_ssim are expected to accept tensors and return floats or tensors
            psnr_val = calculate_psnr(pred, hr)
            ssim_val = calculate_ssim(pred, hr)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
    psnr_mean = float(torch.tensor(psnr_list).mean()) if psnr_list else 0.0
    ssim_mean = float(torch.tensor(ssim_list).mean()) if ssim_list else 0.0
    return psnr_mean, ssim_mean


# ===============================================================
# Main training routine
# ===============================================================
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    print(f"/content/SpinOff\n📂 Using device: {device}")
    print("🔹 Loading splits:", args.splits)

    train_loader, val_loader = make_dataloaders(
        splits_file=args.splits,
        batch_size=args.batch_size,
        create_lr_on_fly=args.create_lr_on_fly,
        scale_factor=args.scale,
        noise_level=args.noise_level
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model, optimizer, and loss
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    # ---- Resume training from checkpoint if exists (supports old & new formats) ----
    start_epoch = 1
    best_psnr = -1.0
    resume_path = os.path.join(args.save_dir, "unet_final.pt")

    if os.path.exists(resume_path):
        print(f"🔄 Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        # New-format checkpoint: dict with keys 'model_state' etc.
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_psnr = checkpoint.get("best_psnr", -1.0)
            print(f"➡️ Resuming from epoch {start_epoch} | Best PSNR so far: {best_psnr:.4f}")
        else:
            # Old-format: checkpoint is a state_dict (weights only)
            # Load weights, and attempt to infer last epoch from saved epoch files
            model.load_state_dict(checkpoint)
            inferred = infer_last_epoch_from_dir(args.save_dir)
            if inferred > 0:
                start_epoch = inferred + 1
                print(f"⚙️ Loaded old-format weights. Inferred last epoch {inferred} -> resuming from {start_epoch}")
            else:
                # No epoch files to infer from; assume you trained 2 epochs previously as a practical fallback
                # (if you know the exact number, adjust this line)
                start_epoch = 3
                print(f"⚙️ Loaded old-format weights. No epoch files found; defaulting to resume from epoch {start_epoch}")
    else:
        print("⚠️ No checkpoint found — starting new training.")

    # ---- Training loop ----
    best_psnr = best_psnr if best_psnr is not None else -1.0
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_psnr, val_ssim = validate(model, val_loader, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.4f} | Val SSIM: {val_ssim:.4f}")

        # Save checkpoint dict
        # Update best_psnr properly
        if val_psnr > best_psnr:
            best_psnr = val_psnr

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_psnr": best_psnr
        }

        # Save best model (new-format)
        if val_psnr >= best_psnr:
            best_path = os.path.join(args.save_dir, "unet_best.pt")
            torch.save(checkpoint, best_path)
            print(f"✅ New best saved: {best_path}")

        # Save per-epoch and rolling checkpoints (new-format)
        epoch_path = os.path.join(args.save_dir, f"unet_epoch{epoch}.pt")
        torch.save(checkpoint, epoch_path)
        torch.save(checkpoint, os.path.join(args.save_dir, "unet_latest.pt"))

    # Final save
    final_path = os.path.join(args.save_dir, "unet_final.pt")
    torch.save(checkpoint, final_path)
    print(f"\n🎯 Training finished. Final saved to: {final_path}")
    print(f"Best val PSNR: {best_psnr:.4f}")


if __name__ == "__main__":
    main()
