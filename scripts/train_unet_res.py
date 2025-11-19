"""
Resume-capable trainer for SpinOff U-Net Super-Resolution.
Fully compatible with your original train_unet.py.
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
from models.unet import UNet
from utils.metrics import calculate_psnr, calculate_ssim


# ---------------------------------------------------------------
# Helper: infer last epoch from unet_epochN.pt files
# ---------------------------------------------------------------
def infer_last_epoch(save_dir):
    pattern = re.compile(r"unet_epoch(\d+)\.pt$")
    max_epoch = 0
    for p in glob(os.path.join(save_dir, "unet_epoch*.pt")):
        m = pattern.search(os.path.basename(p))
        if m:
            n = int(m.group(1))
            max_epoch = max(max_epoch, n)
    return max_epoch


# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SpinOff U-Net Trainer with Resume Support")

    p.add_argument("--splits", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_dir", type=str, default="./results/unet")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--create_lr_on_fly", action="store_true")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--noise_level", type=float, default=0.02)

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
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for lr_img, hr_img in tqdm(loader, desc="Train", leave=False):
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        pred = model(lr_img)
        loss = criterion(pred, hr_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------
def validate(model, loader, device):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for lr_img, hr_img in tqdm(loader, desc="Val", leave=False):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            pred = model(lr_img)
            psnrs.append(calculate_psnr(pred, hr_img))
            ssims.append(calculate_ssim(pred, hr_img))

    psnr_mean = float(torch.tensor(psnrs).mean())
    ssim_mean = float(torch.tensor(ssims).mean())
    return psnr_mean, ssim_mean


# ---------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    print("/content/SpinOff")
    print(f"📂 Using device: {device}")
    print("🔹 Loading splits:", args.splits)

    train_loader, val_loader = make_dataloaders(
        splits_file=args.splits,
        batch_size=args.batch_size,
        create_lr_on_fly=args.create_lr_on_fly,
        scale_factor=args.scale,
        noise_level=args.noise_level
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    # ===== Resume logic (exactly matching your script) =====
    start_epoch = 1
    best_psnr = -1.0
    checkpoint_path = os.path.join(args.save_dir, "unet_final.pt")

    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        if "model_state" in ckpt:
            # new format
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            best_psnr = ckpt.get("best_psnr", -1.0)
            print(f"➡️ Resuming from epoch {start_epoch} | Best PSNR: {best_psnr:.4f}")

        else:
            # old format – weights only
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
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_psnr, val_ssim = validate(model, val_loader, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val PSNR: {val_psnr:.4f} | Val SSIM: {val_ssim:.4f}")

        # Update best_psnr BEFORE saving
        if val_psnr > best_psnr:
            best_psnr = val_psnr

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_psnr": best_psnr
        }

        # Save best
        if val_psnr >= best_psnr:
            best_path = os.path.join(args.save_dir, "unet_best.pt")
            torch.save(ckpt, best_path)
            print(f"✅ New best saved: {best_path}")

        # epoch file + latest
        epoch_path = os.path.join(args.save_dir, f"unet_epoch{epoch}.pt")
        torch.save(ckpt, epoch_path)
        torch.save(ckpt, os.path.join(args.save_dir, "unet_latest.pt"))

    # final
    final_path = os.path.join(args.save_dir, "unet_final.pt")
    torch.save(ckpt, final_path)
    print(f"\n🎯 Training finished. Final saved to: {final_path}")
    print(f"Best val PSNR: {best_psnr:.4f}")
