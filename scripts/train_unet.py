"""
Train U-Net baseline for Diffusion Super-Resolution (SpinOff)
--------------------------------------------------------------
This script trains the U-Net model on processed IXI MRI images.
It uses data/dataset.py and utils/metrics.py from this repo.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Path fix for Colab and local runs ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Now these imports will work ---
from data.dataset import MRIDataset
from models.unet import UNet
from utils.metrics import psnr, ssim



def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for MRI Super-Resolution")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to processed IXI dataset (e.g. .../IXI/processed/png_slices)")
    parser.add_argument("--splits_file", type=str, required=True,
                        help="Path to splits.json file generated during preprocessing")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./results/unet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for lr_imgs, hr_imgs in tqdm(loader, desc="Training", leave=False):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        preds = model(lr_imgs)
        loss = criterion(preds, hr_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(loader, desc="Validation", leave=False):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            preds = model(lr_imgs)
            psnr_scores.append(psnr(preds, hr_imgs))
            ssim_scores.append(ssim(preds, hr_imgs))
    return sum(psnr_scores) / len(psnr_scores), sum(ssim_scores) / len(ssim_scores)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    print(f"📂 Using device: {device}")

    # Load datasets
    print("🔹 Loading dataset splits...")
    train_data = IXIDataset(root=args.data_dir, split="train", splits_file=args.splits_file)
    val_data = IXIDataset(root=args.data_dir, split="val", splits_file=args.splits_file)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Model setup
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    best_psnr = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n🌍 Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_psnr, val_ssim = validate(model, val_loader, device)

        print(f"Loss: {train_loss:.4f} | PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.3f}")

        # Save checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = os.path.join(args.save_dir, "unet_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved: {save_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, "unet_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n🎯 Training complete. Final model saved at: {final_path}")


if __name__ == "__main__":
    main()
