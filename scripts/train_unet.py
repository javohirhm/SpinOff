"""
Train U-Net baseline for Diffusion Super-Resolution (SpinOff)
--------------------------------------------------------------
Self-contained version that includes PSNR and SSIM implementations.
Works in Colab with MRIDataset and UNet.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from math import log10

# --- Fix paths for Colab/local runs ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Internal repo imports ---
from data.dataset import MRIDataset
from models.unet import UNet


# ============================================================
# Metric functions (built-in, so no external import required)
# ============================================================

def psnr(pred, target):
    """Compute Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(pred, target, C1=0.01**2, C2=0.03**2):
    """Compute Structural Similarity Index (simplified PyTorch version)"""
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    ssim_value = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                 ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_value


# ============================================================
# Training and validation
# ============================================================

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
            psnr_scores.append(psnr(preds, hr_imgs).item())
            ssim_scores.append(ssim(preds, hr_imgs).item())
    return sum(psnr_scores) / len(psnr_scores), sum(ssim_scores) / len(ssim_scores)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for MRI Super-Resolution")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to processed IXI dataset (e.g. .../IXI/processed/png_slices)")
    parser.add_argument("--splits_file", type=str, required=True,
                        help="Path to splits.json file generated during preprocessing")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./results/unet")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    print(f"📂 Using device: {device}")

    # Load datasets
    print("🔹 Loading dataset splits...")
    train_data = MRIDataset(root=args.data_dir, split="train", splits_file=args.splits_file)
    val_data = MRIDataset(root=args.data_dir, split="val", splits_file=args.splits_file)
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

        # Save best model
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
