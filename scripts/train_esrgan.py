#!/usr/bin/env python3
"""
scripts/train_esrgan.py
Training entrypoint for ESRGAN baseline.

- Pretrain generator with L1
- Finetune with adversarial (RaGAN) + perceptual + L1
- Optional segmentation-guided loss if models.unet.UNet is available

Usage example:
python scripts/train_esrgan.py --hr-dir ./data/HR --lr-dir ./data/LR --out-dir ./models/checkpoints/esrgan \
    --in-ch 1 --batch 8 --pretrain-epochs 20 --gan-epochs 60
"""

import os
import argparse
import random
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ESRGAN model modules
from models.esrgan import (
    RRDBNet,
    Discriminator,
    PerceptualLoss,
    SegmentationGuidedLoss,
    gan_loss_discriminator,
    gan_loss_generator,
)

# YOUR metrics
from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    MetricsCalculator
)

# --------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------
class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, hr_dir, lr_dir=None, crop_size=128, augment=True, in_ch=1):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.files = sorted([
            p for p in self.hr_dir.iterdir()
            if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])
        self.crop_size = crop_size
        self.augment = augment
        self.in_ch = in_ch
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def random_crop(self, img, size):
        w, h = img.size
        if w == size and h == size:
            return img
        left = random.randint(0, w - size)
        top = random.randint(0, h - size)
        return img.crop((left, top, left + size, top + size))

    def load_img(self, p):
        return Image.open(p).convert('L' if self.in_ch == 1 else 'RGB')

    def __getitem__(self, idx):
        hr_path = self.files[idx]
        hr = self.load_img(hr_path)
        hr = self.random_crop(hr, self.crop_size)

        # LR image if available, otherwise downsample HR
        if self.lr_dir and (self.lr_dir / hr_path.name).exists():
            lr = self.load_img(self.lr_dir / hr_path.name)
            lr = lr.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        else:
            down = hr.resize((self.crop_size // 2, self.crop_size // 2), Image.BICUBIC)
            lr = down.resize((self.crop_size, self.crop_size), Image.BICUBIC)

        # Augmentations
        if self.augment:
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
                lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
                lr = lr.transpose(Image.FLIP_TOP_BOTTOM)

        hr_t = self.to_tensor(hr)
        lr_t = self.to_tensor(lr)

        # Convert RGB→grayscale if needed
        if hr_t.shape[0] != self.in_ch:
            hr_t = hr_t.mean(dim=0, keepdim=True)
            lr_t = lr_t.mean(dim=0, keepdim=True)

        return lr_t, hr_t, hr_path.name


# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_ckpt(state, path):
    torch.save(state, path)


# --------------------------------------------------------------------
# Pretraining
# --------------------------------------------------------------------
def pretrain_generator(generator, dataloader, opt_g, device, epochs, l1_weight=1.0, log_interval=100, amp=True):
    l1_loss = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device == 'cuda'))
    generator.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f'Pretrain G {epoch}/{epochs}')
        running = 0.0

        for i, (lr, hr, _) in enumerate(pbar):
            lr, hr = lr.to(device), hr.to(device)
            opt_g.zero_grad()

            with torch.cuda.amp.autocast(enabled=(amp and device == 'cuda')):
                sr = generator(lr)
                loss = l1_loss(sr, hr) * l1_weight

            scaler.scale(loss).backward()
            scaler.step(opt_g)
            scaler.update()

            running += loss.item()
            if (i + 1) % log_interval == 0:
                pbar.set_postfix({'L1': running / (i + 1)})


# --------------------------------------------------------------------
# GAN Finetune
# --------------------------------------------------------------------
def gan_finetune(generator, discriminator, dataloader, opt_g, opt_d, device,
                 epochs, lambda_l1, lambda_perc, lambda_gan, seg_loss=None,
                 log_interval=100, amp=True, ema=None):

    l1_loss = nn.L1Loss()
    perceptual = PerceptualLoss(device=device)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device == 'cuda'))

    generator.train()
    discriminator.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f'GAN Epoch {epoch}/{epochs}')

        for i, (lr, hr, _) in enumerate(pbar):
            lr, hr = lr.to(device), hr.to(device)

            # --------------------
            # Train Discriminator
            # --------------------
            opt_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=(amp and device == 'cuda')):
                with torch.no_grad():
                    sr_det = generator(lr)

                d_real = discriminator(hr)
                d_fake = discriminator(sr_det.detach())
                loss_d = gan_loss_discriminator(d_real, d_fake)

            scaler.scale(loss_d).backward()
            scaler.step(opt_d)
            scaler.update()

            # --------------------
            # Train Generator
            # --------------------
            opt_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=(amp and device == 'cuda')):
                sr = generator(lr)
                d_real = discriminator(hr)
                d_fake = discriminator(sr)

                loss_l1 = l1_loss(sr, hr) * lambda_l1
                loss_perc = perceptual(sr, hr) * lambda_perc
                loss_gan = gan_loss_generator(d_real, d_fake) * lambda_gan

                loss_g = loss_l1 + loss_perc + loss_gan

                if seg_loss:
                    loss_seg = seg_loss(sr, hr)
                    loss_g += loss_seg

            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()

            # EMA update
            if ema is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema.parameters(), generator.parameters()):
                        p_ema.data.mul_(0.999).add_(p.data, alpha=1 - 0.999)

            if (i + 1) % log_interval == 0:
                pbar.set_postfix({
                    'L1': loss_l1.item(),
                    'Perc': loss_perc.item(),
                    'GAN': loss_gan.item()
                })


# --------------------------------------------------------------------
# Evaluation using YOUR metrics.py
# --------------------------------------------------------------------
def evaluate_esrgan(model, dataloader, device):
    model.eval()
    metrics = MetricsCalculator()

    with torch.no_grad():
        for lr, hr, _ in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr).clamp(0, 1)
            metrics.update(sr.cpu(), hr.cpu())

    return metrics.get_metrics()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr-dir', type=str, required=True)
    parser.add_argument('--lr-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='./models/checkpoints/esrgan')
    parser.add_argument('--in-ch', type=int, default=1)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--pretrain-epochs', type=int, default=20)
    parser.add_argument('--gan-epochs', type=int, default=60)
    parser.add_argument('--lr-g', type=float, default=2e-4)
    parser.add_argument('--lr-d', type=float, default=1e-4)
    parser.add_argument('--lambda-l1', type=float, default=1.0)
    parser.add_argument('--lambda-perc', type=float, default=0.006)
    parser.add_argument('--lambda-gan', type=float, default=0.005)
    parser.add_argument('--use-seg-guided', action='store_true')
    parser.add_argument('--unet-path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)

    # Dataset
    ds = PairedImageDataset(args.hr_dir, args.lr_dir, crop_size=128, augment=True, in_ch=args.in_ch)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    # Models
    generator = RRDBNet(in_nc=args.in_ch, out_nc=args.in_ch, nf=64, nb=8).to(args.device)
    discriminator = Discriminator(in_nc=args.in_ch, nf=64).to(args.device)

    # EMA model
    ema_gen = RRDBNet(in_nc=args.in_ch, out_nc=args.in_ch, nf=64, nb=8).to(args.device)
    ema_gen.load_state_dict(generator.state_dict())
    for p in ema_gen.parameters():
        p.requires_grad = False

    # Optional segmentation guided UNet
    seg_loss = None
    if args.use_seg_guided:
        try:
            if args.unet_path:
                mod = __import__(args.unet_path, fromlist=['UNet'])
                UNet = getattr(mod, 'UNet')
            else:
                from models.unet import UNet

            unet_model = UNet(in_channels=args.in_ch, out_channels=args.in_ch).to(args.device)
            seg_loss = SegmentationGuidedLoss(unet_model, weight=0.2)
            print("Segmentation-guided mode ENABLED.")
        except Exception as e:
            print("Failed to load UNet:", e)

    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))

    # ---------------------------------
    # 1. Pretrain Generator (L1 only)
    # ---------------------------------
    print("Starting generator pretraining...")
    pretrain_generator(generator, loader, opt_g, args.device, args.pretrain_epochs, l1_weight=args.lambda_l1)

    torch.save({'generator': generator.state_dict()},
               os.path.join(args.out_dir, 'pretrain_generator.pth'))

    # ---------------------------------
    # 2. GAN Finetuning
    # ---------------------------------
    print("Starting adversarial finetune...")
    gan_finetune(generator, discriminator, loader, opt_g, opt_d, args.device,
                 epochs=args.gan_epochs,
                 lambda_l1=args.lambda_l1,
                 lambda_perc=args.lambda_perc,
                 lambda_gan=args.lambda_gan,
                 seg_loss=seg_loss,
                 ema=ema_gen)

    torch.save({'generator': generator.state_dict()},
               os.path.join(args.out_dir, 'generator_final.pth'))
    torch.save({'generator_ema': ema_gen.state_dict()},
               os.path.join(args.out_dir, 'generator_ema_final.pth'))
    torch.save({'discriminator': discriminator.state_dict()},
               os.path.join(args.out_dir, 'discriminator_final.pth'))

    # ---------------------------------
    # Evaluation (using your metrics)
    # ---------------------------------
    print("Evaluating ESRGAN...")
    results = evaluate_esrgan(ema_gen, loader, args.device)

    print("\nFinal ESRGAN Metrics:")
    print(f"PSNR:  {results['psnr']:.4f}")
    print(f"SSIM:  {results['ssim']:.4f}")
    print(f"MAE:   {results['mae']:.6f}")
    print(f"RMSE:  {results['rmse']:.6f}")


if __name__ == "__main__":
    main()
