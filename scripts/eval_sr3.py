"""
Fast SR3 Evaluation Script
Evaluates ONLY N test samples (default 10) for quick benchmarking
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import MRIDataset
from models.sr3 import create_sr3_model
from utils.metrics import calculate_psnr, calculate_ssim


# ---------------------------------------------------------
# Load Checkpoint
# ---------------------------------------------------------
def load_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    return ckpt


# ---------------------------------------------------------
# Fast Evaluation (only N samples)
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_test(model, diffusion, test_loader, device, num_samples=10):
    model.eval()

    psnrs = []
    ssims = []
    sample_count = 0

    for lr, hr in tqdm(test_loader, desc="Evaluating"):
        if sample_count >= num_samples:
            break

        lr = lr.to(device)
        hr = hr.to(device)
        b = lr.shape[0]

        pred = diffusion.sample(lr, batch_size=b)

        psnr = calculate_psnr(pred, hr)
        ssim = calculate_ssim(pred, hr)

        psnrs.append(psnr)
        ssims.append(ssim)

        sample_count += b

    if len(psnrs) == 0:
        return 0.0, 0.0

    return float(torch.tensor(psnrs).mean()), float(torch.tensor(ssims).mean())


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate SR3 model on test split")

    parser.add_argument("--checkpoint", required=True, help="Path to SR3 checkpoint")
    parser.add_argument("--splits", required=True, help="splits.json file")
    parser.add_argument("--save_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test images")
    parser.add_argument("--create_lr_on_fly", action="store_true", help="Generate LR images automatically")
    parser.add_argument("--scale", type=int, default=2, help="Downsampling scale")
    parser.add_argument("--noise_level", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=0)  # Required for Windows

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Load splits
    # ---------------------------------------------------------
    with open(args.splits, "r") as f:
        splits = json.load(f)

    test_paths = splits.get("test", [])
    print(f"Original test samples: {len(test_paths)}")

 # FAST mode – use only 10 images
    test_paths = test_paths[:1000]
    print(f"Using only {len(test_paths)} samples for fast evaluation")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    test_ds = MRIDataset(
        hr_paths=test_paths,
        lr_paths=None,
        transform=None,
        create_lr_on_fly=args.create_lr_on_fly,
        scale_factor=args.scale,
        noise_level=args.noise_level,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------
    model, diffusion = create_sr3_model(
        image_size=256,
        in_channels=1,
        base_channels=64,
        timesteps=1000,
        lr_scale=args.scale,
    )

    model = model.to(device)

    # Move diffusion tensors
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    diffusion.posterior_variance = diffusion.posterior_variance.to(device)

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------
    load_checkpoint(model, args.checkpoint, device)

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------
    print("\nStarting Fast Evaluation (only {} samples)…".format(args.num_samples))

    psnr, ssim = evaluate_test(
        model,
        diffusion,
        test_loader,
        device,
        num_samples=args.num_samples
    )

    print("\n=============================")
    print(" FAST TEST RESULTS")
    print("=============================")
    print(f"PSNR: {psnr:.4f}")
    print(f"SSIM: {ssim:.4f}")

    # Save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump({"psnr": psnr, "ssim": ssim}, f, indent=4)

    print("Saved metrics.json")


if __name__ == "__main__":
    main()
