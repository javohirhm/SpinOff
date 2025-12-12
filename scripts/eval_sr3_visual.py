import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# SR3 model
from models.sr3 import create_sr3_model

# Dataset
from data.dataset import MRIDataset


def save_visual(lr, sr, hr, out_file):
    """Save LR, SR, and HR images side by side."""
    lr = lr.squeeze().cpu().numpy()
    sr = sr.squeeze().cpu().numpy()
    hr = hr.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Low-Resolution Input")
    plt.imshow(lr, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("SR3 Output")
    plt.imshow(sr, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth (HR)")
    plt.imshow(hr, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/eval_sr3")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--create_lr_on_fly", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    # ----------------------------------------------------------
    # Load test split
    # ----------------------------------------------------------
    with open(args.splits, "r") as f:
        splits = json.load(f)

    test_paths = splits["test"]
    print("Total test slices:", len(test_paths))

    # Take only num_samples
    test_paths = test_paths[: args.num_samples]
    print("Evaluating ONLY:", len(test_paths))

    # Build dataset
    test_ds = MRIDataset(
        hr_paths=test_paths,
        lr_paths=None,
        create_lr_on_fly=args.create_lr_on_fly
    )

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load SR3 model
    # ----------------------------------------------------------
    print("Loading SR3 model...")
    model, diffusion = create_sr3_model()
    model = model.to(device)
    model.eval()

    print("Loading checkpoint:", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Your checkpoints always store weights under model_state
    if "model_state" in ckpt:
        print("Loading weights from model_state")
        model.load_state_dict(ckpt["model_state"])
    else:
        print("Loading entire checkpoint directly")
        model.load_state_dict(ckpt)

    print("Checkpoint loaded successfully.\n")

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    print("Running SR3 evaluation...")

    results = []

    for i, (lr, hr) in enumerate(tqdm(test_loader)):
        lr = lr.to(device)
        hr = hr.to(device)

        with torch.no_grad():
            sr = diffusion.sample(lr, batch_size=1)

        # Compute PSNR
        mse = F.mse_loss(sr, hr).item()
        psnr = -10 * np.log10(mse + 1e-8)

        # simplified SSIM
        mu_x, mu_y = hr.mean().item(), sr.mean().item()
        sigma_x, sigma_y = hr.var().item(), sr.var().item()
        sigma_xy = ((hr - mu_x) * (sr - mu_y)).mean().item()

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

        results.append({"psnr": psnr, "ssim": ssim})

        # Save visualization
        out_path = save_dir / f"sample_{i:03d}.png"
        save_visual(lr, sr, hr, out_path)
        print("Saved visualization:", out_path)

    # Metrics summary
    mean_psnr = float(np.mean([r["psnr"] for r in results]))
    mean_ssim = float(np.mean([r["ssim"] for r in results]))

    metrics = {
        "num_samples": len(results),
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim
    }

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n=== SR3 EVALUATION DONE ===")
    print("PSNR:", mean_psnr)
    print("SSIM:", mean_ssim)


if __name__ == "__main__":
    main()
