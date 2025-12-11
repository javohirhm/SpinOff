# scripts/eval_sr3_film_fast.py
"""
Fast SR3-FiLM evaluation (EMA-preferred).
- Uses DDIM-like fast sampling (sample_fast) with --num_steps (default 20).
- Truncates test list to --num_samples BEFORE DataLoader to avoid iterating the full split.
- Applies EMA shadow weights if --use_ema and ema_state exist in checkpoint (recommended).
- PowerShell-friendly: pass args in one line or use backtick ` for continuation.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# allow running from project scripts/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import MRIDataset
from models.sr3_film import create_sr3_film_model, GaussianDiffusion
from utils.metrics import calculate_psnr, calculate_ssim

# Reuse EMA defined in your training script (train_sr3_film.py)
try:
    from train_sr3_film import EMA as TrainEMA
except Exception:
    # fallback minimal EMA (shouldn't be needed if train_sr3_film.py is present)
    class TrainEMA:
        def __init__(self, model, decay=0.9999):
            self.model = model; self.decay = decay; self.shadow = {}; self.backup = {}
            for n, p in model.named_parameters():
                if p.requires_grad: self.shadow[n] = p.data.clone()
        def load_state_dict(self, sd): self.shadow = sd
        def apply_shadow(self):
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.backup[n] = p.data
                    p.data = self.shadow[n]
        def restore(self):
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data = self.backup[n]
            self.backup = {}

# ---------------------------
# Add fast sampling to diffusion
# ---------------------------
def _add_sample_fast():
    @torch.no_grad()
    def sample_fast(self, cond, batch_size=1, steps=20):
        """
        Fast DDIM-like sampling. Deterministic-ish with small number of steps.
        """
        device = cond.device
        image_size = self.model.image_size
        channels = self.model.in_channels

        img = torch.randn(batch_size, channels, image_size, image_size, device=device)

        # uniform schedule from 0..timesteps-1 (inclusive)
        step_indices = torch.linspace(0, self.timesteps - 1, steps, dtype=torch.long, device=device)

        for idx in reversed(range(steps)):
            t = int(step_indices[idx].item())
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            eps = self.model(img, t_tensor, cond)

            alpha_t = self.alphas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            # basic DDIM-ish update (approx)
            img = (img - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

            if idx > 0:
                sigma = sqrt_one_minus_alpha_t
                img = img + sigma * torch.randn_like(img)

        return img.clamp(0.0, 1.0)

    GaussianDiffusion.sample_fast = sample_fast

_add_sample_fast()


# ---------------------------
# Helpers
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fast eval SR3-FiLM (EMA preferred)")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (sr3_film_best.pt / sr3_film_final.pt)")
    p.add_argument("--splits", required=True, help="Path to splits JSON")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to evaluate")
    p.add_argument("--save_dir", default="results/eval_sr3_film", help="Where to save metrics/preds")
    p.add_argument("--num_samples", type=int, default=10, help="Limit total images evaluated (truncate split first)")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for sampling")
    p.add_argument("--num_steps", type=int, default=20, help="Number of fast sampling steps (20 recommended)")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 recommended for Windows)")
    p.add_argument("--create_lr_on_fly", action="store_true", help="Generate LR on-the-fly (if training used that)")
    p.add_argument("--scale", type=int, default=2, help="Downsampling scale factor used in training")
    p.add_argument("--use_amp", action="store_true", help="Use autocast during sampling")
    p.add_argument("--use_ema", action="store_true", help="Prefer EMA weights if available in checkpoint (RECOMMENDED)")
    p.add_argument("--save_predictions", action="store_true", help="Save predicted HR arrays as .npy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image_size", type=int, default=256, help="HR image size (will adapt if data differs)")
    return p.parse_args()

def load_checkpoint_for_model(ckpt_path, model, ema_obj=None, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    # new format: dict with model_state
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        if ema_obj is not None and "ema_state" in ckpt:
            try:
                ema_obj.load_state_dict(ckpt["ema_state"])
            except Exception:
                ema_obj.shadow = ckpt["ema_state"]
    else:
        # assume raw state dict
        try:
            model.load_state_dict(ckpt)
        except Exception as e:
            raise RuntimeError("Unrecognized checkpoint format") from e
    return ckpt

def move_diffusion_tensors_to_device(diffusion, device):
    for attr in ['betas','alphas','alphas_cumprod','alphas_cumprod_prev',
                 'sqrt_alphas_cumprod','sqrt_one_minus_alphas_cumprod','posterior_variance']:
        if hasattr(diffusion, attr):
            val = getattr(diffusion, attr)
            if isinstance(val, torch.Tensor):
                setattr(diffusion, attr, val.to(device))


# ---------------------------
# Evaluation
# ---------------------------
@torch.no_grad()
def evaluate_fast(model, diffusion, loader, device, num_images=10, num_steps=20, use_amp=False, save_preds=False, out_dir=None):
    model.eval()
    psnrs = []
    ssims = []
    per_image = []
    seen = 0

    pbar = tqdm(loader, desc="Evaluating", leave=True)
    for batch in pbar:
        if seen >= num_images:
            break

        # dataset expected to return (lr, hr) or (lr, hr, meta)
        if isinstance(batch, (list, tuple)):
            lr = batch[0]
            hr = batch[1]
        else:
            raise RuntimeError("Dataset must return (lr, hr) tuples")

        b = lr.shape[0]
        if seen + b > num_images:
            trim = num_images - seen
            lr = lr[:trim]
            hr = hr[:trim]
            b = trim

        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        if use_amp:
            with autocast():
                preds = diffusion.sample_fast(lr, batch_size=b, steps=num_steps)
        else:
            preds = diffusion.sample_fast(lr, batch_size=b, steps=num_steps)

        preds = preds.clamp(0.0, 1.0).cpu()
        hr_cpu = hr.cpu()

        for i in range(b):
            p = preds[i:i+1]
            h = hr_cpu[i:i+1]

            psnr = calculate_psnr(p, h)
            ssim = calculate_ssim(p, h)

            psnrs.append(float(psnr))
            ssims.append(float(ssim))
            per_image.append({"index": seen + i, "psnr": float(psnr), "ssim": float(ssim)})

            if save_preds and out_dir is not None:
                import numpy as np
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, f"pred_{seen + i:06d}.npy"), p.squeeze(0).numpy())

        seen += b
        pbar.set_postfix({"seen": seen, "psnr_mean": (sum(psnrs) / len(psnrs)) if psnrs else 0.0})

    if len(psnrs) == 0:
        return {"psnr_mean": 0.0, "ssim_mean": 0.0, "per_image": []}

    import torch as _torch
    return {
        "psnr_mean": float(_torch.tensor(psnrs).mean()),
        "ssim_mean": float(_torch.tensor(ssims).mean()),
        "per_image": per_image
    }

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # load splits and truncate to requested num_samples BEFORE DataLoader
    with open(args.splits, "r") as f:
        splits = json.load(f)

    paths = splits.get(args.split, [])
    if not paths:
        raise RuntimeError(f"No paths found for split '{args.split}' in {args.splits}")

    orig_count = len(paths)
    print(f"Found {orig_count} files in split '{args.split}'")
    if args.num_samples is not None and args.num_samples < orig_count:
        paths = paths[: args.num_samples]
        print(f"Truncating to first {len(paths)} samples for fast eval")

    # create dataset + loader
    ds = MRIDataset(hr_paths=paths, lr_paths=None, transform=None,
                    create_lr_on_fly=args.create_lr_on_fly, scale_factor=args.scale, noise_level=0.02)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"DataLoader created with {len(loader)} batches (batch_size={args.batch_size})")

    # build model + diffusion (use defaults then load ckpt)
    model, diffusion = create_sr3_film_model(
        image_size=args.image_size,
        in_channels=1,
        base_channels=64,
        channel_multipliers=(1,2,4,8),
        num_res_blocks=2,
        attention_resolutions=(32,16),
        timesteps=1000,
        lr_scale=args.scale,
        dropout=0.1
    )
    model = model.to(device)

    # prepare EMA object (for loading/applying ema_state)
    ema = TrainEMA(model) if args.use_ema else None

    print("Loading checkpoint:", args.checkpoint)
    ckpt = load_checkpoint_for_model(args.checkpoint, model, ema_obj=ema, device=device)

    # If checkpoint saved config with timesteps, recreate diffusion to match config:
    ck_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    desired_timesteps = None
    if ck_cfg and "timesteps" in ck_cfg:
        try:
            desired_timesteps = int(ck_cfg["timesteps"])
        except Exception:
            desired_timesteps = None

    if desired_timesteps:
        model, diffusion = create_sr3_film_model(
            image_size=getattr(model, "image_size", args.image_size),
            in_channels=getattr(model, "in_channels", 1),
            base_channels=64,
            channel_multipliers=(1,2,4,8),
            num_res_blocks=2,
            attention_resolutions=(32,16),
            timesteps=desired_timesteps,
            lr_scale=args.scale,
            dropout=0.1
        )
        model = model.to(device)
        # reload weights into new model object
        load_checkpoint_for_model(args.checkpoint, model, ema_obj=ema, device=device)

    # move diffusion tensors and link model
    move_diffusion_tensors_to_device(diffusion, device)
    diffusion.model = model

    # apply EMA shadow if requested and available
    if args.use_ema and ema is not None:
        # Try load ema state (if not loaded above)
        if hasattr(ema, "shadow") and ema.shadow:
            print("Applying EMA shadow weights for evaluation.")
            ema.apply_shadow()
        else:
            # ckpt may not have ema_state; warn but continue
            print("Warning: --use_ema requested but no ema_state found in checkpoint; using model weights.")

    # run fast evaluation
    print(f"Evaluating {len(paths)} images (fast steps={args.num_steps}) on device {device}")
    out_preds_dir = os.path.join(args.save_dir, "preds") if args.save_predictions else None
    metrics = evaluate_fast(model, diffusion, loader, device,
                            num_images=len(paths),
                            num_steps=args.num_steps,
                            use_amp=args.use_amp,
                            save_preds=args.save_predictions,
                            out_dir=out_preds_dir)

    # restore model if EMA applied
    if args.use_ema and ema is not None and hasattr(ema, "backup") and ema.backup:
        ema.restore()

    # save metrics
    metrics_path = os.path.join(args.save_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(f"PSNR mean: {metrics['psnr_mean']:.4f}, SSIM mean: {metrics['ssim_mean']:.4f}")

if __name__ == "__main__":
    main()
