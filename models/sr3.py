"""
SR3: Image Super-Resolution via Iterative Refinement
Implementation for medical image super-resolution baseline
Based on: Saharia et al. (2021) - https://arxiv.org/abs/2104.07636

This implementation provides a diffusion-based baseline for comparison
with the main DiT/SiT approach in the SpinOff project.

FIXED: Proper handling of LR (128x128) -> HR (256x256) super-resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal time step embeddings for diffusion models.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding and conditional input.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        use_attention: bool = False
    ):
        super().__init__()
        self.use_attention = use_attention
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        # Self-attention (optional)
        if use_attention:
            self.attention = SelfAttention(out_channels)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        
        # Skip connection
        h = h + self.skip(x)
        
        # Apply attention if specified
        if self.use_attention:
            h = self.attention(h)
        
        return h


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for spatial features.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.matmul(attn, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Project
        h = self.proj(h)
        
        # Residual
        return x + h


class DownBlock(nn.Module):
    """
    Downsampling block with residual connections.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        use_attention: bool = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                use_attention=use_attention
            )
            for i in range(num_layers)
        ])
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block with residual connections and skip connections.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        use_attention: bool = False
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.layers = nn.ModuleList([
            ResidualBlock(
                in_channels + out_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                use_attention=use_attention
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for layer in self.layers:
            x = layer(x, time_emb)
        return x


class MiddleBlock(nn.Module):
    """
    Middle block with attention.
    """
    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.block1 = ResidualBlock(channels, channels, time_emb_dim, use_attention=True)
        self.block2 = ResidualBlock(channels, channels, time_emb_dim, use_attention=False)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        return x


class SR3UNet(nn.Module):
    """
    U-Net architecture for SR3 diffusion model.
    Conditions on low-resolution images for super-resolution.
    
    FIXED: Properly handles LR (H/scale, W/scale) -> HR (H, W) super-resolution
    by upsampling LR to match HR resolution before concatenation.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.1,
        image_size: int = 256,  # HR image size
        lr_scale: int = 2       # LR is image_size // lr_scale
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size  # HR size (e.g., 256)
        self.lr_scale = lr_scale      # Scale factor (e.g., 2 means LR is 128)
        self.lr_size = image_size // lr_scale  # LR size (e.g., 128)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution for noisy high-res input (concatenated with upsampled LR)
        # Input: noisy HR (in_channels) + upsampled LR (in_channels) = 2 * in_channels
        self.init_conv = nn.Conv2d(in_channels * 2, base_channels, 3, padding=1)
        
        # Conditioning: upsample LR to HR resolution
        # LR (H/2, W/2) -> HR (H, W) using learned upsampling
        self.cond_upsample = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )
        
        # Downsampling path
        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            use_attention = (image_size // (2 ** i)) in attention_resolutions
            
            self.downs.append(
                DownBlock(
                    now_channels,
                    out_ch,
                    time_emb_dim,
                    num_res_blocks,
                    use_attention
                )
            )
            now_channels = out_ch
            channels.append(now_channels)
        
        # Middle
        self.middle = MiddleBlock(now_channels, time_emb_dim)
        
        # Upsampling path
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            use_attention = (image_size // (2 ** (len(channel_multipliers) - i - 1))) in attention_resolutions
            
            self.ups.append(
                UpBlock(
                    now_channels,
                    out_ch,
                    time_emb_dim,
                    num_res_blocks + 1,
                    use_attention
                )
            )
            now_channels = out_ch
        
        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of SR3 U-Net.
        
        Args:
            x: Noisy high-resolution image [B, C, H, W] (e.g., 256x256)
            time: Diffusion timestep [B]
            cond: Low-resolution conditioning image [B, C, H/scale, W/scale] (e.g., 128x128)
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        time_emb = self.time_mlp(time)
        
        # Upsample LR conditioning to match HR resolution
        # cond: [B, C, 128, 128] -> cond_up: [B, C, 256, 256]
        cond_up = self.cond_upsample(cond)
        
        # Concatenate noisy HR with upsampled LR (channel-wise)
        # This is the standard SR3 approach: condition by concatenation
        x = torch.cat([x, cond_up], dim=1)  # [B, 2*C, H, W]
        
        # Initial projection
        x = self.init_conv(x)  # [B, base_channels, H, W]
        
        # Encoder
        skips = []
        for down in self.downs:
            x, skip = down(x, time_emb)
            skips.append(skip)
        
        # Middle
        x = self.middle(x, time_emb)
        
        # Decoder
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, time_emb)
        
        # Final
        x = self.final(x)
        
        return x


class GaussianDiffusion:
    """
    Gaussian diffusion process for SR3.
    Implements training and sampling procedures.
    """
    def __init__(
        self,
        model: SR3UNet,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        self.model = model
        self.timesteps = timesteps
        
        # Define beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute diffusion parameters
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in Nichol & Dhariwal (2021).
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate training loss.
        
        Args:
            x_start: High-resolution ground truth [B, C, H, W] (e.g., 256x256)
            cond: Low-resolution conditioning [B, C, H/scale, W/scale] (e.g., 128x128)
            t: Timestep [B]
            noise: Optional noise tensor
        
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion on HR image
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise (model will upsample LR internally)
        predicted_noise = self.model(x_noisy, t, cond)
        
        # Simple MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: int,
        t_index: int
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from x_t.
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Create time tensor
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = self.model(x, t_tensor, cond)
        
        # Extract parameters
        betas_t = self._extract(self.betas, t_tensor, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape
        )
        sqrt_recip_alphas_t = self._extract(
            torch.sqrt(1.0 / self.alphas), t_tensor, x.shape
        )
        
        # Equation 11 in DDPM paper
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t_tensor, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, ...],
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        Generate samples through reverse diffusion process.
        
        Args:
            cond: Low-resolution conditioning image [B, C, H/scale, W/scale]
            shape: Shape of output (B, C, H, W) - HR resolution
            return_all_timesteps: Whether to return intermediate timesteps
        
        Returns:
            Generated high-resolution image
        """
        device = cond.device
        batch_size = shape[0]
        
        # Start from pure noise at HR resolution
        img = torch.randn(shape, device=device)
        
        imgs = []
        
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, cond, i, i)
            if return_all_timesteps:
                imgs.append(img)
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        else:
            return img
    
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Generate super-resolved images.
        
        Args:
            cond: Low-resolution input [B, C, H_lr, W_lr] (e.g., 128x128)
            batch_size: Batch size (should match cond batch size)
        
        Returns:
            Super-resolved image [B, C, H_hr, W_hr] (e.g., 256x256)
        """
        image_size = self.model.image_size  # HR size
        channels = self.model.in_channels
        
        return self.p_sample_loop(
            cond,
            shape=(batch_size, channels, image_size, image_size)
        )
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extract values from a 1-D tensor for a batch of indices.
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def create_sr3_model(
    image_size: int = 256,      # HR output size
    lr_size: int = 128,         # LR input size (optional, computed from scale)
    in_channels: int = 1,
    base_channels: int = 64,
    timesteps: int = 1000,
    beta_schedule: str = 'linear',
    lr_scale: int = 2           # Upsampling factor
) -> Tuple[SR3UNet, GaussianDiffusion]:
    """
    Factory function to create SR3 model and diffusion process.
    
    Args:
        image_size: Size of high-resolution output (e.g., 256)
        lr_size: Size of low-resolution input (optional, default = image_size // lr_scale)
        in_channels: Number of input channels
        base_channels: Base number of channels in U-Net
        timesteps: Number of diffusion steps
        beta_schedule: Noise schedule ('linear' or 'cosine')
        lr_scale: Upsampling scale factor (2 for 2x SR)
    
    Returns:
        Tuple of (model, diffusion)
    """
    model = SR3UNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=base_channels * 4,
        attention_resolutions=(16,),
        dropout=0.1,
        image_size=image_size,  # HR size
        lr_scale=lr_scale
    )
    
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=timesteps,
        beta_schedule=beta_schedule
    )
    
    return model, diffusion


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Testing SR3 Model: LR (128x128) -> HR (256x256)")
    print("=" * 60)
    
    # Create model for 2x super-resolution
    # LR: 128x128, HR: 256x256
    model, diffusion = create_sr3_model(
        image_size=256,      # HR output size
        in_channels=1,
        base_channels=64,
        timesteps=1000,
        lr_scale=2           # 2x upsampling
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
    
    # Test forward pass
    batch_size = 2
    x_hr = torch.randn(batch_size, 1, 256, 256).to(device)  # High-res ground truth
    x_lr = torch.randn(batch_size, 1, 128, 128).to(device)  # Low-res condition
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    print(f"\nInput shapes:")
    print(f"  LR (condition): {x_lr.shape}")
    print(f"  HR (target):    {x_hr.shape}")
    
    # Test training
    loss = diffusion.p_losses(x_hr, x_lr, t)
    print(f"\nTraining loss: {loss.item():.4f}")
    
    # Test sampling (reduced timesteps for speed)
    print("\nTesting sampling...")
    with torch.no_grad():
        # Quick test with subset of steps
        samples = diffusion.sample(x_lr, batch_size=batch_size)
    print(f"Sample shape: {samples.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ SR3 model test passed!")
