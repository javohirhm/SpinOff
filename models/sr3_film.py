"""
SR3-FiLM: Optimized Diffusion Model for Medical Image Super-Resolution

Architecture: U-Net backbone with attention mechanisms
Conditioning: Feature-wise Linear Modulation (FiLM) for LR image conditioning
Optimized for: Medical imaging data (MRI)

Key Features:
- FiLM conditioning instead of concatenation (more parameter efficient)
- Attention at multiple resolutions
- Optimized for 128x128 patch-based training
- Compatible with mixed-precision training (FP16)

Reference: SR3 (Saharia et al., 2021) with FiLM conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ============================================================================
# Time Embeddings
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time step embeddings for diffusion models."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """Time embedding MLP."""
    
    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


# ============================================================================
# FiLM Conditioning Module
# ============================================================================

class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for conditioning.
    Encodes the low-resolution image and produces scale/shift parameters
    for each resolution level in the U-Net.
    """
    
    def __init__(self, in_channels: int, base_channels: int, channel_multipliers: Tuple[int, ...]):
        super().__init__()
        
        self.channel_multipliers = channel_multipliers
        self.base_channels = base_channels
        
        # Initial conv to process LR image
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        
        # Encoder blocks (downsampling)
        self.encoder = nn.ModuleList()
        ch = base_channels
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            self.encoder.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU()
            ))
            ch = out_ch
        
        # FiLM generators for each level (encoder + decoder have same channel counts at each level)
        # We need FiLM params for: encoder levels + middle + decoder levels
        # Channel sizes at each level: base*1, base*2, base*4, base*8 (for mults 1,2,4,8)
        self.film_generators = nn.ModuleDict()
        for mult in channel_multipliers:
            ch_size = base_channels * mult
            self.film_generators[str(ch_size)] = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch_size, ch_size * 2)  # gamma and beta
            )
    
    def forward(self, lr_image: torch.Tensor) -> dict:
        """
        Returns dict mapping channel_size -> (gamma, beta) for FiLM modulation.
        """
        film_params = {}
        
        x = self.init_conv(lr_image)
        
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            ch_size = x.shape[1]
            
            # Generate FiLM params for this channel size
            if str(ch_size) in self.film_generators:
                params = self.film_generators[str(ch_size)](x)
                gamma, beta = params.chunk(2, dim=-1)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
                film_params[ch_size] = (gamma, beta)
        
        return film_params


# ============================================================================
# Core Building Blocks
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with time embedding and optional FiLM conditioning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        use_film: bool = False
    ):
        super().__init__()
        self.use_film = use_film
        self.out_channels = out_channels
        
        # First conv block
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second conv block
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Initialize last conv to zero for better training dynamics
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor,
        film_params: Optional[dict] = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        
        # Apply FiLM conditioning if provided
        if self.use_film and film_params is not None:
            ch_size = h.shape[1]
            if ch_size in film_params:
                gamma, beta = film_params[ch_size]
                h = gamma * h + beta
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """
    Efficient self-attention for spatial features.
    Uses memory-efficient attention when available.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # Initialize to zero for residual connection
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for attention
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Efficient attention (uses Flash Attention if available)
        if hasattr(F, 'scaled_dot_product_attention'):
            h = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            h = torch.matmul(attn, v)
        
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


# ============================================================================
# U-Net Encoder/Decoder Blocks
# ============================================================================

class DownBlock(nn.Module):
    """Downsampling block with residual connections and optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        use_film: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            self.res_blocks.append(
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    time_emb_dim,
                    dropout,
                    use_film=use_film
                )
            )
            if use_attention:
                self.attention_blocks.append(SelfAttention(out_channels))
            else:
                self.attention_blocks.append(nn.Identity())
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor,
        film_params: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for res_block, attn_block in zip(self.res_blocks, self.attention_blocks):
            x = res_block(x, time_emb, film_params)
            x = attn_block(x)
        
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with residual connections, skip connections, and optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        use_film: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        self.res_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = in_channels + skip_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, time_emb_dim, dropout, use_film=use_film)
            )
            if use_attention:
                self.attention_blocks.append(SelfAttention(out_channels))
            else:
                self.attention_blocks.append(nn.Identity())
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip: torch.Tensor, 
        time_emb: torch.Tensor,
        film_params: Optional[dict] = None
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        
        for res_block, attn_block in zip(self.res_blocks, self.attention_blocks):
            x = res_block(x, time_emb, film_params)
            x = attn_block(x)
        
        return x


class MiddleBlock(nn.Module):
    """Middle block with attention."""
    
    def __init__(self, channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_emb_dim, dropout, use_film=True)
        self.attn = SelfAttention(channels)
        self.res2 = ResidualBlock(channels, channels, time_emb_dim, dropout, use_film=True)
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor,
        film_params: Optional[dict] = None
    ) -> torch.Tensor:
        x = self.res1(x, time_emb, film_params)
        x = self.attn(x)
        x = self.res2(x, time_emb, film_params)
        return x


# ============================================================================
# Main SR3-FiLM Model
# ============================================================================

class SR3FiLM(nn.Module):
    """
    SR3 with FiLM conditioning for super-resolution.
    
    Architecture:
    - U-Net backbone with attention at specified resolutions
    - FiLM conditioning for efficient LR image incorporation
    - Optimized for medical image super-resolution
    
    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        out_channels: Number of output channels
        base_channels: Base channel count (scales with multipliers)
        channel_multipliers: Channel scaling at each resolution
        num_res_blocks: Residual blocks per resolution level
        attention_resolutions: Image sizes where attention is applied
        dropout: Dropout rate
        image_size: Target HR image size
        lr_scale: Super-resolution scale factor
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (32, 16),
        dropout: float = 0.1,
        image_size: int = 256,
        lr_scale: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.lr_scale = lr_scale
        self.lr_size = image_size // lr_scale
        
        time_emb_dim = base_channels * 4
        
        # Time embedding
        self.time_embedding = TimeEmbedding(base_channels, time_emb_dim)
        
        # FiLM conditioner for LR image
        self.film_conditioner = FiLMConditioner(in_channels, base_channels, channel_multipliers)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        current_res = image_size
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            use_attn = current_res in attention_resolutions
            
            self.encoder_blocks.append(
                DownBlock(
                    now_channels, out_ch, time_emb_dim,
                    num_layers=num_res_blocks,
                    use_attention=use_attn,
                    use_film=True,
                    dropout=dropout
                )
            )
            now_channels = out_ch
            channels.append(now_channels)
            current_res //= 2
        
        # Middle
        self.middle = MiddleBlock(now_channels, time_emb_dim, dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        num_levels = len(channel_multipliers)
        
        for i in range(num_levels):
            # Skip channels from encoder (popped in reverse order)
            skip_ch = channels[num_levels - i]
            
            # Output channels
            if i < num_levels - 1:
                out_ch = base_channels * channel_multipliers[num_levels - 2 - i]
            else:
                out_ch = base_channels
            
            current_res *= 2
            use_attn = current_res in attention_resolutions
            
            self.decoder_blocks.append(
                UpBlock(
                    now_channels, out_ch, skip_ch, time_emb_dim,
                    num_layers=num_res_blocks + 1,
                    use_attention=use_attn,
                    use_film=True,
                    dropout=dropout
                )
            )
            now_channels = out_ch
        
        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(min(32, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
        # Initialize final conv to zero
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy HR image [B, C, H, W]
            time: Diffusion timestep [B]
            cond: LR conditioning image [B, C, H/scale, W/scale]
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        time_emb = self.time_embedding(time)
        
        # Get FiLM parameters from LR image (dict: channel_size -> (gamma, beta))
        film_params = self.film_conditioner(cond)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder
        skips = []
        for encoder in self.encoder_blocks:
            x, skip = encoder(x, time_emb, film_params)
            skips.append(skip)
        
        # Middle
        x = self.middle(x, time_emb, film_params)
        
        # Decoder
        for decoder in self.decoder_blocks:
            skip = skips.pop()
            x = decoder(x, skip, time_emb, film_params)
        
        # Final
        x = self.final(x)
        
        return x


# ============================================================================
# Gaussian Diffusion
# ============================================================================

class GaussianDiffusion:
    """
    Gaussian diffusion process for SR3-FiLM.
    
    Implements:
    - Forward process: Fixed Markov chain adding Gaussian noise over T steps
    - Reverse process: Network learns to denoise conditioned on LR input
    - Loss: Simplified variational bound (predict noise)
    """
    
    def __init__(
        self,
        model: SR3FiLM,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        self.model = model
        self.timesteps = timesteps
        
        # Beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute diffusion parameters
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Posterior
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
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
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_t * x_start + sqrt_one_minus_t * noise
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training loss (simplified variational bound).
        Predicts the noise added to x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, cond)
        
        # Simple MSE loss on noise prediction
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
        """Sample x_{t-1} from x_t."""
        device = x.device
        batch_size = x.shape[0]
        
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        predicted_noise = self.model(x, t_tensor, cond)
        
        betas_t = self._extract(self.betas, t_tensor, x.shape)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t_tensor, x.shape)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_var_t = self._extract(self.posterior_variance, t_tensor, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var_t) * noise
    
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        batch_size: int = 1
    ) -> torch.Tensor:
        """Generate super-resolved images."""
        device = cond.device
        image_size = self.model.image_size
        channels = self.model.in_channels
        
        img = torch.randn(batch_size, channels, image_size, image_size, device=device)
        
        for i in reversed(range(self.timesteps)):
            img = self.p_sample(img, cond, i, i)
        
        return img
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# ============================================================================
# Factory Function
# ============================================================================

def create_sr3_film_model(
    image_size: int = 256,
    in_channels: int = 1,
    base_channels: int = 64,
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
    num_res_blocks: int = 2,
    attention_resolutions: Tuple[int, ...] = (32, 16),
    timesteps: int = 1000,
    beta_schedule: str = 'linear',
    lr_scale: int = 2,
    dropout: float = 0.1
) -> Tuple[SR3FiLM, GaussianDiffusion]:
    """
    Factory function to create SR3-FiLM model and diffusion process.
    """
    model = SR3FiLM(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        image_size=image_size,
        lr_scale=lr_scale
    )
    
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=timesteps,
        beta_schedule=beta_schedule
    )
    
    return model, diffusion


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Testing SR3-FiLM Model")
    print("=" * 60)
    
    # Create model
    model, diffusion = create_sr3_film_model(
        image_size=256,
        in_channels=1,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(32, 16),
        timesteps=1000,
        lr_scale=2
    )
    model = model.to(device)
    
    # Move diffusion to device
    for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))
    
    # Test data
    batch_size = 2
    x_hr = torch.randn(batch_size, 1, 256, 256).to(device)
    x_lr = torch.randn(batch_size, 1, 128, 128).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    print(f"\nInput shapes:")
    print(f"  LR: {x_lr.shape}")
    print(f"  HR: {x_hr.shape}")
    
    # Test forward
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(x_hr, t, x_lr)
    print(f"  Output: {output.shape}")
    
    # Test loss
    print("\nTesting loss...")
    loss = diffusion.p_losses(x_hr, x_lr, t)
    print(f"  Loss: {loss.item():.4f}")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total:,}")
    print(f"Size: {total * 4 / 1024 / 1024:.2f} MB")
    
    print("\n✅ Test passed!")
