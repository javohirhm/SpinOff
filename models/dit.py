"""
DiT: Scalable Diffusion Models with Transformers
Implementation for medical image super-resolution

Based on: Peebles & Xie (2023) - "Scalable Diffusion Models with Transformers"
https://arxiv.org/abs/2212.09748

This implementation provides a transformer-based diffusion model for
super-resolution, adapted from the original DiT for class-conditional generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange


# =============================================================================
# Core Building Blocks
# =============================================================================

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


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    Converts image into sequence of patches.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class Unpatchify(nn.Module):
    """
    Converts patch embeddings back to image.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        out_channels: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.out_channels = out_channels
        
        # Project to patch pixels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_patches, embed_dim]
        Returns:
            [B, C, H, W]
        """
        x = self.proj(x)  # [B, num_patches, patch_size^2 * C]
        x = rearrange(
            x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.grid_size, w=self.grid_size,
            p1=self.patch_size, p2=self.patch_size,
            c=self.out_channels
        )
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# =============================================================================
# AdaLN (Adaptive Layer Normalization) - Key DiT Innovation
# =============================================================================

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization with learnable scale and shift.
    Conditions the layer norm on timestep and conditioning embeddings.
    """
    
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # Project conditioning to scale (gamma) and shift (beta)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2)
        )
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]
            cond: [B, cond_dim]
        Returns:
            [B, N, dim]
        """
        # Get scale and shift
        scale_shift = self.proj(cond)  # [B, dim * 2]
        scale, shift = scale_shift.chunk(2, dim=-1)  # [B, dim] each
        
        # Apply adaptive normalization
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero: Initializes residual connections to zero.
    This is the key modification that makes DiT training stable.
    """
    
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # Project to scale, shift, and gate (alpha)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 3)
        )
        # Initialize to zero for stable training
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns normalized x and gate value for residual connection.
        """
        params = self.proj(cond)  # [B, dim * 3]
        scale, shift, gate = params.chunk(3, dim=-1)
        
        x_norm = self.norm(x)
        x_norm = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x_norm, gate.unsqueeze(1)


# =============================================================================
# DiT Block
# =============================================================================

class DiTBlock(nn.Module):
    """
    DiT Transformer Block with AdaLN-Zero conditioning.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Attention with AdaLN-Zero
        self.adaln_attn = AdaLNZero(dim, cond_dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        
        # FFN with AdaLN-Zero
        self.adaln_ffn = AdaLNZero(dim, cond_dim)
        self.ffn = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] - patch embeddings
            cond: [B, cond_dim] - conditioning (time + LR condition)
        Returns:
            [B, N, dim]
        """
        # Attention block with AdaLN-Zero
        x_norm, gate = self.adaln_attn(x, cond)
        x = x + gate * self.attn(x_norm)
        
        # FFN block with AdaLN-Zero
        x_norm, gate = self.adaln_ffn(x, cond)
        x = x + gate * self.ffn(x_norm)
        
        return x


# =============================================================================
# Final Layer with AdaLN
# =============================================================================

class FinalLayer(nn.Module):
    """
    Final layer of DiT with AdaLN conditioning.
    """
    
    def __init__(self, dim: int, cond_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = AdaLN(dim, cond_dim)
        self.proj = nn.Linear(dim, patch_size * patch_size * out_channels)
        
        # Initialize to zero
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x, cond)
        x = self.proj(x)
        return x


# =============================================================================
# Main DiT Model for Super-Resolution
# =============================================================================

class DiTSR(nn.Module):
    """
    Diffusion Transformer for Super-Resolution.
    
    Conditions on low-resolution images by:
    1. Upsampling LR to HR resolution
    2. Concatenating with noisy HR (channel-wise)
    3. Using combined conditioning for AdaLN
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        image_size: int = 256,      # HR image size
        patch_size: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        lr_scale: int = 2           # LR is image_size // lr_scale
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.lr_scale = lr_scale
        self.lr_size = image_size // lr_scale
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Time embedding
        time_dim = embed_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # LR condition encoder: upsample LR to HR, then encode
        self.cond_upsample = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 4, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, in_channels, 3, padding=1)
        )
        
        # Patch embedding for concatenated input (noisy HR + upsampled LR)
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels * 2,  # Concatenated channels
            embed_dim=embed_dim
        )
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Condition embedding projection (time -> cond_dim)
        cond_dim = time_dim
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(
            dim=embed_dim,
            cond_dim=cond_dim,
            patch_size=patch_size,
            out_channels=out_channels
        )
        
        # For unpatchifying
        self.grid_size = image_size // patch_size
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following DiT paper."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)
        
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.patch_embed.proj.bias)
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch embeddings back to image.
        x: [B, num_patches, patch_size^2 * out_channels]
        """
        x = rearrange(
            x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.grid_size, w=self.grid_size,
            p1=self.patch_size, p2=self.patch_size,
            c=self.out_channels
        )
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of DiT for SR.
        
        Args:
            x: Noisy HR image [B, C, H, W] (e.g., 256x256)
            time: Diffusion timestep [B]
            cond: LR conditioning image [B, C, H/scale, W/scale] (e.g., 128x128)
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        t_emb = self.time_mlp(time)  # [B, cond_dim]
        
        # Upsample LR condition to HR resolution
        cond_up = self.cond_upsample(cond)  # [B, C, H, W]
        
        # Concatenate noisy HR with upsampled LR
        x_concat = torch.cat([x, cond_up], dim=1)  # [B, 2*C, H, W]
        
        # Patchify
        x = self.patch_embed(x_concat)  # [B, num_patches, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)
        
        # Final layer
        x = self.final_layer(x, t_emb)  # [B, num_patches, patch_size^2 * out_channels]
        
        # Unpatchify
        x = self.unpatchify(x)  # [B, out_channels, H, W]
        
        return x


# =============================================================================
# Gaussian Diffusion for DiT
# =============================================================================

class GaussianDiffusion:
    """
    Gaussian diffusion process for DiT.
    Same as SR3 diffusion but works with DiT model.
    """
    
    def __init__(
        self,
        model: DiTSR,
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
        
        # Calculations for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in Nichol & Dhariwal (2021)."""
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
        """Forward diffusion process: q(x_t | x_0)"""
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
            x_start: HR ground truth [B, C, H, W]
            cond: LR conditioning [B, C, H/scale, W/scale]
            t: Timestep [B]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t, cond)
        
        # MSE loss
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
        
        # DDPM sampling
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
        """Generate samples through reverse diffusion."""
        device = cond.device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        imgs = []
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, cond, i, i)
            if return_all_timesteps:
                imgs.append(img)
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        return img
    
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        batch_size: int = 1
    ) -> torch.Tensor:
        """Generate super-resolved images."""
        image_size = self.model.image_size
        channels = self.model.in_channels
        
        return self.p_sample_loop(
            cond,
            shape=(batch_size, channels, image_size, image_size)
        )
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """Extract values from 1-D tensor for batch of indices."""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# =============================================================================
# Model Configurations (following DiT paper)
# =============================================================================

def DiT_XL_2(**kwargs):
    """DiT-XL/2: 118M params (patch_size=2)"""
    return DiTSR(patch_size=2, embed_dim=1152, depth=28, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    """DiT-XL/4: 118M params (patch_size=4)"""
    return DiTSR(patch_size=4, embed_dim=1152, depth=28, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    """DiT-L/2: 48M params"""
    return DiTSR(patch_size=2, embed_dim=1024, depth=24, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    """DiT-L/4: 48M params"""
    return DiTSR(patch_size=4, embed_dim=1024, depth=24, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    """DiT-B/2: 22M params"""
    return DiTSR(patch_size=2, embed_dim=768, depth=12, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    """DiT-B/4: 22M params"""
    return DiTSR(patch_size=4, embed_dim=768, depth=12, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    """DiT-S/2: 6M params"""
    return DiTSR(patch_size=2, embed_dim=384, depth=12, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    """DiT-S/4: 6M params"""
    return DiTSR(patch_size=4, embed_dim=384, depth=12, num_heads=6, **kwargs)


# =============================================================================
# Factory Function
# =============================================================================

def create_dit_model(
    image_size: int = 256,
    in_channels: int = 1,
    patch_size: int = 4,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    timesteps: int = 1000,
    beta_schedule: str = 'linear',
    lr_scale: int = 2,
    model_type: str = None
) -> Tuple[DiTSR, GaussianDiffusion]:
    """
    Factory function to create DiT model and diffusion process.
    
    Args:
        image_size: HR output size
        in_channels: Number of input channels
        patch_size: Patch size for ViT
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        timesteps: Number of diffusion steps
        beta_schedule: Noise schedule
        lr_scale: Upsampling factor
        model_type: Optional preset ('S', 'B', 'L', 'XL')
    """
    # Use preset if specified
    if model_type:
        presets = {
            'S': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
            'B': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
            'L': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
            'XL': {'embed_dim': 1152, 'depth': 28, 'num_heads': 16},
        }
        if model_type in presets:
            embed_dim = presets[model_type]['embed_dim']
            depth = presets[model_type]['depth']
            num_heads = presets[model_type]['num_heads']
    
    model = DiTSR(
        in_channels=in_channels,
        out_channels=in_channels,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
        lr_scale=lr_scale
    )
    
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=timesteps,
        beta_schedule=beta_schedule
    )
    
    return model, diffusion


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Testing DiT Model: LR (128x128) -> HR (256x256)")
    print("=" * 60)
    
    # Create DiT-B/4 model
    model, diffusion = create_dit_model(
        image_size=256,
        in_channels=1,
        patch_size=4,
        model_type='B',
        timesteps=1000,
        lr_scale=2
    )
    model = model.to(device)
    
    # Move diffusion parameters to device
    for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))
    
    # Test shapes
    batch_size = 2
    x_hr = torch.randn(batch_size, 1, 256, 256).to(device)
    x_lr = torch.randn(batch_size, 1, 128, 128).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    print(f"\nInput shapes:")
    print(f"  LR (condition): {x_lr.shape}")
    print(f"  HR (target):    {x_hr.shape}")
    
    # Test training
    loss = diffusion.p_losses(x_hr, x_lr, t)
    print(f"\nTraining loss: {loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test sampling (just forward pass, not full loop for speed)
    print("\nTesting single denoising step...")
    with torch.no_grad():
        noise_pred = model(x_hr, t, x_lr)
    print(f"Noise prediction shape: {noise_pred.shape}")
    
    print("\n✅ DiT model test passed!")
