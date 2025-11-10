"""
U-Net Architecture for Super-Resolution
Based on original U-Net paper with modifications for SR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers with ReLU activation
    (Conv2d -> BatchNorm -> ReLU) x 2
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block with conv transpose then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                        kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Output from previous layer (lower resolution)
            x2: Skip connection from encoder (same resolution as output)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch (if input size is not perfectly divisible)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for Super-Resolution
    
    Input: Low-resolution image (e.g., 128x128)
    Output: High-resolution image (e.g., 256x256)
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: list = [64, 128, 256, 512],
                 bilinear: bool = False):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output channels (1 for grayscale)
            features: List of feature dimensions for each layer
            bilinear: Use bilinear upsampling instead of transposed conv
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Initial convolution (no downsampling)
        self.inc = DoubleConv(in_channels, features[0])
        
        # Encoder (downsampling path)
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        
        # Extra upsampling for super-resolution (2x)
        self.sr_upsample = nn.Sequential(
            nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.outc = OutConv(features[0], out_channels)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input LR image (B, C, H, W)
            
        Returns:
            HR image (B, C, H*2, W*2)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Super-resolution upsampling
        x = self.sr_upsample(x)
        
        # Final output
        logits = self.outc(x)
        
        return torch.sigmoid(logits)  # Output in [0, 1]


class UNetSimple(nn.Module):
    """
    Simpler U-Net for faster training and inference
    Fewer layers and parameters
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: list = [32, 64, 128]):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[2], features[2] * 2)
        
        # Decoder
        self.up1 = Up(features[2] * 2, features[2])
        self.up2 = Up(features[2], features[1])
        self.up3 = Up(features[1], features[0])
        
        # SR upsampling
        self.sr_upsample = nn.ConvTranspose2d(features[0], features[0],
                                             kernel_size=2, stride=2)
        
        # Output
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # SR
        x = self.sr_upsample(x)
        x = self.outc(x)
        
        return torch.sigmoid(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    print("Testing U-Net models...")
    
    # Create models
    model = UNet(in_channels=1, out_channels=1)
    model_simple = UNetSimple(in_channels=1, out_channels=1)
    
    # Count parameters
    print(f"\nFull U-Net parameters: {count_parameters(model):,}")
    print(f"Simple U-Net parameters: {count_parameters(model_simple):,}")
    
    # Test forward pass
    batch_size = 2
    lr_input = torch.randn(batch_size, 1, 128, 128)
    
    with torch.no_grad():
        output = model(lr_input)
        output_simple = model_simple(lr_input)
    
    print(f"\nInput shape: {lr_input.shape}")
    print(f"Full U-Net output shape: {output.shape}")
    print(f"Simple U-Net output shape: {output_simple.shape}")
    
    # Check output values
    print(f"\nOutput value range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Memory footprint (approximate)
    def get_model_size_mb(model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024**2
    
    print(f"\nFull U-Net size: {get_model_size_mb(model):.2f} MB")
    print(f"Simple U-Net size: {get_model_size_mb(model_simple):.2f} MB")
    
    print("\n✅ All models working correctly!")
