"""
models/esrgan.py
ESRGAN baseline model components and loss wrappers.

- RRDBNet (generator) - reduced-by-default for memory (nb param)
- Relativistic discriminator (simple PatchGAN-like)
- VGG perceptual loss (handles grayscale input by repeating to 3 channels)
- Optional SegmentationGuidedLoss that uses models.unet.UNet if available
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

# ------------------------------
# RRDB components (transparent, readable)
# ------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super().__init__()
        gc = growth_channels
        self.conv1 = nn.Conv2d(in_channels, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + gc*2, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + gc*3, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + gc*4, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, in_channels, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, gc)
        self.rdb2 = ResidualDenseBlock(in_channels, gc)
        self.rdb3 = ResidualDenseBlock(in_channels, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDBNet generator.
    in_nc/out_nc should match dataset (1 for grayscale MRI).
    nb can be increased (23 in original ESRGAN) but is reduced here for memory.
    """
    def __init__(self, in_nc=1, out_nc=1, nf=64, nb=8, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        body = self.body(fea)
        fea = self.conv_body(body) + fea
        out = self.conv_last(fea)
        return out


# ------------------------------
# Discriminator (PatchGAN-like)
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_nc=1, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1), nn.LeakyReLU(0.2, True),

            nn.Conv2d(nf, nf*2, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*2, nf*2, 4, 2, 1), nn.LeakyReLU(0.2, True),

            nn.Conv2d(nf*2, nf*4, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*4, nf*4, 4, 2, 1), nn.LeakyReLU(0.2, True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf*4, 1, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x).view(x.size(0))


# ------------------------------
# Perceptual loss (VGG)
# ------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, device='cpu', use_input_norm=True):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        # capture features up to relu4_4 (index ~23)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:23]).to(device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.use_input_norm = use_input_norm
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    def forward(self, x):
        # x: BxCxHxW ; if C==1 (grayscale), repeat to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.feature_extractor(x)


class PerceptualLoss(nn.Module):
    def __init__(self, device='cpu', weight=1.0):
        super().__init__()
        self.vgg = VGGFeatureExtractor(device=device)
        self.criterion = nn.L1Loss()
        self.weight = weight

    def forward(self, sr, hr):
        f_sr = self.vgg(sr)
        f_hr = self.vgg(hr)
        return self.criterion(f_sr, f_hr) * self.weight


# ------------------------------
# Relativistic GAN losses (RaGAN)
# ------------------------------
bce_loss = nn.BCEWithLogitsLoss()

def gan_loss_discriminator(d_real, d_fake):
    # d_real, d_fake: logits for real and fake
    real_loss = bce_loss(d_real - d_fake.mean(), torch.ones_like(d_real))
    fake_loss = bce_loss(d_fake - d_real.mean(), torch.zeros_like(d_fake))
    return (real_loss + fake_loss) / 2

def gan_loss_generator(d_real, d_fake):
    loss = bce_loss(d_real - d_fake.mean(), torch.zeros_like(d_real)) + \
           bce_loss(d_fake - d_real.mean(), torch.ones_like(d_fake))
    return loss / 2


# ------------------------------
# Optional segmentation-guided loss (Dice) using your UNet if present
# ------------------------------
class SegmentationGuidedLoss(nn.Module):
    """
    If you want to use SegGuided training:
    - place your UNet implementation in models/unet.py (or ensure importable from models.unet)
    - pass an instance of that UNet (pretrained) to this class.
    """
    def __init__(self, unet_model=None, weight=0.2):
        super().__init__()
        self.unet = unet_model
        self.weight = weight
        if self.unet is not None:
            self.unet.eval()
            for p in self.unet.parameters():
                p.requires_grad = False

    def dice_loss(self, pred, target, eps=1e-6):
        # pred, target -> BxCxHxW
        num = 2 * (pred * target).sum(dim=(2,3))
        den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
        dice = 1 - (num / den)
        return dice.mean()

    def forward(self, sr, hr):
        # expects sr/hr normalized in [0,1]; UNet should output logits or probabilities
        if self.unet is None:
            return torch.tensor(0.0, device=sr.device)
        with torch.no_grad():
            seg_hr = torch.sigmoid(self.unet(hr))
        seg_sr = torch.sigmoid(self.unet(sr))
        # if multiple output channels, you may need to adapt to class-wise dice
        if seg_hr.shape[1] > 1:
            seg_hr = torch.softmax(seg_hr, dim=1)
            seg_sr = torch.softmax(seg_sr, dim=1)
        return self.weight * self.dice_loss(seg_sr, seg_hr)

# ------------------------------
# Exportable items
# ------------------------------
__all__ = ['RRDBNet', 'Discriminator', 'PerceptualLoss', 'SegmentationGuidedLoss',
           'gan_loss_discriminator', 'gan_loss_generator']
