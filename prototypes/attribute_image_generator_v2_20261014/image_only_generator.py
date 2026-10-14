"""独立image-only G(I)：轻量Encoder、ResNet Blocks、Decoder。"""
import torch
from torch import nn
from torch.nn import functional as F
from budget_projection import DEFAULT_EPSILON, checked_epsilon, check_pixels, epsilon_tensor

def conv_block(in_channels, out_channels, kernel=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, kernel//2),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
        nn.SiLU())

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            conv_block(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=False))
    def forward(self, image):
        return image + self.body(image)

class ImageOnlyGenerator(nn.Module):
    def __init__(self, width=32, blocks=4, epsilon=DEFAULT_EPSILON):
        super().__init__()
        if type(width) is not int or width < 1 or type(blocks) is not int or blocks < 1:
            raise ValueError("positive integer width/blocks required")
        self.width, self.blocks = width, blocks
        self.epsilon = checked_epsilon(epsilon)
        self.encoder = conv_block(3, width, kernel=7)
        self.down1 = conv_block(width, 2*width, stride=2)
        self.down2 = conv_block(2*width, 4*width, stride=2)
        self.residual = nn.Sequential(*(ResidualBlock(4*width) for _ in range(blocks)))
        self.up1 = conv_block(4*width, 2*width)
        self.up2 = conv_block(2*width, width)
        self.output = nn.Conv2d(width, 3, 7, padding=3)

    def forward(self, image):
        check_pixels(image)
        if min(image.shape[2:]) < 8:
            raise ValueError("generator spatial dimensions must be at least 8")
        h0 = self.encoder(2*image - 1)
        h1 = self.down1(h0)
        hidden = self.residual(self.down2(h1))
        hidden = self.up1(F.interpolate(hidden, size=h1.shape[-2:], mode="bilinear", align_corners=False))
        hidden = self.up2(F.interpolate(hidden, size=image.shape[-2:], mode="bilinear", align_corners=False))
        return epsilon_tensor(image, self.epsilon) * torch.tanh(self.output(hidden))
