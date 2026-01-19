import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math


class GaussianBlurLayer(nn.Module):
    """
    A1-1: GPU 可控、可导的高斯模糊层
    """

    def __init__(self, kernel_size=5, sigma=1.0, channels=3):
        super(GaussianBlurLayer, self).__init__()
        self.padding = kernel_size // 2

        # 生成高斯核
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # 重塑为卷积权重: [channels, 1, k, k] (Depthwise Conv)
        kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        self.register_buffer('weight', kernel.repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)


class LocalPatchBranch(nn.Module):
    def __init__(self, patch_size=32, embed_dim=256, pretrained=False, num_patches=32):
        super(LocalPatchBranch, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches  # A1-2: 采样数量 M

        # A1-1: GPU Blur
        self.blur = GaussianBlurLayer(kernel_size=5, sigma=1.0)

        # Backbone
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_resnet = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-2])

        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # A1-3: Instance Norm 用于残差
        self.res_norm = nn.InstanceNorm2d(3, affine=False)

    def extract_patches(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(b, -1, c, p, p)
        return patches

    def sample_patches(self, patches):
        """A1-2: 随机采样 M 个 Patch"""
        b, n, c, h, w = patches.shape
        if n <= self.num_patches:
            return patches

        if self.training:
            # 训练时随机采样
            indices = torch.stack([torch.randperm(n, device=patches.device)[:self.num_patches] for _ in range(b)])
            # [B, M, C, H, W]
            sampled = torch.stack([patches[i][indices[i]] for i in range(b)])
            return sampled
        else:
            # 推理时固定采样中心或间隔采样，保证确定性
            # 这里简单取前 M 个
            return patches[:, :self.num_patches]

    def forward(self, x):
        # A1-1 & A1-3
        with torch.no_grad():
            x_blur = self.blur(x)
            x_res = x - x_blur
            x_res = self.res_norm(x_res)  # 强制高频规范化

        # Extract & Sample
        patches = self.extract_patches(x_res)
        patches = self.sample_patches(patches)  # A1-2

        # Encode
        b, n, c, h, w = patches.shape

        # 【修复点】使用 reshape 替代 view，自动处理非连续内存
        patches_flat = patches.reshape(-1, c, h, w)

        feat = self.backbone(patches_flat)
        if feat.size(-1) > 1:
            feat = F.adaptive_avg_pool2d(feat, (1, 1))

        feat = feat.view(feat.size(0), -1)
        out = self.proj(feat)  # [B, M, D]

        # 恢复 Batch 维度
        out = out.view(b, n, -1)

        return out