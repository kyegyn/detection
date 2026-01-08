import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class LocalPatchBranch(nn.Module):
    """
    支路二：局部纹理取证模块
    功能：通过 Patch 打乱（Shuffle）破坏全局语义，利用 CNN 提取局部伪影特征。
    """

    def __init__(self, patch_size=32, embed_dim=256, pretrained=False):
        super(LocalPatchBranch, self).__init__()
        self.patch_size = patch_size

        # 1. 局部特征提取器 (Backbone)
        # 使用 ResNet18 的前 4 层（直到 layer4），输出特征维度为 512
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_resnet = resnet18(weights=weights)

        # 去掉最后的全局平均池化 (AvgPool) 和全连接层 (FC)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-2])

        # 2. 投影层：将 CNN 输出维度映射到模型统一的 embed_dim
        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def extract_patches(self, x):
        """
        利用 unfold 高效切分 Patch
        输入 x: [B, 3, 224, 224]
        输出 patches: [B, N, 3, P, P], 其中 N 是 Patch 数量, P 是 patch_size
        """
        b, c, h, w = x.shape
        p = self.patch_size

        # 在高度和宽度维度上进行滑动窗口切分
        # [B, 3, 224, 224] -> [B, 3, 7, 7, 32, 32] (假设 224/32=7)
        patches = x.unfold(2, p, p).unfold(3, p, p)

        # 调整维度顺序: [B, 7, 7, 3, 32, 32]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        # 展平 Patch 数量维: [B, 49, 3, 32, 32]
        patches = patches.view(b, -1, c, p, p)
        return patches

    def shuffle_patches(self, patches):
        """
        在图像内部打乱 Patch 的顺序
        patches: [B, N, 3, P, P]
        """
        b, n, c, ph, pw = patches.shape
        # 为每一张图生成独立的随机索引
        # 如果追求性能，也可以在 Batch 级别统一使用一个索引，但独立索引安全性更高
        device = patches.device
        # per-sample random permutations: [B, N]
        idx = torch.stack([torch.randperm(n, device=device) for _ in range(b)], dim=0)
        # reshape 成适合 gather 的索引: [B, N, C, P, P]
        idx = idx.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, c, ph, pw)
        shuffled = patches.gather(1, idx)
        return shuffled

    def forward(self, x):
        """
        前向传播逻辑
        """
        b, _, _, _ = x.shape

        # 1. 切分 Patch: [B, 49, 3, 32, 32]
        patches = self.extract_patches(x)

        # 2. Shuffle (只在训练模式下强制 Shuffle，或者为了防止模型依赖位置，推理也 Shuffle)
        patches = self.shuffle_patches(patches)

        # 3. 准备喂入 CNN
        # 为了提高效率，将 Batch 和 Patch 维度合并: [B*49, 3, 32, 32]
        n = patches.size(1)
        patches_flattened = patches.view(-1, 3, self.patch_size, self.patch_size)

        # 4. 特征提取
        # ResNet18 的 layer4 输出通常是 [B*49, 512, 1, 1] (对于 32x32 的输入)
        feat = self.backbone(patches_flattened)

        # 如果输入不是 32 的整数倍导致输出不是 1x1，进行全局平均池化
        if feat.size(-1) > 1:
            feat = F.adaptive_avg_pool2d(feat, (1, 1))

        feat = feat.view(feat.size(0), -1)  # [B*49, 512]

        # 5. 投影与恢复维度
        feat_proj = self.proj(feat)  # [B*49, embed_dim]

        # 还原回序列格式: [B, 49, embed_dim]
        out = feat_proj.view(b, n, -1)

        return out


# --- 单元测试 ---
if __name__ == "__main__":
    dummy_img = torch.randn(2, 3, 224, 224)
    model = LocalPatchBranch(patch_size=32, embed_dim=256)
    output = model(dummy_img)

    print(f"输入图像形状: {dummy_img.shape}")
    print(f"输出特征序列形状: {output.shape}")  # 应为 [2, 49, 256]
