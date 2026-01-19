import torch
import torch.nn as nn
import math


class GlobalFreqBranch(nn.Module):
    """
    支路三 (A2)：多频带频率 Token 分支
    改进点 (方案A)：
    1. 引入 Instance Normalization 消除亮度/对比度影响。
    2. 将频谱按径向切分为 K 个频带 Token，显式建模不同频段的异常。
    """

    def __init__(self, embed_dim=256, num_bands=8, img_size=224):
        super(GlobalFreqBranch, self).__init__()
        self.num_bands = num_bands
        self.register_buffer('band_masks', self._create_radial_masks(img_size, num_bands))

        # A2-1: 输入维度是 3 (Mean, Std, Max)
        input_stats_dim = 3

        self.band_proj = nn.Sequential(
            nn.Linear(input_stats_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def _create_radial_masks(self, size, k):
        center = size // 2
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        r = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        max_r = size / math.sqrt(2)
        masks = []
        step = max_r / k
        for i in range(k):
            r_min = step * i
            r_max = step * (i + 1)
            mask = (r >= r_min) & (r < r_max)
            masks.append(mask.float())
        return torch.stack(masks)

    def forward(self, x):
        B = x.shape[0]
        # 1. 预处理
        x_gray = torch.mean(x, dim=1, keepdim=True)
        fft = torch.fft.fft2(x_gray)
        fft_shift = torch.fft.fftshift(fft)
        mag_log = torch.log(1 + torch.abs(fft_shift))

        # 2. 归一化 (Instance Normalization)
        mean = mag_log.mean(dim=[2, 3], keepdim=True)
        std = mag_log.std(dim=[2, 3], keepdim=True) + 1e-6
        mag_norm = (mag_log - mean) / std
        mag_norm = mag_norm.squeeze(1)  # [B, H, W]

        # 3. 准备广播
        mag_expand = mag_norm.unsqueeze(1)  # [B, 1, H, W]
        masks_expand = self.band_masks.unsqueeze(0)  # [1, K, H, W]

        # 4. 屏蔽无效区域 (Masking)
        masked_val = mag_expand * masks_expand  # [B, K, H, W]

        # --- 统计量计算 ---

        # A. Mean
        mask_sums = masks_expand.sum(dim=[2, 3]) + 1e-6
        means = masked_val.sum(dim=[2, 3]) / mask_sums  # [B, K]

        # B. Max (关键修复)
        # 使用 masked_fill 替代布尔索引赋值，解决形状不匹配问题
        # 将掩码为 0 (False) 的区域填充为 -1e9，以免影响 Max 计算
        masked_val_for_max = masked_val.masked_fill(masks_expand == 0, -1e9)
        maxs = masked_val_for_max.amax(dim=[2, 3])  # [B, K]

        # C. Std
        means_expand = means.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        var_part = ((mag_expand - means_expand) ** 2) * masks_expand
        stds = torch.sqrt(var_part.sum(dim=[2, 3]) / mask_sums + 1e-6)  # [B, K]

        # 5. 拼接与投影
        band_feats = torch.stack([means, stds, maxs], dim=-1)  # [B, K, 3]
        z_freq_seq = self.band_proj(band_feats)  # [B, K, D]

        return z_freq_seq