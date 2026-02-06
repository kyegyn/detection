import torch
import torch.nn as nn
import math

class MiniBandCNN(nn.Module):
    """
    Step 4: 小型可学习特征提取器 (Lightweight Band-CNN)
    作用：从每个频带的双路频谱图中提取纹理特征。
    输入: [B*K, 2, H, W] (双路: Energy + Shape)
    输出: [B*K, D] (频带特征向量)
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # 设计一个非常轻量级的 CNN
        # 输入通道=2 (Energy View + Shape View)
        self.net = nn.Sequential(
            # Layer 1: 降采样，提取初步特征 [H, W] -> [H/2, W/2]
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 2: 继续降采样 [H/2, W/2] -> [H/4, W/4]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 3: 进一步提取 [H/4, W/4] -> [H/8, W/8]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Global Average Pooling: 压缩空间维度 -> [B*K, 64, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            # Projection: 映射到目标维度 D
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class GlobalFreqBranch(nn.Module):
    """
    改进版支路三 (A2)：双路频谱 + 频带掩码 + Band-CNN
    逻辑：
    1. 生成 Energy View (绝对强度) 和 Shape View (相对形态)。
    2. 利用径向 Mask 将频谱切分为 K 个频带。
    3. 每个频带截取双路视图，送入共享权重的 Band-CNN 提取特征。
    4. 加入频带位置编码，输出 Token 序列。
    """
    def __init__(self, embed_dim=256, num_bands=8, img_size=224):
        super(GlobalFreqBranch, self).__init__()
        self.num_bands = num_bands
        self.img_size = img_size

        # 1. 预先生成径向 Mask [1, K, H, W]
        # 使用 register_buffer 确保它会被保存到 state_dict 但不会被优化器更新
        self.register_buffer('band_masks', self._create_radial_masks(img_size, num_bands))

        # 2. 实例化 Mini-CNN (所有频带共享权重)
        self.band_cnn = MiniBandCNN(embed_dim=embed_dim)

        # 3. 频带位置编码 (Learnable Band Positional Embedding)
        # 因为 CNN 是共享权重的，需要显式告诉模型哪个特征来自低频，哪个来自高频
        self.band_pos_embed = nn.Parameter(torch.zeros(1, num_bands, embed_dim))

        # 初始化位置编码
        nn.init.trunc_normal_(self.band_pos_embed, std=0.02)

    def _create_radial_masks(self, size, k):
        """
        生成 K 个同心圆环掩码
        """
        center = size // 2
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        r = torch.sqrt((x - center) ** 2 + (y - center) ** 2)

        # 覆盖到角点，最大半径为对角线的一半
        max_r = size / math.sqrt(2)
        masks = []

        # 线性分桶 (Linear Binning)
        step = max_r / k
        for i in range(k):
            r_min = step * i
            r_max = step * (i + 1)
            # 生成环形 Mask (布尔值)
            mask = (r >= r_min) & (r < r_max)
            masks.append(mask.float())

        # 堆叠并增加 Batch 维度 -> [1, K, H, W]
        return torch.stack(masks).unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            z_freq_seq: 频带 Token 序列 [B, K, D]
        """
        B, C, H, W = x.shape

        # --- Step 1: 生成双路频谱视图 ---

        # 1.1 预处理：转灰度 [B, 1, H, W]
        x_gray = torch.mean(x, dim=1, keepdim=True)

        # 1.2 FFT & Shift
        fft = torch.fft.fft2(x_gray)
        fft_shift = torch.fft.fftshift(fft)

        # 1.3 能量视图 (Energy View): S_eng
        # log(1 + |F|) 保留绝对强度，捕捉强 Fingerprint
        mag_raw = torch.abs(fft_shift)
        s_eng = torch.log(1 + mag_raw) # [B, 1, H, W]

        # 1.4 形态视图 (Shape View): S_shape
        # Instance Normalization per image，消除亮度差异，专注分布形态
        mean = s_eng.mean(dim=[2, 3], keepdim=True)
        std = s_eng.std(dim=[2, 3], keepdim=True) + 1e-6
        s_shape = (s_eng - mean) / std # [B, 1, H, W]

        # --- Step 2 & 3: 频带分桶与双路拼接 ---

        # 扩展维度以匹配 K [B, K, H, W]
        s_eng_expand = s_eng.expand(-1, self.num_bands, -1, -1)
        s_shape_expand = s_shape.expand(-1, self.num_bands, -1, -1)

        # 获取 Masks [1, K, H, W] (自动广播到 B)
        masks = self.band_masks

        # 应用掩码 (Masking)
        # 将非本频带区域置为 0
        b_eng = s_eng_expand * masks
        b_shape = s_shape_expand * masks

        # 拼接双路 [B, K, 2, H, W]
        # dim=2 是通道维度，每个频带现在是一张 2 通道的 "特征图"
        band_input = torch.stack([b_eng, b_shape], dim=2)

        # --- Step 4: Band-CNN 特征提取 ---

        # Reshape 为 CNN 输入格式: [B*K, 2, H, W]
        cnn_input = band_input.view(B * self.num_bands, 2, H, W)

        # 提取特征: [B*K, D]
        band_features = self.band_cnn(cnn_input)

        # 还原维度: [B, K, D]
        band_features = band_features.view(B, self.num_bands, -1)

        # --- Step 5: 加入位置编码 ---

        # 加上可学习的 Positional Embedding
        # 这一步至关重要，因为 CNN 是共享的，它自己不知道处理的是低频还是高频
        z_freq_seq = band_features + self.band_pos_embed

        # 输出序列 [B, K, D]，供后续 Fusion 模块 (如 Agent Attention) 使用
        return z_freq_seq
