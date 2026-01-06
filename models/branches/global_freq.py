import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalFreqBranch(nn.Module):
    """
    支路三：全局频率指纹提取模块
    功能：利用 FFT 捕捉图像在生成过程中由于上采样导致的频谱异常。
    """

    def __init__(self, embed_dim=256):
        super(GlobalFreqBranch, self).__init__()

        # 定义一个轻量级 CNN，用于从频谱图中提取特征
        self.freq_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # 压缩为 [B, 128, 1, 1]
            nn.Flatten()
        )

        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        """
        输入 x: [B, 3, 224, 224]
        """
        # 1. 转为灰度图 (频谱分析通常在单通道进行)
        x_gray = torch.mean(x, dim=1, keepdim=True)  # [B, 1, 224, 224]

        # 2. 执行快速傅里叶变换 (FFT)
        # fft2 得到复数张量
        fft_complex = torch.fft.fft2(x_gray)

        # 3. 将低频成分移到频谱中心 (Shift)
        fft_shift = torch.fft.fftshift(fft_complex)

        # 4. 计算幅度谱 (Magnitude Spectrum)
        # abs 得到复数的模
        mag = torch.abs(fft_shift)

        # 5. 对数变换 (Log Transform)
        # 关键步骤：频谱值跨度极大，对数化能让高频弱信号（AI 伪影）更明显
        mag_log = torch.log(1 + mag)

        # 6. CNN 提取特征
        feat = self.freq_net(mag_log)  # [B, 128]
        z_freq = self.proj(feat)  # [B, embed_dim]

        return z_freq


# --- 单元测试 ---
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    model = GlobalFreqBranch(embed_dim=256)
    out = model(dummy_input)
    print(f"全局频率特征形状: {out.shape}")  # [2, 256]
