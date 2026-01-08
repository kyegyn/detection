import torch
import torch.nn as nn

# 导入所有子模块
from .branches.semantic import SemanticBranch
from .branches.local_patch import LocalPatchBranch
from .branches.global_freq import GlobalFreqBranch
from .fusion import CrossAttentionFusion, FinalClassifier


class TSFNet(nn.Module):
    """
    Tri-Stream Forensic Network (TSF-Net) 主模型
    """

    def __init__(self, config):
        """
        Args:
            config: 字典，包含 'clip_model', 'use_lora', 'lora_r' 等所有配置
        """
        super(TSFNet, self).__init__()

        # --- 1. 实例化三大支路 ---
        # 支路一：语义流 (CLIP) - 内部处理加载与 LoRA
        self.branch1 = SemanticBranch(config)

        # 支路二：局部纹理流 (Patch Shuffle)
        self.branch2 = LocalPatchBranch(
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim']
        )

        # 支路三：全局频率流 (FFT)
        self.branch3 = GlobalFreqBranch(
            embed_dim=config['embed_dim']
        )

        # --- 2. 实例化融合模块 ---
        self.fusion = CrossAttentionFusion(
            embed_dim=config['embed_dim'],
            num_heads=8
        )

        # --- 3. 实例化分类头 ---
        self.classifier = FinalClassifier(
            semantic_dim=config['projection_dim'],
            forensic_dim=config['embed_dim'],
            hidden_dim=256
        )

    def forward(self, img):
        """
        Args:
            img: 原始图像 Tensor [B, 3, 224, 224]

        注意：当使用 LoRA 时，CLIP 是可训练的，因此必须输入原始图像，
        不能再使用 train.py 里那种 `with torch.no_grad(): clip(img)` 预提取的方式。
        """

        # --- Step 1: 支路一 (语义) ---
        # z_sem_norm: 用于 SupCon Loss
        # f_sem_raw:  用于融合
        z_sem_norm, f_sem_raw = self.branch1(img)

        # --- Step 2: 支路二 (局部) ---
        f_loc = self.branch2(img)

        # --- Step 3: 支路三 (全局) ---
        z_freq = self.branch3(img)

        # --- Step 4: 交叉注意力融合 ---
        v_forensic, attn_weights = self.fusion(f_loc, z_freq)

        # --- Step 5: 最终分类 ---
        logits = self.classifier(f_sem_raw, v_forensic)

        return logits, z_sem_norm, attn_weights
