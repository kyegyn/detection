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
        super().__init__()

        # --- 1. 实例化三大支路 ---
        # 支路一：语义流 (CLIP)
        self.branch1 = SemanticBranch(
            model_name=config.get('clip_model', "openai/clip-vit-base-patch32"),
            projection_dim=config['projection_dim'],
            freeze_clip=True  # 默认冻结 CLIP
        )

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
            semantic_dim=config['projection_dim'],  # 对应支路一的输出维度
            forensic_dim=config['embed_dim'],  # 对应融合后的维度
            hidden_dim=256
        )

    def forward(self, img, clip_embed=None):
        """
        Args:
            img: 原始图像 Tensor [B, 3, 224, 224]
            clip_embed: (可选) 如果在外部预提取了 CLIP 特征，可直接传入以节省显存。
                        格式应为 CLIP VisionModel 的输出 (pixel_values input)。
                        *注意：根据 train.py 的设计，我们通常传入的是 clip_features*
        """

        # --- Step 1: 支路一 (语义) ---
        # SemanticBranch 返回:
        # z_sem_norm: L2归一化后的特征 (用于 SupCon Loss)
        # f_sem_raw:  未归一化的投影特征 (用于最终融合)
        # 注意：这里需要 SemanticBranch 支持接收 raw features 或者直接接收 img
        # 为了兼容 train.py 中 "预提取" 的逻辑，我们假设 branch1 内部逻辑已适配
        # 或者我们在这里做一个判断：
        if clip_embed is not None:
            # 如果是预提取的特征(train.py逻辑)，SemanticBranch 需要稍微改动 forward 接口
            # 但通常更简单的做法是：SemanticBranch 接收 pixel_values
            # 如果为了省显存，train.py 里预提取的是 pooler_output
            # 那么 branch1.forward 应该处理 embedding。
            # **为了简化，这里假设 branch1 接收的是 clip 预提取后的 raw_embed**
            z_sem_norm, f_sem_raw = self.branch1.forward_from_embed(clip_embed)
        else:
            # 推理模式：直接传图
            z_sem_norm, f_sem_raw = self.branch1(img)

        # --- Step 2: 支路二 (局部) ---
        # 输出: [B, N, D]
        f_loc = self.branch2(img)

        # --- Step 3: 支路三 (全局) ---
        # 输出: [B, D]
        z_freq = self.branch3(img)

        # --- Step 4: 交叉注意力融合 ---
        # 使用全局频率引导局部 Patch
        v_forensic, attn_weights = self.fusion(f_loc, z_freq)

        # --- Step 5: 最终分类 ---
        # 拼接 语义特征(f_sem_raw) 和 取证特征(v_forensic)
        logits = self.classifier(f_sem_raw, v_forensic)

        return logits, z_sem_norm, attn_weights
