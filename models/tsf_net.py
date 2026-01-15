import torch
import torch.nn as nn

# 导入所有子模块
from .branches.semantic import SemanticBranch
from .branches.local_patch import LocalPatchBranch
from .branches.global_freq import GlobalFreqBranch
from .fusion import CrossAttentionFusion, FinalClassifier, DiscrepancyFusion, GatingFusion


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
        self.config = config
        # --- 1. 实例化三大支路 ---
        # 支路一：语义流 (CLIP) - 内部处理加载与 LoRA
        self.branch1 = SemanticBranch(config)
        embed_dim = config['embed_dim']
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
        # --- 3. 高级融合策略选择 (Switch) ---
        # 默认为 'concat' (老方法), 可选 'discrepancy' (情况1), 'gating' (情况2)
        self.fusion_type = config.get('fusion_type', 'concat')

        if self.fusion_type == 'discrepancy':
            print("🚀 Using Strategy 1: Discrepancy-Aware Fusion")
            self.adv_fusion = DiscrepancyFusion(dim=embed_dim)
            cls_input_dim = embed_dim # 融合后维度保持为 D

        elif self.fusion_type == 'gating':
            print("🚀 Using Strategy 2: Dynamic Gating Fusion")
            self.adv_fusion = GatingFusion(dim=embed_dim)
            cls_input_dim = embed_dim # 融合后维度保持为 D

        else:
            print("🚀 Using Default Strategy: Concatenation")
            self.adv_fusion = None
            cls_input_dim = embed_dim + config['projection_dim'] # 拼接后维度 D + D


        # --- 3. 实例化分类头 ---
        self.classifier = FinalClassifier(input_dim=cls_input_dim, hidden_dim=256)

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
        f_tex_global = f_loc.mean(dim=1)
        # --- Step 3: 支路三 (全局) ---
        z_freq = self.branch3(img)

        # --- Step 4: 交叉注意力融合 ---
        v_forensic, attn_weights, x_seq = self.fusion(f_loc, z_freq)

        # =======================================================
        # 【核心修改 Step 3.5】: Modality Dropout (语义丢弃)
        # =======================================================
        # 逻辑：在训练时，以 40% 的概率将语义特征强行置零。
        # 目的：欺骗门控网络和分类器，让它们以为语义流失效了，
        #       从而被迫去挖掘 v_forensic (物理流) 中的有用信息。
        # f_sem_for_fusion = f_sem_raw

        # if self.training:
        #     # 概率建议设为 0.3 - 0.5。这里设为 0.4 (40% 概率丢弃语义)
        #     if torch.rand(1).item() < 0.5:
        #         f_sem_for_fusion = torch.zeros_like(f_sem_raw)
        # =======================================================
        alpha = None
        # 3. 最终融合决策 (Strategy Switch)
        if self.fusion_type == 'discrepancy':
            # 情况1：传入 语义向量 + 取证序列特征
            final_feat = self.adv_fusion(f_sem_raw, x_seq)

        elif self.fusion_type == 'gating':
            # 情况2：传入 语义向量 + 取证聚合向量
            final_feat, alpha = self.adv_fusion(f_sem_raw, v_forensic)

        else:
            # 默认：简单拼接
            final_feat = torch.cat([f_sem_raw, v_forensic], dim=1)
        # --- Step 5: 最终分类 ---
        logits = self.classifier(final_feat)

        return logits, z_sem_norm, attn_weights, f_sem_raw, v_forensic, alpha, f_tex_global, z_freq
