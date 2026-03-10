import torch
import torch.nn as nn
from .branches.semantic import SemanticBranch
from .branches.local_patch import LocalPatchBranch
from .branches.global_freq import GlobalFreqBranch
from .fusion import (FinalClassifier, DiscrepancyFusion,
                     GatingFusion, ChannelWiseIndependentGatingFusion,
                     SparseSpatialFrequencyAttention)


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
        self.ablation_mode = config.get('ablation_mode', 'full')
        self.fusion_type = config.get('fusion_type', 'concat')
        embed_dim = config['embed_dim']
        print(f"🔬 Ablation Mode: {self.ablation_mode}")
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

        # # --- 2. 实例化融合模块 ---
        print("🚀 Using Sparse Spatial Frequency Attention Fusion")
        self.fusion = SparseSpatialFrequencyAttention(
            dim=embed_dim,
            num_bands=8,   # 对应 K
            topk=3         # 稀疏路由保留前3个可疑频带
        )
        # --- 3. 高级融合策略选择 (Switch) ---
        # 默认为 'concat' (老方法), 可选 'discrepancy' (情况1), 'gating' (情况2)

        if self.fusion_type == 'discrepancy':
            print("🚀 Using Strategy 1: Discrepancy-Aware Fusion")
            self.adv_fusion = DiscrepancyFusion(dim=embed_dim)
            cls_input_dim = embed_dim # 融合后维度保持为 D

        elif self.fusion_type == 'gating':
            # print("🚀 Using Strategy 2: Dynamic Gating Fusion")
            # self.adv_fusion = GatingFusion(dim=embed_dim)
            print("🚀 Using Strategy: Channel-wise Independent Gating Fusion")
            self.adv_fusion = ChannelWiseIndependentGatingFusion(dim=embed_dim)
            cls_input_dim = embed_dim # 融合后维度保持为 D

        else:
            print("🚀 Using Default Strategy: Concatenation")
            self.adv_fusion = None
            cls_input_dim = embed_dim + config['projection_dim'] # 拼接后维度 D + D

        if self.ablation_mode in ['semantic_only', 'texture_only', 'freq_only']:
            # 单支路：输入维度 = embed_dim
            # 注意：SemanticBranch 的 projector 输出是 projection_dim (通常等于 embed_dim，配置里确认)
            cls_input_dim = config['projection_dim']

        elif self.ablation_mode == 'naive_concat':
            # 朴素拼接：语义 + 纹理(GlobalPool) + 频率(GlobalPool)
            # 维度 = D + D + D = 3*D
            cls_input_dim = config['projection_dim'] + embed_dim + embed_dim

        elif self.ablation_mode == 'with_ssfa':
            # 引入 SSFA 后，物理流被聚合成一个向量 v_forensic (D)
            # 融合方式为 Concat: Semantic (D) + Forensic (D) = 2*D
            cls_input_dim = config['projection_dim'] + embed_dim

        elif self.ablation_mode == 'full':
            # 完整模型：CIGF 融合后输出 D
            cls_input_dim = embed_dim

        else:
            # 默认 fallback
            cls_input_dim = embed_dim * 2
        print(f"📏 Classifier Input Dim: {cls_input_dim}")
        self.classifier = FinalClassifier(input_dim=cls_input_dim, hidden_dim=256)

    def forward(self, img):
        # 初始化变量
        z_sem_norm, f_sem_raw = None, None
        f_loc, f_tex_global = None, None
        z_freq = None
        v_forensic = None
        alpha, beta = None, None
        attn_weights = None
        logits = None

        # ==========================================
        # 1. 特征提取阶段 (根据消融模式执行)
        # ==========================================

        # 语义流
        if self.ablation_mode in ['semantic_only', 'naive_concat', 'with_ssfa', 'full']:
            z_sem_norm, f_sem_raw = self.branch1(img)

        # 纹理流
        if self.ablation_mode in ['texture_only', 'naive_concat', 'with_ssfa', 'full']:
            f_loc = self.branch2(img)
            f_tex_global = f_loc.mean(dim=1)  # GAP 作为全局纹理特征

        # 频率流
        if self.ablation_mode in ['freq_only', 'naive_concat', 'with_ssfa', 'full']:
            z_freq = self.branch3(img)
            # z_freq_global = z_freq.mean(dim=1) # 备用

        # ==========================================
        # 2. 物理互证阶段 (SSFA)
        # ==========================================
        if self.ablation_mode in ['with_ssfa', 'full']:
            # 使用 SSFA 提纯物理特征
            v_forensic, _, attn_weights = self.fusion(f_loc, z_freq)
        elif self.ablation_mode == 'naive_concat':
            # 不使用 SSFA，v_forensic 尚未生成，后续直接拼接原始特征
            pass

        # ==========================================
        # 3. 融合与决策阶段
        # ==========================================
        final_feat = None

        # --- A. 单支路模式 ---
        if self.ablation_mode == 'semantic_only':
            final_feat = f_sem_raw
        elif self.ablation_mode == 'texture_only':
            final_feat = f_tex_global
        elif self.ablation_mode == 'freq_only':
            # 频域序列做 GAP
            final_feat = z_freq.mean(dim=1)

        # --- B. 朴素三支路拼接 (无 SSFA, 无 CIGF) ---
        elif self.ablation_mode == 'naive_concat':
            # 拼接：语义 + 全局纹理 + 全局频率
            # [B, D] + [B, D] + [B, D] -> [B, 3D]
            z_freq_global = z_freq.mean(dim=1)
            final_feat = torch.cat([f_sem_raw, f_tex_global, z_freq_global], dim=1)

        # --- C. 引入 SSFA (拼接融合) ---
        elif self.ablation_mode == 'with_ssfa':
            # 拼接：语义 + SSFA提纯后的物理向量
            # [B, D] + [B, D] -> [B, 2D]
            final_feat = torch.cat([f_sem_raw, v_forensic], dim=1)

        # --- D. 完整模型 (CIGF 动态门控) ---
        elif self.ablation_mode == 'full':
            # 使用 CIGF 融合
            if self.fusion_type == 'gating':
                final_feat, alpha, beta = self.adv_fusion(f_sem_raw, v_forensic)
            elif self.fusion_type == 'discrepancy':
                # 兼容差异融合
                # 需要 x_seq (SSFA 的未池化输出)，这里简化处理，假设 ssfa 返回了它
                # 如需严格复现 discrepancy，需修改 ssfa 返回值
                final_feat = self.adv_fusion(f_sem_raw, v_forensic)
            else:
                final_feat = torch.cat([f_sem_raw, v_forensic], dim=1)

        # ==========================================
        # 4. 分类输出
        # ==========================================
        logits = self.classifier(final_feat)

        return logits, z_sem_norm, attn_weights, f_sem_raw, v_forensic, alpha, beta, f_tex_global, z_freq
