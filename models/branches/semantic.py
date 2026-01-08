import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


class SemanticBranch(nn.Module):
    """
    支路一：基于 CLIP 的语义提取与映射模块
    功能：提取图像全局语义，并投影到对比学习空间。
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", projection_dim=256, freeze_clip=True):
        super(SemanticBranch, self).__init__()

        # 1. 加载预训练的 CLIP 视觉模型
        # 我们只使用 VisionTower，不需要 TextEncoder
        self.clip_v = CLIPVisionModel.from_pretrained(model_name)
        self.clip_dim = self.clip_v.config.hidden_size  # 通常为 768 (Base) 或 1024 (Large)

        # 2. 映射模块 (Mapping Module / Projector)
        # 采用非线性瓶颈结构，增强特征表达能力
        self.projector = nn.Sequential(
            nn.Linear(self.clip_dim, self.clip_dim),
            nn.LayerNorm(self.clip_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.clip_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # 3. 冻结策略
        if freeze_clip:
            self.freeze_backbone()

    def freeze_backbone(self):
        """冻结 CLIP 参数，训练初期建议开启，防止权重被破坏"""
        for param in self.clip_v.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_n_layers=None):
        """
        如果 last_n_layers 为 None，则全部解冻；
        否则尝试只解冻 vision encoder 的最后 n 层（若结构可访问）。
        """
        if last_n_layers is None:
            for p in self.clip_v.parameters():
                p.requires_grad = True
            return

        # 先全部冻结
        for p in self.clip_v.parameters():
            p.requires_grad = False

        # 尝试找到 encoder layers 并只解冻最后 n 层
        vis = getattr(self.clip_v, "vision_model", None)
        if vis is not None and hasattr(vis, "encoder"):
            enc = vis.encoder
            layers = getattr(enc, "layers", getattr(enc, "layer", None))
            if layers is not None:
                for layer in layers[-last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                return

        # 回退：若无法定位层结构，则全部解冻
        for p in self.clip_v.parameters():
            p.requires_grad = True

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: 经过 CLIPProcessor 处理后的图像 Tensor [B, 3, 224, 224]
        Returns:
            f_semantic: 经过 L2 归一化的特征向量，用于 SupCon Loss [B, projection_dim]
            z: CLIP 原始的池化特征，可用于最终分类融合 [B, clip_dim]
        """
        # 1. CLIP 提取原始特征
        outputs = self.clip_v(pixel_values=pixel_values)
        raw_embed = outputs.pooler_output

        # 2. 映射到判别空间
        f_semantic = self.projector(raw_embed)  # 获取 L2Norm 之前的特征

        # 3. 经过最后一层并归一化，用于计算 SupCon Loss
        z = F.normalize(f_semantic, p=2, dim=1)

        # 返回 h 用于后期融合，返回 z 用于计算当前支路的损失
        return z, f_semantic

    def forward_from_embed(self, raw_embed):
        """
        专门处理 train.py 中预提取好的 CLIP pooler_output [B, 768]
        """
        # 1. 映射
        f_semantic = self.projector(raw_embed)

        # 2. 归一化
        z = F.normalize(f_semantic, p=2, dim=1)

        return z, f_semantic


# --- 单元测试 ---
if __name__ == "__main__":
    # 模拟输入：BatchSize=4, RGB, 224x224
    dummy_input = torch.randn(4, 3, 224, 224)

    # 实例化模块
    branch1 = SemanticBranch(projection_dim=256, freeze_clip=True)

    # 前向传播
    z, h = branch1(dummy_input)

    print(f"CLIP 原始特征维度: {h.shape}")  # torch.Size([4, 768])
    print(f"映射后特征维度 (z): {z.shape}")  # torch.Size([4, 256])
    print(f"z 是否已归一化 (模长): {torch.norm(z, p=2, dim=1)}")  # 应接近 1.0
