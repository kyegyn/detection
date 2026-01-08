import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
# 引入 LoRA 库
from peft import get_peft_model, LoraConfig
import logging


class SemanticBranch(nn.Module):
    """
    支路一：基于 CLIP 的语义提取与映射模块
    功能：内部管理 CLIP 模型（加载、冻结、LoRA微调），并投影到对比学习空间。
    """

    def __init__(self, config):
        """
        config: 包含所有模型配置的字典，例如 'clip_model', 'projection_dim', 'use_lora' 等
        """
        super(SemanticBranch, self).__init__()
        self.logger = logging.getLogger("TSF-Net")

        model_name = config.get('clip_model', "openai/clip-vit-base-patch32")
        projection_dim = config.get('projection_dim', 256)

        self.logger.info(f"🔄 Loading CLIP Vision Model: {model_name} inside SemanticBranch...")

        # 1. 加载 CLIP (Vision Tower Only)
        self.clip_v = CLIPVisionModel.from_pretrained(model_name)
        self.clip_dim = self.clip_v.config.hidden_size

        # 2. 训练策略：LoRA vs 解冻微调
        if config.get('use_lora', False):
            # --- A. LoRA 模式 ---
            self.logger.info("🔧 Applying LoRA to CLIP...")
            lora_config = LoraConfig(
                r=config.get('lora_r', 8),
                lora_alpha=config.get('lora_alpha', 16),
                # Hugging Face CLIP 的 Attention 层命名通常是 q_proj, v_proj
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none"
            )
            # 使用 peft 包装模型，这将自动冻结非 LoRA 参数
            self.clip_v = get_peft_model(self.clip_v, lora_config)
            self.clip_v.print_trainable_parameters()

        else:
            # --- B. 传统微调模式 (解冻最后一层) ---
            self.logger.info("❄️ Freezing CLIP backbone initially...")
            self.freeze_backbone()

            # 默认解冻最后 1 层 Block + LayerNorm
            self.logger.info("🔓 Unfreezing last visual block for fine-tuning...")
            self.unfreeze_backbone(last_n_layers=1)

        # 3. 映射模块 (Projector)
        # 保持你原有的非线性瓶颈结构
        self.projector = nn.Sequential(
            nn.Linear(self.clip_dim, self.clip_dim),
            nn.LayerNorm(self.clip_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.clip_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

    def freeze_backbone(self):
        for param in self.clip_v.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_n_layers=None):
        """
        解冻策略：针对 Hugging Face CLIP Vision Model 的结构
        """
        # 1. 必须要解冻最后的 LayerNorm (post_layernorm)
        if hasattr(self.clip_v, "vision_model") and hasattr(self.clip_v.vision_model, "post_layernorm"):
            for param in self.clip_v.vision_model.post_layernorm.parameters():
                param.requires_grad = True

        # 2. 解冻 Transformer Layers
        if last_n_layers is None:
            # 全部解冻
            for param in self.clip_v.parameters():
                param.requires_grad = True
        else:
            # 只解冻最后 N 层
            # 路径: vision_model.encoder.layers
            if hasattr(self.clip_v, "vision_model") and hasattr(self.clip_v.vision_model, "encoder"):
                layers = self.clip_v.vision_model.encoder.layers
                total_layers = len(layers)
                for i in range(total_layers - last_n_layers, total_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: 图像 Tensor [B, 3, 224, 224]
        """
        # 1. CLIP 提取特征
        # 如果使用了 LoRA，这里必须传递 pixel_values 让梯度回流，不能用预提取的特征
        outputs = self.clip_v(pixel_values=pixel_values)
        raw_embed = outputs.pooler_output  # [B, clip_dim]

        # 2. 映射
        f_semantic = self.projector(raw_embed)

        # 3. 归一化 (用于 SupCon)
        z = F.normalize(f_semantic, p=2, dim=1)

        return z, f_semantic
