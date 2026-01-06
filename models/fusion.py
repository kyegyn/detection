import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块
    核心逻辑：使用全局频率特征 (Global Context) 作为 Key/Value，
             去“扫描”和增强局部 Patch 特征 (Query)。
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        # Cross-Attention 层
        # batch_first=True 意味着输入形状为 [Batch, Seq_Len, Dim]
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # 标准 Transformer Block 的组件：Norm 和 Feed-Forward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, f_loc, z_freq):
        """
        Args:
            f_loc: 局部 Patch 特征序列 [B, N, D] (Query)
            z_freq: 全局频率特征向量 [B, D] (Key/Value)
        Returns:
            v_forensic: 聚合后的取证特征 [B, D]
            attn_weights: 注意力权重矩阵 [B, N, 1] (用于可视化)
        """
        # 1. 维度调整：将全局向量扩展为序列格式，作为 Context
        # z_freq: [B, D] -> [B, 1, D]
        kv = z_freq.unsqueeze(1)

        # 2. Cross-Attention 计算
        # Query=f_loc (局部), Key=kv (全局), Value=kv (全局)
        # attn_output: [B, N, D]
        # attn_weights: [B, N, 1] (表示每个 Patch 与全局频率的相关性)
        attn_output, attn_weights = self.attn(query=f_loc, key=kv, value=kv)

        # 3. 残差连接 + Norm (Add & Norm)
        x = self.norm1(f_loc + attn_output)

        # 4. 前馈网络 + 残差 + Norm
        x = self.norm2(x + self.mlp(x))

        # 5. 全局平均池化 (GAP)
        # 将增强后的 N 个 Patch 特征聚合成一个向量
        v_forensic = x.mean(dim=1)

        return v_forensic, attn_weights


class FinalClassifier(nn.Module):
    """
    最终分类头
    逻辑：拼接 (语义特征 + 取证特征) -> MLP -> Logits
    """

    def __init__(self, semantic_dim=256, forensic_dim=256, hidden_dim=256):
        super().__init__()
        input_dim = semantic_dim + forensic_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BN 有助于二分类收敛
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出 Logits，不带 Sigmoid (因为 Loss 里有 BCEWithLogits)
        )

    def forward(self, z_sem, v_forensic):
        # z_sem: [B, D_sem]
        # v_forensic: [B, D_for]
        combined = torch.cat([z_sem, v_forensic], dim=1)
        return self.net(combined)
