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

        return v_forensic, attn_weights, x


# --- 【新增】情况1：差异感知融合 ---
class DiscrepancyFusion(nn.Module):
    """
    情况1：基于不一致性挖掘的差异感知融合
    逻辑：计算语义与纹理的空间一致性，利用差异图加权，突出伪造区域。
    """

    def __init__(self, dim=256):
        super().__init__()
        # 用于特征对齐的投影层 (可选，这里假设维度已对齐)
        self.scale = dim ** -0.5

    def forward(self, z_sem, f_forensic_seq):
        """
        Args:
            z_sem: 语义特征 [B, D]
            f_forensic_seq: 增强后的取证特征序列 [B, N, D] (来自 CrossAttention 的未池化输出)
        """
        B, N, D = f_forensic_seq.shape

        # 1. 语义广播: [B, D] -> [B, N, D]
        z_sem_expanded = z_sem.unsqueeze(1).expand(-1, N, -1)

        # 2. 计算一致性图 (Cosine Similarity)
        # normalize
        z_sem_norm = F.normalize(z_sem_expanded, dim=2)
        f_for_norm = F.normalize(f_forensic_seq, dim=2)

        # Similarity Map: [B, N, 1]
        consistency_map = (z_sem_norm * f_for_norm).sum(dim=2, keepdim=True)

        # 3. 生成差异图 (Discrepancy Map)
        # 值越大，表示语义和物理特征冲突越严重（可能是伪造痕迹）
        discrepancy_map = 1 - consistency_map

        # 4. 差异引导加权
        # 强化那些冲突严重的区域
        f_enhanced = f_forensic_seq * (1 + discrepancy_map)

        # 5. 融合语义
        f_fused = f_enhanced + z_sem_expanded

        # 6. 池化输出
        return f_fused.mean(dim=1)  # [B, D]


# --- 【新增】情况2：动态门控融合 ---
class GatingFusion(nn.Module):
    """
    情况2：语义引导的自适应门控
    逻辑：通过门控网络生成权重 alpha，动态平衡语义流与取证流。
    """

    def __init__(self, dim=256):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z_sem, v_forensic):
        """
        Args:
            z_sem: [B, D]
            v_forensic: [B, D]
        """
        # 1. 生成门控权重
        combined = torch.cat([z_sem, v_forensic], dim=1)
        alpha = self.gate_net(combined)  # [B, 1]

        # 2. 动态加权融合
        f_fused = alpha * z_sem + (1 - alpha) * v_forensic

        return f_fused


# --- 【新增】辅助工具：正交损失 ---
class OrthogonalLoss(nn.Module):
    """
    情况2配套：正交损失，强制语义特征与取证特征互补（不相关）
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_sem, v_forensic):
        # 归一化
        z_sem_n = F.normalize(z_sem, dim=1)
        v_for_n = F.normalize(v_forensic, dim=1)

        # 计算余弦相似度的平方，我们希望它趋近于0
        cosine = (z_sem_n * v_for_n).sum(dim=1)
        loss = torch.mean(cosine ** 2)
        return loss


class FinalClassifier(nn.Module):
    """
    最终分类头
    逻辑：拼接 (语义特征 + 取证特征) -> MLP -> Logits
    """

    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BN 有助于二分类收敛
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出 Logits，不带 Sigmoid (因为 Loss 里有 BCEWithLogits)
        )

    def forward(self, x):
        return self.net(x)
