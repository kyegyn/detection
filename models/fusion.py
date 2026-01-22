import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    支路三 (A3)：Token-level Fusion + Evidence Pooling
    改进点 (方案A)：
    1. Token-to-Token Cross Attention (Loc <-> Freq)
    2. Attention Pooling 替代 Mean Pooling
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        # Cross-Attention: Loc Queries Freq
        self.attn_loc_freq = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        # A3-2: 增加 MLP
        self.mlp1 = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(4 * embed_dim, embed_dim))

        # Cross-Attention: Freq Queries Loc
        self.attn_freq_loc = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        # A3-2: 给 Freq 也加 MLP
        self.mlp2 = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(4 * embed_dim, embed_dim))

        # Evidence Pooling
        self.pool_loc = AttentionPooling(embed_dim)
        self.pool_freq = AttentionPooling(embed_dim)

        # Final Projection
        # Concatenate (Loc + Freq) -> D
        self.final_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, f_loc, f_freq):
        """
        Args:
            f_loc: 局部 Token 序列 [B, N, D]
            f_freq: 频带 Token 序列 [B, K, D]
        Returns:
            v_forensic: [B, D] 最终取证向量
            attn_weights: [B, N, K] 可视化用
            x_seq: [B, N, D] 增强后的局部序列 (用于 DiscrepancyFusion)
        """

        # 1. Loc Queries Freq
        loc_enhanced, attn_weights = self.attn_loc_freq(query=f_loc, key=f_freq, value=f_freq)
        x_loc = self.norm1(f_loc + loc_enhanced)
        x_loc = x_loc + self.mlp1(x_loc)

        # 2. Freq Queries Loc (A3-1: Key/Value 使用更新后的 x_loc)
        freq_enhanced, _ = self.attn_freq_loc(query=f_freq, key=x_loc, value=x_loc)
        x_freq = self.norm2(f_freq + freq_enhanced)
        x_freq = x_freq + self.mlp2(x_freq)  # A3-2: 对称增强

        # 3. Evidence Pooling
        v_loc_pooled = self.pool_loc(x_loc)  # [B, D]
        v_freq_pooled = self.pool_freq(x_freq)  # [B, D]

        # 4. 融合
        v_cat = torch.cat([v_loc_pooled, v_freq_pooled], dim=1)  # [B, 2D]
        v_forensic = self.final_proj(v_cat)  # [B, D]

        return v_forensic, attn_weights, x_loc


class AgentAttention(nn.Module):
    """
    Agent Attention 模块 (支持可控 Top-K 稀疏化)
    """
    def __init__(self, dim, num_heads=8, agent_num=8, dropout=0.1, topk_ratio=0.5, active_topk=True):
        """
        Args:
            topk_ratio: 保留比例
            active_topk: 是否启用 Top-K (用于区分正反向)
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.agent_num = agent_num
        self.topk_ratio = topk_ratio
        self.active_topk = active_topk  # ✅ 新增控制开关

        # 1. Agent 生成器
        self.agent_proj = nn.Linear(dim, dim)

        # 2. Agent 交互 (Agent -> Key)
        self.q_agent = nn.Linear(dim, dim)
        self.k_context = nn.Linear(dim, dim)
        self.v_context = nn.Linear(dim, dim)

        # 3. 广播 (Query -> Agent)
        self.q_orig = nn.Linear(dim, dim)
        self.k_agent = nn.Linear(dim, dim)
        self.v_agent = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, N, D = query.shape
        S = key.shape[1]
        H = self.num_heads

        # --- Step 1: 生成 Agent ---
        agent_tokens = F.adaptive_avg_pool1d(query.transpose(1, 2), self.agent_num).transpose(1, 2)
        agent_tokens = self.agent_proj(agent_tokens)

        # --- Step 2: Agent 读取 Context (Key) ---
        q_a = self.q_agent(agent_tokens).reshape(B, self.agent_num, H, D//H).permute(0, 2, 1, 3)
        k_c = self.k_context(key).reshape(B, S, H, D//H).permute(0, 2, 1, 3)
        v_c = self.v_context(value).reshape(B, S, H, D//H).permute(0, 2, 1, 3)

        # [B, H, Agent, S] -> 这里 S 代表 Key 的长度 (频带数 或 Patch数)
        attn_agent = (q_a @ k_c.transpose(-2, -1)) * self.scale

        # ✅ 【修改点 1】 仅在开关开启时执行 Top-K
        if self.active_topk:
            k_to_keep = max(1, int(S * self.topk_ratio))
            if k_to_keep < S:
                topk_vals, _ = attn_agent.topk(k_to_keep, dim=-1)
                mask = attn_agent < topk_vals[..., -1, None]
                attn_agent = attn_agent.masked_fill(mask, float('-inf'))

        attn_agent_weights = F.softmax(attn_agent, dim=-1) # 保存这个用于返回
        agent_out = (attn_agent_weights @ v_c)

        agent_out = agent_out.transpose(1, 2).reshape(B, self.agent_num, D)

        # --- Step 3: Query 读取 Agent ---
        q_o = self.q_orig(query).reshape(B, N, H, D//H).permute(0, 2, 1, 3)
        k_a = self.k_agent(agent_out).reshape(B, self.agent_num, H, D//H).permute(0, 2, 1, 3)
        v_a = self.v_agent(agent_out).reshape(B, self.agent_num, H, D//H).permute(0, 2, 1, 3)

        attn_final = (q_o @ k_a.transpose(-2, -1)) * self.scale
        attn_final = F.softmax(attn_final, dim=-1)

        out = (attn_final @ v_a).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.dropout(out)

        # ✅ 【修改点 2】 返回 attn_agent_weights (Agent看Key的权重)
        return out, attn_final, attn_agent_weights


class AgentAttentionFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        agent_num = 8
        topk_ratio = 0.5

        # 1. Loc -> Freq (正向): 开启 Top-K (S=8, 筛选频带)
        self.attn_loc_freq = AgentAttention(
            embed_dim, num_heads, agent_num, dropout, topk_ratio,
            active_topk=True  # ✅ 开启
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*embed_dim, embed_dim))

        # 2. Freq -> Loc (反向): 关闭 Top-K (S=49, 保持对所有 Patch 的全感知)
        self.attn_freq_loc = AgentAttention(
            embed_dim, num_heads, agent_num, dropout, topk_ratio,
            active_topk=False # ✅ 关闭 (Dense Attention)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*embed_dim, embed_dim))

        self.pool_loc = AttentionPooling(embed_dim)
        self.pool_freq = AttentionPooling(embed_dim)
        self.final_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, f_loc, f_freq):
        # 1. 正向: Loc 找 异常频带 (Top-K 生效)
        # band_attn: [B, H, Agent, Num_Bands] -> 这就是你要可视化的 "哪些频带被选中"
        loc_enhanced, _, band_attn = self.attn_loc_freq(query=f_loc, key=f_freq, value=f_freq)
        x_loc = self.norm1(f_loc + loc_enhanced)
        x_loc = x_loc + self.mlp1(x_loc)

        # 2. 反向: Freq 找 空间位置 (Dense)
        # 这里返回的第二个 attn 是 agent 看 patch 的权重，如果想看空间热力图可以用这个
        freq_enhanced, _, spatial_attn = self.attn_freq_loc(query=f_freq, key=x_loc, value=x_loc)
        x_freq = self.norm2(f_freq + freq_enhanced)
        x_freq = x_freq + self.mlp2(x_freq)

        v_loc_pooled = self.pool_loc(x_loc)
        v_freq_pooled = self.pool_freq(x_freq)

        v_cat = torch.cat([v_loc_pooled, v_freq_pooled], dim=1)
        v_forensic = self.final_proj(v_cat)

        # ✅ 【修改点 3】 返回 band_attn 用于后续分析
        # 为了兼容性，也可以把它存到 attn_weights 里返回
        return v_forensic, band_attn, x_loc


class AttentionPooling(nn.Module):
    """
    A3-3: Evidence-aware Attention Pooling
    学习每个 Token 的重要性权重，加权求和。
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Tanh()  # 限制范围，增加非线性
        )

    def forward(self, x):
        """
        x: [B, Seq_Len, D]
        return: [B, D]
        """
        # 计算 scores: [B, N, 1]
        scores = self.attn(x)
        # Softmax over sequence dimension
        weights = F.softmax(scores, dim=1)
        # Weighted sum
        out = torch.sum(x * weights, dim=1)
        return out


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
            nn.Linear(dim * 4, dim),
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
        # G-1: Gate 输入前统一尺度归一化
        # 这不会影响最后输出特征的模长(feature fusion)，只影响 Gate 的判断
        z_sem_n = F.normalize(z_sem, p=2, dim=1)
        v_forensic_n = F.normalize(v_forensic, p=2, dim=1)

        # 1. 计算交互特征
        diff_feat = torch.abs(z_sem_n - v_forensic_n)  # 差异特征 (Difference)
        prod_feat = z_sem_n * v_forensic_n  # 交互特征 (Product)

        # 2. 拼接所有信息 [B, D*4]
        combined = torch.cat([z_sem_n, v_forensic_n, diff_feat, prod_feat], dim=1)

        # 3. 生成门控权重
        alpha = self.gate_net(combined)  # [B, 1]
        if not self.training:
            # 这里的 item() 会把 tensor 转为 python float
            avg_alpha = alpha.mean().item()
            min_alpha = alpha.min().item()
            max_alpha = alpha.max().item()

            print(
                f"\n[Gating Debug] Alpha Stats -> Mean: {avg_alpha:.4f} | Min: {min_alpha:.4f} | Max: {max_alpha:.4f}")
            print(f"               (1.0 = Rely on CLIP Semantic, 0.0 = Rely on Forensic Artifacts)")
        # -----------------------
        # 2. 动态加权融合 (融合使用原始特征)
        f_fused = alpha * z_sem + (1 - alpha) * v_forensic

        return f_fused, alpha


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


class SubspaceDecorrelationLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z_a, z_b):
        """
        输入: z_a [Batch, Dim_A], z_b [Batch, Dim_B]
        """
        # 维度检查与自适应处理
        if z_a.dim() > 2:
            z_a = z_a.mean(dim=1)  # [B, K, D] -> [B, D]
        if z_b.dim() > 2:
            z_b = z_b.mean(dim=1)  # [B, K, D] -> [B, D]

        B = z_a.size(0)
        if B < 2:
            # 如果 Batch Size 小于 2，无法计算协方差，直接返回 0 Loss
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)
        # 1. 中心化 (Centering)
        z_a = z_a - z_a.mean(dim=0, keepdim=True)
        z_b = z_b - z_b.mean(dim=0, keepdim=True)

        # 2. 计算协方差 (Covariance)
        # 注意：这里除以 B-1 是无偏估计，但在深度学习 Loss 中除以 B 也没问题，只要统一即可
        cov = (z_a.T @ z_b) / (B - 1)

        # 3. 计算标准差 (Standard Deviation)
        # 加上 eps 防止方差为 0 导致 sqrt 后除以 0
        std_a = torch.sqrt(z_a.var(dim=0) + self.eps)
        std_b = torch.sqrt(z_b.var(dim=0) + self.eps)

        # 4. 计算相关系数矩阵 (Correlation Matrix)
        # 利用广播机制：[Dim_A, Dim_B] / ([Dim_A, 1] * [1, Dim_B])
        corr = cov / (std_a[:, None] * std_b[None, :])

        # 5. 最小化相关系数的平方均值
        # 即使相关系数是负的（负相关），平方后也是正的，我们希望它趋近于 0
        return torch.mean(corr ** 2)


class FineGrainedDecorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.decorr_loss = SubspaceDecorrelationLoss()

    def forward(self, f_sem, f_tex, f_freq):
        # 1. 语义 vs 纹理 去相关
        loss_tex = self.decorr_loss(f_sem, f_tex)

        # 2. 语义 vs 频域 去相关
        # 注意：f_freq 可能是 [B, K, D]，decorr_loss 内部会自动 mean(1) 变成 [B, D]
        loss_freq = self.decorr_loss(f_sem, f_freq)

        # 总损失
        return loss_tex + loss_freq


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
