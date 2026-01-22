import os
import sys
import yaml
import torch
import numpy as np  # 【新增】用于统计计算
from torch.utils.data import DataLoader

# 1. 获取当前脚本的绝对路径
current_path = os.path.abspath(__file__)
# 2. 获取当前脚本所在的目录
script_dir = os.path.dirname(current_path)
# 3. 获取项目的根目录
project_root = os.path.dirname(script_dir)
# 4. 将项目根目录添加到系统路径中
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目模块
from models.tsf_net import TSFNet
from data.dataset import ForensicDataset
from utils.fft_utils import seed_everything
from utils.metrics import BinaryMetrics


def load_config():
    """加载配置"""
    cfg_rel_path = os.path.join(project_root, 'config', 'model_config.yaml')
    if os.path.exists(cfg_rel_path):
        with open(cfg_rel_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    fallback = '/root/autodl-tmp/detection/config/model_config.yaml'
    with open(fallback, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_val_loader(config, val_root: str | None = None):
    """构建验证集 DataLoader"""
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))
    ])
    default_root = '/root/autodl-tmp/data/val'
    root = val_root or default_root
    val_ds = ForensicDataset(root_dir=root, transform=val_transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    return val_loader


def evaluate_on_val(model, val_loader, config):
    """
    在验证集上运行推理并输出 BinaryMetrics 报告，同时进行 Gate 逻辑自测。
    """
    model.eval()
    evaluator = BinaryMetrics()

    # --- 自测数据容器 ---
    all_l_sem = []
    all_l_for = []
    all_l_fused = []
    all_alphas = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(config['device'])
            labels = labels.to(config['device']).float()

            # 解包：捕获中间特征和 Alpha
            # TSFNet return: logits, z_sem, attn, f_sem_raw, v_forensic, alpha, f_tex, z_freq
            logits, _, _, f_sem_raw, v_forensic, alpha, _, _ = model(imgs)

            evaluator.update(logits.squeeze(), labels)

            # --- 收集自测数据 (仅在 Gating 模式下有效) ---
            if alpha is not None:
                # 确保 Classifier 输入维度匹配 (Gating 模式下 embed_dim 对齐)
                # 如果是 Concat 模式，Classifier 输入维度是 2*dim，这里会报错，需跳过
                if hasattr(model.classifier, 'net') and \
                        model.classifier.net[0].in_features == f_sem_raw.shape[1]:

                    # 1. 模拟单流决策：如果只用语义/只用取证，Classifier 会输出什么？
                    l_sem = model.classifier(f_sem_raw).squeeze()
                    l_for = model.classifier(v_forensic).squeeze()

                    all_l_sem.append(l_sem.cpu())
                    all_l_for.append(l_for.cpu())
                    all_l_fused.append(logits.squeeze().cpu())
                    all_alphas.append(alpha.squeeze().cpu())

    metrics = evaluator.print_report()

    # --- 计算并打印自测统计量 (Self-Test Analysis) ---
    if len(all_alphas) > 0:
        # 拼接所有 Batch
        l_sem = torch.cat(all_l_sem).numpy()
        l_for = torch.cat(all_l_for).numpy()
        l_fused = torch.cat(all_l_fused).numpy()
        alphas = torch.cat(all_alphas).numpy()

        # 处理 batch_size=1 或 scalars
        if l_sem.ndim == 0: l_sem = np.array([l_sem])
        if l_for.ndim == 0: l_for = np.array([l_for])
        if alphas.ndim == 0: alphas = np.array([alphas])

        print(f"\n🔍 [Gate Mechanism Self-Test]")

        # 1. Logits 分布 (均值/方差/分位数)
        def print_dist(name, data):
            p10, p50, p90 = np.percentile(data, [10, 50, 90])
            print(f"    - {name:<10}: Mean={data.mean():.3f} | Std={data.std():.3f} | P10={p10:.3f} P50={p50:.3f} P90={p90:.3f}")

        print(f"  > Logits Distribution (What each expert thinks):")
        print_dist("Semantic", l_sem)
        print_dist("Forensic", l_for)
        print_dist("Fused", l_fused)

        # 2. Alpha 分位数 (检查是否塌缩)
        p10, p50, p90 = np.percentile(alphas, [10, 50, 90])
        print(f"  > Alpha Distribution (Gate Activity):")
        print(f"    - P10: {p10:.4f} | P50: {p50:.4f} | P90: {p90:.4f}")
        print(f"    - Dynamic Range: {p90 - p10:.4f} (Should be > 0.1)")

        # 3. 核心相关性分析 (检查 Gate 是否理性)
        # 差异度：两个专家意见分歧有多大
        disagreement = np.abs(l_sem - l_for)
        # 计算 Alpha 与 分歧度的相关性
        if np.std(alphas) > 1e-6 and np.std(disagreement) > 1e-6:
            corr = np.corrcoef(alphas, disagreement)[0, 1]
        else:
            corr = 0.0

        print(f"  > Gate Logic Check:")
        print(f"    - Corr(Alpha, |Logit_Sem - Logit_For|): {corr:.4f}")
        print(f"      👉 Interpretation:")
        print(f"         * High Positive (+): Conflict -> Trust Semantic (Clip is authority)")
        print(f"         * High Negative (-): Conflict -> Trust Forensic (Physics is veto)")
        print(f"         * Near Zero     (0): Gate is blind to conflict (Random/Constant)")
        print("-" * 40)

    return metrics


def evaluate_on_multiple_val_dirs(model, config, val_dirs: list[str]):
    results = []
    for vdir in val_dirs:
        if not os.path.isdir(vdir):
            print(f"[Skip] val dir not found: {vdir}")
            continue
        # 打印验证集名称，方便日志定位
        dirname = os.path.basename(vdir.rstrip('/\\'))
        print(f"\n{'='*10} Evaluating on: {dirname} {'='*10}")
        val_loader = build_val_loader(config, val_root=vdir)
        metrics = evaluate_on_val(model, val_loader, config)
        results.append((vdir, metrics))
    return results


def list_checkpoint_paths(checkpoints_dir: str):
    if not checkpoints_dir or not os.path.isdir(checkpoints_dir):
        return []
    ckpts = []
    for fname in os.listdir(checkpoints_dir):
        if fname.startswith('model_epoch_') and fname.endswith('.pth'):
            try:
                ep_str = fname[len('model_epoch_'):-len('.pth')]
                ep = int(ep_str)
            except Exception:
                ep = -1
            ckpts.append((ep, os.path.join(checkpoints_dir, fname)))
    ckpts = [(ep, path) for ep, path in ckpts if ep >= 0]
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def evaluate_checkpoints_over_val_dirs(model, config, checkpoints_dir: str, val_dirs: list[str], specific_epoch: int | None = None):
    results = []
    if specific_epoch is not None:
        ckpt_path = os.path.join(checkpoints_dir, f'model_epoch_{specific_epoch}.pth')
        if not os.path.exists(ckpt_path):
            print(f"[Warn] Specific epoch checkpoint not found: {ckpt_path}")
            return results
        checkpoints = [(specific_epoch, ckpt_path)]
    else:
        checkpoints = list_checkpoint_paths(checkpoints_dir)
        if not checkpoints:
            print(f"[Warn] No checkpoints found under: {checkpoints_dir}")
            return results

    for ep, path in checkpoints:
        print(f"\n\n################# Epoch {ep} #################")
        print(f"[Info] Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=config['device'], weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        ep_results = evaluate_on_multiple_val_dirs(model, config, val_dirs)
        results.append((ep, ep_results))
    return results


def average_metrics_across_val_dirs(m_list: list[tuple[str, dict]]):
    if not m_list:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _, metrics in m_list:
        if not isinstance(metrics, dict):
            continue
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts.get(k, 0) > 0}


def main():
    os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
    config = load_config()
    seed_everything(config['seed'])
    model = TSFNet(config).to(config['device'])

    val_dirs = config.get('VAL_DIRS', "/root/autodl-tmp/data/val")
    checkpoint_dir = config['CHECKPOINT_DIR']
    checkpoint_epoch = config.get('CHECKPOINT_EPOCH', None)

    if not val_dirs or not checkpoint_dir:
        print("[Error] Please check VAL_DIRS and CHECKPOINT_DIR in config.")
        return None

    val_dirs = [p.strip() for p in val_dirs.split(',') if p.strip()]
    specific_epoch = int(checkpoint_epoch) if checkpoint_epoch else None

    results = evaluate_checkpoints_over_val_dirs(
        model,
        config,
        checkpoints_dir=checkpoint_dir,
        val_dirs=val_dirs,
        specific_epoch=specific_epoch,
    )

    print("\n[Summary] Evaluation results:")
    for ep, m_list in results:
        print(f"- epoch {ep}:")
        for vdir, metrics in m_list:
            vname = os.path.basename(vdir.rstrip('/\\'))
            print(f"  - {vname:<10}: Acc={metrics.get('Acc',0):.4f}, AUC={metrics.get('AUC',0):.4f}")
        epoch_avg = average_metrics_across_val_dirs(m_list)
        if epoch_avg:
            print(f"  - Avg       : Acc={epoch_avg.get('Acc',0):.4f}, AUC={epoch_avg.get('AUC',0):.4f}")
    return results


if __name__ == '__main__':
    main()
