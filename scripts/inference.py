import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

# 1. 获取当前脚本的绝对路径 (例如: D:/.../detection/scripts/inference.py)
current_path = os.path.abspath(__file__)
# 2. 获取当前脚本所在的目录 (例如: D:/.../detection/scripts)
script_dir = os.path.dirname(current_path)
# 3. 获取项目的根目录，即 scripts 的上一级 (例如: D:/.../detection)
project_root = os.path.dirname(script_dir)
# 4. 将项目根目录添加到系统路径中，这样就能找到 models/data/utils 了
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目模块
from models.tsf_net import TSFNet
from data.dataset import ForensicDataset
from utils.fft_utils import seed_everything
from utils.metrics import BinaryMetrics


def load_config():
    """
    尝试优先使用项目相对路径加载配置文件，其次回退到训练脚本中的绝对路径。
    """
    # 优先：项目内的配置文件
    cfg_rel_path = os.path.join(project_root, 'config', 'model_config.yaml')
    if os.path.exists(cfg_rel_path):
        with open(cfg_rel_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    # 回退：训练脚本中的硬编码路径（如在服务器环境）
    fallback = '/root/autodl-tmp/detection/config/model_config.yaml'
    with open(fallback, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_val_loader(config, val_root: str | None = None):
    """构建验证集 DataLoader，增强与 train.py 对齐。支持传入自定义 val_root。"""
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


def restore_checkpoint(model, config, checkpoints_dir=None, epoch=None):
    """
    恢复模型权重。
    - 若提供 epoch，则从该 epoch 的权重恢复。
    - 否则将尝试在保存目录中选择最新的 checkpoint。
    返回：已加载的 checkpoint 字典或 None。
    """
    device = config['device']

    # 推断默认的 checkpoints 目录
    if checkpoints_dir is None:
        # 训练脚本保存为 save_path/exp_name/model_epoch_X.pth
        # 这里无法确定 exp_name；尝试从 save_path 下的所有子目录中挑选最新的目录
        save_root = config.get('save_path', './checkpoints')
        if os.path.isdir(save_root):
            # 选择按修改时间排序的最新实验目录
            subdirs = [os.path.join(save_root, d) for d in os.listdir(save_root) if os.path.isdir(os.path.join(save_root, d))]
            if subdirs:
                checkpoints_dir = sorted(subdirs, key=lambda p: os.path.getmtime(p), reverse=True)[0]
            else:
                checkpoints_dir = save_root
        else:
            checkpoints_dir = save_root

    ckpt_path = None
    if epoch is not None:
        # 指定 epoch 的路径（不含实验名时，优先在推断出的目录下找）
        candidate = os.path.join(checkpoints_dir, f'model_epoch_{epoch}.pth')
        if os.path.exists(candidate):
            ckpt_path = candidate
    else:
        # 自动选择最新的 model_epoch_*.pth
        if os.path.isdir(checkpoints_dir):
            pths = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
            if not pths:
                # 如果在实验目录下没有，尝试递归到所有子目录中找
                for root, _, files in os.walk(checkpoints_dir):
                    for f in files:
                        if f.startswith('model_epoch_') and f.endswith('.pth'):
                            pths.append(os.path.join(root, f))
            if pths:
                ckpt_path = sorted(pths, key=lambda p: os.path.getmtime(p), reverse=True)[0]

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[Warn] No checkpoint found in: {checkpoints_dir}. Running with random-initialized weights.")
        return None

    print(f"[Info] Restoring checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # 支持两种保存格式：字典包含 'model_state_dict' 或直接 state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def evaluate_on_val(model, val_loader, config):
    """在验证集上运行推理并输出 BinaryMetrics 报告。"""
    model.eval()
    evaluator = BinaryMetrics()

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(config['device'])
            labels = labels.to(config['device']).float()
            logits, z_sem, _, _, _ = model(imgs)
            evaluator.update(logits.squeeze(), labels)

    metrics = evaluator.print_report()
    return metrics


def evaluate_on_multiple_val_dirs(model, config, val_dirs: list[str]):
    """
    依次在多个验证数据集目录上评估模型，返回列表[(val_dir, metrics_dict)]。
    """
    results = []
    for vdir in val_dirs:
        if not os.path.isdir(vdir):
            print(f"[Skip] val dir not found: {vdir}")
            continue
        print(f"\n[Info] Evaluating on val dir: {vdir}")
        val_loader = build_val_loader(config, val_root=vdir)
        metrics = evaluate_on_val(model, val_loader, config)
        results.append((vdir, metrics))
    return results


def list_checkpoint_paths(checkpoints_dir: str):
    """在目录中查找所有 model_epoch_*.pth，按 epoch 数字排序返回完整路径列表。"""
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
    # 过滤非法 epoch，并按 epoch 升序
    ckpts = [(ep, path) for ep, path in ckpts if ep >= 0]
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def evaluate_checkpoints_over_val_dirs(model, config, checkpoints_dir: str, val_dirs: list[str], specific_epoch: int | None = None):
    """
    在 checkpoints_dir 下评估：
    - 若 specific_epoch 提供，则仅评估该 epoch 的权重；
    - 否则自动遍历该目录下所有 model_epoch_*.pth。
    对每个 checkpoint，依次在多个 val 目录上评估，返回列表[(epoch, [(val_dir, metrics_dict)])]。
    """
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
        print(f"\n[Info] Evaluating epoch {ep} -> {path}")
        checkpoint = torch.load(path, map_location=config['device'], weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        ep_results = evaluate_on_multiple_val_dirs(model, config, val_dirs)
        results.append((ep, ep_results))
    return results


def average_metrics_across_val_dirs(m_list: list[tuple[str, dict]]):
    """
    对 m_list = [(val_dir, metrics_dict)] 中的数值型指标逐键求平均。
    仅统计 int/float（排除 bool）；返回 {metric_name: avg_value}。
    """
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
    # 仅依赖两个输入：VAL_DIRS 和 INFER_DIR；可选 INFER_EPOCH 指定单个 epoch
    os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
    config = load_config()
    seed_everything(config['seed'])
    model = TSFNet(config).to(config['device'])

    val_dirs = config.get('VAL_DIRS',
                                  "/root/autodl-tmp/adm,/root/autodl-tmp/biggan,/root/autodl-tmp/glide,/root/autodl-tmp/midjourney,/root/autodl-tmp/sdv5,/root/autodl-tmp/vqdm,/root/autodl-tmp/wukong")
    checkpoint_dir = config['CHECKPOINT_DIR']
    checkpoint_epoch = config.get('CHECKPOINT_EPOCH', None)

    if not val_dirs or not checkpoint_dir:
        print("[Error] Please set VAL_DIRS (comma-separated) and INFER_DIR (checkpoint directory). Optionally set INFER_EPOCH for a single epoch.")
        print("Example PowerShell:")
        print('$env:VAL_DIRS="D:\\data\\val1,D:\\data\\val2,D:\\data\\val3,D:\\data\\val4,D:\\data\\val5,D:\\data\\val6,D:\\data\\val7"')
        print('$env:INFER_DIR="D:\\l\\实验\\detection\\scripts\\checkpoints\\exp_20260108_200756"')
        print('python D:\\l\\实验\\detection\\scripts\\inference.py')
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

    # 总结输出：按 epoch 分组，每个 val 目录一个结果
    print("\n[Summary] Evaluation results:")
    for ep, m_list in results:
        print(f"- epoch {ep}:")
        for vdir, metrics in m_list:
            print(f"  - val '{vdir}': {metrics}")
        # 额外：对每个 epoch 下所有验证集的相同指标取平均
        epoch_avg = average_metrics_across_val_dirs(m_list)
        if epoch_avg:
            print(f"  - epoch {ep} avg across {len(m_list)} vals: {epoch_avg}")
        else:
            print(f"  - epoch {ep} avg: N/A")
    return results


if __name__ == '__main__':
    main()
