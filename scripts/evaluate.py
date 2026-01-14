import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

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
    """
    优先加载项目相对路径的配置文件，若不存在则尝试绝对路径。
    """
    # 优先：项目 config 目录下的 model_config.yaml
    cfg_rel_path = os.path.join(project_root, 'config', 'model_config.yaml')
    if os.path.exists(cfg_rel_path):
        print(f"[Info] Loading config from: {cfg_rel_path}")
        with open(cfg_rel_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # 回退：硬编码的绝对路径 (适配你的 autodl 环境)
    fallback = '/root/autodl-tmp/detection/config/model_config.yaml'
    if os.path.exists(fallback):
        print(f"[Info] Loading config from fallback path: {fallback}")
        with open(fallback, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(f"Config file not found in {cfg_rel_path} or {fallback}")


def build_val_loader(config, val_root: str):
    """构建验证集 DataLoader"""
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))
    ])

    # 确保路径存在
    if not os.path.exists(val_root):
        print(f"[Warn] Validation directory not found: {val_root}")
        return None

    val_ds = ForensicDataset(root_dir=val_root, transform=val_transform)
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
    """在验证集上运行推理并输出 BinaryMetrics 报告。"""
    model.eval()
    evaluator = BinaryMetrics()

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(config['device'])
            labels = labels.to(config['device']).float()
            # 注意：这里需要根据你的模型返回值进行解包，TSFNet 返回 5 个值
            logits, z_sem, _, _, _, _ = model(imgs)
            evaluator.update(logits.squeeze(), labels)

    metrics = evaluator.print_report()
    return metrics


def evaluate_on_multiple_val_dirs(model, config, val_dirs: list[str]):
    """
    依次在多个验证数据集目录上评估模型。
    """
    results = []
    for vdir in val_dirs:
        val_loader = build_val_loader(config, val_root=vdir)
        if val_loader is None:
            continue

        print(f"\n[Info] Evaluating on val dir: {vdir}")
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


def evaluate_checkpoints_over_val_dirs(model, config, checkpoints_dir: str, val_dirs: list[str],
                                       specific_epoch: int | None = None):
    """
    加载指定的权重（或遍历目录下的权重），在给定的验证集上进行评估。
    """
    results = []

    # 确定要测试的权重列表
    if specific_epoch is not None:
        ckpt_path = os.path.join(checkpoints_dir, f'model_epoch_{specific_epoch}.pth')
        if not os.path.exists(ckpt_path):
            print(f"[Error] Specific epoch checkpoint not found: {ckpt_path}")
            return results
        checkpoints = [(specific_epoch, ckpt_path)]
    else:
        checkpoints = list_checkpoint_paths(checkpoints_dir)
        if not checkpoints:
            print(f"[Error] No checkpoints found under: {checkpoints_dir}")
            return results

    # 遍历每个权重文件进行评估
    for ep, path in checkpoints:
        print(f"\n{'=' * 20} Evaluating Epoch {ep} {'=' * 20}")
        print(f"[Info] Loading checkpoint: {path}")

        try:
            checkpoint = torch.load(path, map_location=config['device'], weights_only=True)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)

            # 在所有验证集上测试
            ep_results = evaluate_on_multiple_val_dirs(model, config, val_dirs)
            results.append((ep, ep_results))

        except Exception as e:
            print(f"[Error] Failed to load or evaluate checkpoint {path}: {e}")
            continue

    return results


def average_metrics_across_val_dirs(m_list: list[tuple[str, dict]]):
    """计算平均指标"""
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
    # 1. 设置 HF 镜像（如果需要）
    os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

    # 2. 加载配置
    config = load_config()
    seed_everything(config['seed'])

    # 3. 初始化模型
    print("[Info] Initializing Model...")
    model = TSFNet(config).to(config['device'])

    # 4. 从 Config 中获取推理参数
    # 验证集列表
    val_dirs_str = config.get('VAL_DIRS', "")
    if not val_dirs_str:
        print("[Error] 'VAL_DIRS' not found or empty in model_config.yaml")
        return
    val_dirs = [p.strip() for p in val_dirs_str.split(',') if p.strip()]

    # 权重目录
    checkpoint_dir = config.get('CHECKPOINT_DIR', "")
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        print(f"[Error] 'CHECKPOINT_DIR' is invalid or not found: {checkpoint_dir}")
        return

    # 指定 Epoch (可选)
    checkpoint_epoch = config.get('CHECKPOINT_EPOCH', None)
    specific_epoch = int(checkpoint_epoch) if checkpoint_epoch is not None else None

    print(f"[Info] Params -> Checkpoint Dir: {checkpoint_dir}")
    print(f"[Info] Params -> Specific Epoch: {specific_epoch if specific_epoch is not None else 'All'}")
    print(f"[Info] Params -> Val Dirs ({len(val_dirs)}):")
    for d in val_dirs:
        print(f"  - {d}")

    # 5. 执行评估
    results = evaluate_checkpoints_over_val_dirs(
        model,
        config,
        checkpoints_dir=checkpoint_dir,
        val_dirs=val_dirs,
        specific_epoch=specific_epoch,
    )

    # 6. 打印最终摘要
    print("\n\n################# Final Summary #################")
    for ep, m_list in results:
        print(f"\n>>> Epoch {ep}:")
        for vdir, metrics in m_list:
            # 提取目录名最后一部分作为简称，方便查看
            vname = os.path.basename(vdir.rstrip('/\\'))
            acc = metrics.get('Acc', 'N/A')
            auc = metrics.get('AUC', 'N/A')
            print(f"  - {vname}: Acc={acc}, AUC={auc}")

        epoch_avg = average_metrics_across_val_dirs(m_list)
        if epoch_avg:
            print(f"  [Average]: Acc={epoch_avg.get('Acc', 0):.4f}, AUC={epoch_avg.get('AUC', 0):.4f}")


if __name__ == '__main__':
    main()