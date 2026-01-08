import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import yaml
import time
import sys

# 1. 获取当前脚本的绝对路径 (例如: /root/.../detection/scripts/train.py)
current_path = os.path.abspath(__file__)

# 2. 获取当前脚本所在的目录 (例如: /root/.../detection/scripts)
script_dir = os.path.dirname(current_path)

# 3. 获取项目的根目录，即 scripts 的上一级 (例如: /root/.../detection)
project_root = os.path.dirname(script_dir)

# 4. 将项目根目录添加到系统路径中，这样就能找到 models 了
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入你之前定义的模块
from models.tsf_net import TSFNet
from data.dataset import ForensicDataset
from losses.supcon_loss import SupConLoss
from utils.fft_utils import seed_everything
from utils.metrics import BinaryMetrics
from utils.logger import ExperimentLogger


# --- 1. 定义在全局范围 ---
def worker_init_fn(worker_id):
    """
    这个函数必须定义在全局，Windows 才能序列化它。
    PyTorch 的 DataLoader 会自动处理基础种子，我们只需要取出当前 worker 的种子信息即可。
    """
    # 获取 PyTorch 为当前 worker 分配的种子
    worker_seed = torch.initial_seed() % 2**32

    # 设置 Python 和 NumPy 的种子
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train():
    # --- 1. 加载配置 ---
    # 实际项目中建议使用 yaml.safe_load(open("configs/config.yaml"))
    # config = yaml.safe_load(open("../config/model_config.yaml", 'r', encoding='utf-8'))
    config = yaml.safe_load(open("/root/autodl-tmp/detection/config/model_config.yaml", 'r', encoding='utf-8'))
    seed_everything(config['seed'])
    exp_name = config.get('exp_name') or f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(f"{config['save_path']}/{exp_name}", exist_ok=True)
    logger = ExperimentLogger(log_dir=f"./{config['logs_path']}", experiment_name=exp_name)
    logger.log_hyperparams(config)
    # --- 2. 数据准备 ---
    # 定义基础增强（注意：不要过度增强以免破坏微观伪影）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))  # CLIP 默认归一化
    ])
    train_ds = ForensicDataset(root_dir='/root/autodl-tmp/data/train', transform=train_transform)
    # train_ds = ForensicDataset(root_dir='Z:/genimage/imagenet_ai_0419_sdv4/train', transform=train_transform)
    val_ds = ForensicDataset(root_dir='/root/autodl-tmp/data/val', transform=train_transform)
    # val_ds = ForensicDataset(root_dir='Z:/genimage/imagenet_ai_0419_sdv4/val', transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              persistent_workers=True, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4,
                            persistent_workers=True, pin_memory=True, worker_init_fn=worker_init_fn)

    # 我们的 TSF-Net
    model = TSFNet(config).to(config['device'])
    # --- 4. 优化器与损失函数 ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    # --- 【新增】断点续训逻辑 ---
    resume_epoch = 0  # 默认为 0，表示从头训练
    resume_path = f"{config['save_path']}/model_epoch_{config['resume_epoch']}.pth"
    best_val_acc = 0.0
    # 如果指定了路径，且文件存在
    if config['resume_epoch'] != resume_epoch and resume_path and os.path.exists(resume_path):
        logger.log_info(f"🔄 Resuming training from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=config['device'], weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 需要先定义 optimizer
        resume_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        logger.log_info(f"👉 Successfully loaded. Resuming from Epoch {resume_epoch + 1}")

    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_supcon = SupConLoss(temperature=config['temp'])

    # --- 5. 训练循环 ---

    for epoch in range(resume_epoch, config['epochs']):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        for batch_idx, (imgs, labels) in enumerate(loop):

            imgs, labels = imgs.to(config['device']), labels.to(config['device']).float()

            # 直接把图扔给模型，模型内部会处理 CLIP (包括 LoRA 的梯度更新)
            logits, z_sem, _ = model(imgs)

            # C. 计算复合损失
            loss_bce = criterion_bce(logits.squeeze(), labels)
            loss_sc = criterion_supcon(z_sem, labels)
            total_loss = loss_bce + config['lambda_supcon'] * loss_sc

            # D. 反向传播
            optimizer.zero_grad()
            # 计算 global_step 用于 tensorboard x轴
            global_step = epoch * len(train_loader) + batch_idx

            # 构造 Loss 字典 (这就是监控多任务平衡的关键)
            losses_dict = {
                'total': total_loss.item(),
                'bce': loss_bce.item(),
                'supcon': loss_sc.item()  # 观察这个，看聚类是否生效
            }

            # 记录
            logger.log_step(epoch, batch_idx, global_step, losses_dict)
            total_loss.backward()
            optimizer.step()
            # 统计
            train_loss += total_loss.item()
            preds = (torch.sigmoid(logits).squeeze() > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # 放在 train.py 的循环里
            if batch_idx % 100 == 0:
                # 1. 获取显存指标 (转换为 GB)
                mem_alloc = torch.cuda.memory_allocated() / 1024 ** 3
                mem_res = torch.cuda.memory_reserved() / 1024 ** 3
                mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                # 2. 【关键】重置峰值统计
                # 这样下一次 loop 看到的 peak 就是未来这 50 个 batch 里的新峰值
                torch.cuda.reset_peak_memory_stats()
                mem_info = f"Mem: {mem_alloc:.2f}G(A) / {mem_res:.2f}G(R) / {mem_peak:.2f}G(P)"
                logger.log_file_only(f"Epoch [{epoch + 1}] Step [{batch_idx}] Loss: {total_loss.item():.4f} | Acc: {correct / total} | {mem_info}")

            loop.set_description(f"Epoch [{epoch + 1}/{config['epochs']}]")
            loop.set_postfix(loss=total_loss.item(), acc=correct / total)

        scheduler.step()

        # --- 6. 验证环节 ---
        metrics = validate(model, val_loader, config)
        logger.log_file_only(f"Epoch {epoch + 1} Val Acc: {metrics['Acc']:.4f}")
        # 获取当前 LR
        current_lr = optimizer.param_groups[0]['lr']

        # 构造 epoch 级指标
        train_epoch_metrics = {'loss': train_loss / len(train_loader)}  # 可以加 train_acc

        # 记录日志 (metrics 是 validate 返回的那个丰富字典)
        logger.log_epoch(epoch + 1, train_epoch_metrics, metrics, current_lr)

        # 保存当前 Epoch 的权重
        current_save_path = f"{config['save_path']}/{exp_name}/model_epoch_{epoch + 1}.pth"
        # 推荐的保存方式 (保存更多元数据)
        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # 恢复动量等信息
            'best_val_acc': best_val_acc
        }
        torch.save(checkpoint_dict, current_save_path)
        # 打印一条日志方便确认
        logger.log_file_only(f"Saved checkpoint to {current_save_path}")
        # 根据 EER 或 Acc 保存模型
        if metrics['Acc'] > best_val_acc:
            best_val_acc = metrics['Acc']
            logger.log_file_only(f"🔥 New Best Model saved with Acc: {best_val_acc:.4f} at Epoch [{epoch + 1}]")
            # torch.save(model.state_dict(), f"{config['save_path']}/best_model.pth")


def validate(model, val_loader, config):
    model.eval()

    # 初始化指标计算器
    evaluator = BinaryMetrics()

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(config['device'])
            labels = labels.to(config['device']).float()

            logits, z_sem, _ = model(imgs)

            evaluator.update(logits.squeeze(), labels)

    # 4. 计算并打印报告
    metrics = evaluator.print_report()

    # 5. (可选) 保存 ROC 曲线
    # evaluator.plot_roc(save_path=f"{config['save_path']}/val_roc.png")

    # 返回主要指标用于 Model Checkpoint (通常用 Accuracy 或 AUC)
    return metrics


if __name__ == "__main__":
    # os.environ["http_proxy"] = "http://127.0.0.1:7890"
    # os.environ["https_proxy"] = "http://127.0.0.1:7890"
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    train()
