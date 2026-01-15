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
from PIL import Image
from io import BytesIO
from torch.amp import autocast, GradScaler

# 1. 获取当前脚本的绝对路径
current_path = os.path.abspath(__file__)
# 2. 获取当前脚本所在的目录
script_dir = os.path.dirname(current_path)
# 3. 获取项目的根目录
project_root = os.path.dirname(script_dir)
# 4. 将项目根目录添加到系统路径中
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入你之前定义的模块
from models.tsf_net import TSFNet
from data.dataset import ForensicDataset
from losses.supcon_loss import SupConLoss
from utils.fft_utils import seed_everything
from utils.metrics import BinaryMetrics
from utils.logger import ExperimentLogger
from models.fusion import OrthogonalLoss, FineGrainedDecorrelationLoss


# --- 1. 定义在全局范围 ---
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RandomJPEGCompression(object):
    def __init__(self, quality_range=(50, 90)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        output_buffer = BytesIO()
        img.save(output_buffer, format='JPEG', quality=quality)
        output_buffer.seek(0)
        return Image.open(output_buffer)


def train():
    # --- 1. 加载配置 ---
    # 请确保路径正确
    config = yaml.safe_load(open("/root/autodl-tmp/detection/config/model_config.yaml", 'r', encoding='utf-8'))

    seed_everything(config['seed'])
    exp_name = config.get('exp_name') or f"exp_{time.strftime('%Y%m%d_%H%M%S')}_seed{config['seed']}"
    os.makedirs(f"{config['save_path']}/{exp_name}", exist_ok=True)
    logger = ExperimentLogger(log_dir=f"./{config['logs_path']}", experiment_name=exp_name)
    logger.log_hyperparams(config)

    # --- 2. 数据准备 ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0))
        ], p=0.1),
        transforms.RandomApply([
            RandomJPEGCompression(quality_range=(30, 100))
        ], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))
    ])

    train_ds = ForensicDataset(root_dir='/root/autodl-tmp/data/train', transform=train_transform)
    val_ds = ForensicDataset(root_dir='/root/autodl-tmp/data/val', transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              persistent_workers=True, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4,
                            persistent_workers=True, pin_memory=True, worker_init_fn=worker_init_fn)

    # --- 3. 模型初始化 ---
    model = TSFNet(config).to(config['device'])

    # --- 4. 优化器与损失函数 ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=1e-2
    )

    # --- 【关键修改 A】 初始化 CosineAnnealingLR ---
    # T_max: 设为总 Epochs，让学习率在一个完整的训练周期内下降到 eta_min
    # eta_min: 最小学习率，防止降为 0，建议设为 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )

    # --- 断点续训逻辑 ---
    resume_epoch = 0
    resume_path = f"{config['save_path']}/{exp_name}/model_epoch_{config['resume_epoch']}.pth"
    best_val_acc = 0.0

    if config['resume_epoch'] != resume_epoch and resume_path and os.path.exists(resume_path):
        logger.log_info(f"🔄 Resuming training from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=config['device'], weights_only=True)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 注意：严格来说，这里也应该加载 scheduler 的状态
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        resume_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.log_info(f"👉 Successfully loaded. Resuming from Epoch {resume_epoch + 1}")

    # 初始化损失函数
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_supcon = SupConLoss(temperature=config['temp'])
    # criterion_orth = OrthogonalLoss()
    criterion_decorr = FineGrainedDecorrelationLoss()

    scaler = GradScaler('cuda')

    # --- 5. 训练循环 ---
    logger.log_info("🚀 Start Training...")

    for epoch in range(resume_epoch, config['epochs']):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # 获取当前学习率用于打印
        current_lr = optimizer.param_groups[0]['lr']

        loop = tqdm(train_loader, leave=True)
        for batch_idx, (imgs, labels) in enumerate(loop):
            imgs, labels = imgs.to(config['device']), labels.to(config['device']).float()

            optimizer.zero_grad()

            with autocast('cuda'):
                # 前向传播
                logits, z_sem, _, f_sem_raw, v_forensic, alpha, f_tex_global, z_freq = model(imgs)

                # 计算损失
                loss_bce = criterion_bce(logits.squeeze(), labels)
                loss_sc = criterion_supcon(z_sem, labels)
                total_loss = loss_bce + config['lambda_supcon'] * loss_sc

                # loss_orth_val = 0.0
                loss_decorr_val = 0.0
                if config.get('fusion_type') == 'gating':
                    # loss_orth_val = criterion_orth(f_sem_raw, v_forensic)
                    # total_loss += config.get('lambda_orth', 0.1) * loss_orth_val
                    loss_decorr_val = criterion_decorr(f_sem_raw, f_tex_global, z_freq)
                    total_loss += config.get('lambda_decorr', 0.01) * loss_decorr_val

                # ---------------------------------------------------
                # 【新增】计算 Gating Regularization Loss
                # ---------------------------------------------------
                loss_gate_val = 0.0
                loss_entropy = 0.0
                if config.get('fusion_type') == 'gating' and alpha is not None:
                    # # 1. 计算均值和方差
                    # # alpha 形状是 [B, 1]，先 squeeze 成 [B]
                    # alpha_squeeze = alpha.squeeze()
                    #
                    # alpha_mean = alpha_squeeze.mean()
                    # alpha_var = alpha_squeeze.var()  # 无偏估计方差
                    #
                    # # 2. 定义权重 (可以在 config 里配，这里先硬编码示例)
                    # # λ_var: 鼓励方差大 (负号在公式里)
                    # # λ_mean: 锚定均值
                    # lambda_var = 0.01  # 这是一个经验值，不要太大，否则梯度会爆
                    # lambda_mean = 0.001  # "极弱"锚定
                    #
                    # # 3. 计算损失
                    # # 我们希望 Var 变大 -> Loss 变小 -> -Var
                    # loss_var_term = -alpha_var
                    # loss_mean_term = (alpha_mean - 0.5) ** 2
                    #
                    # loss_gate_val = lambda_var * loss_var_term + lambda_mean * loss_mean_term
                    #
                    # # 加入总损失
                    # total_loss += config.get('lambda_gate', 0.01)* loss_gate_val

                    # 1. 取出 alpha [B, 1] -> [B]
                    alpha_squeeze = alpha.squeeze()

                    # 2. 计算熵 (Entropy)
                    # H(p) = - [p*log(p) + (1-p)*log(1-p)]
                    # 加 epsilon 防止 log(0)
                    eps = 1e-8
                    entropy = - (alpha_squeeze * torch.log(alpha_squeeze + eps) +
                                 (1 - alpha_squeeze) * torch.log(1 - alpha_squeeze + eps))

                    # 3. 计算 Loss = - mean(Entropy)
                    # 我们希望 Entropy 最大 -> Loss 最小
                    loss_entropy = - entropy.mean()

                    # 4. 加权
                    # 建议权重：0.01 ~ 0.1，如果 Alpha 依然卡在 0.9，就加大到 0.1
                    # 加入总 Loss
                    total_loss += config.get('lambda_entropy', 0.001) * loss_entropy

            # 反向传播与更新
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 为了防止变量未定义，先获取数值（安全获取）
            val_bce = loss_bce.item()
            val_sc  = loss_sc.item()
            # val_orth = loss_orth_val.item() if isinstance(loss_orth_val, torch.Tensor) else 0.0
            val_decorr = loss_decorr_val.item() if isinstance(loss_decorr_val, torch.Tensor) else 0.0
            val_ent  = loss_entropy.item() if 'loss_entropy' in locals() and isinstance(loss_entropy, torch.Tensor) else 0.0

            # 记录日志
            global_step = epoch * len(train_loader) + batch_idx
            losses_dict = {
                'total': total_loss.item(),
                'bce': val_bce,
                'supcon': val_sc,
                'decorr': val_decorr,
                'entropy': val_ent,
            }
            logger.log_step(epoch, batch_idx, global_step, losses_dict)

            # 统计指标
            train_loss += total_loss.item()
            preds = (torch.sigmoid(logits).squeeze() > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 显存监控与打印
            if batch_idx % 100 == 0:
                mem_alloc = torch.cuda.memory_allocated() / 1024 ** 3
                mem_res = torch.cuda.memory_reserved() / 1024 ** 3
                mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                torch.cuda.reset_peak_memory_stats()
                mem_info = f"Mem: {mem_alloc:.2f}G(A) / {mem_res:.2f}G(R) / {mem_peak:.2f}G(P)"
                # 这里加入了 LR 的打印
                loss_detail = f"Loss: {total_loss.item():.4f} [BCE:{val_bce:.4f} SC:{val_sc:.4f}"
                # 如果有 gating 相关的损失，也打印出来
                if config.get('fusion_type') == 'gating':
                    loss_detail += f" Decorr:{val_decorr:.4f} Ent:{val_ent:.4f}"
                loss_detail += "]"
                logger.log_file_only(
                    f"Epoch [{epoch + 1}] Step [{batch_idx}] LR: {current_lr:.6f} | {loss_detail} | Acc: {correct / total:.4f} | {mem_info}"
                )

            loop.set_description(f"Epoch [{epoch + 1}/{config['epochs']}]")
            loop.set_postfix(loss=total_loss.item(), acc=correct / total, lr=current_lr)

        # --- 【关键修改 B】 在 Epoch 结束时更新学习率 ---
        scheduler.step()

        # --- 6. 验证环节 ---
        metrics = validate(model, val_loader, config)
        logger.log_file_only(f"Epoch {epoch + 1} Val Acc: {metrics['Acc']:.4f}")

        # 记录 Epoch 级指标
        train_epoch_metrics = {'loss': train_loss / len(train_loader)}
        logger.log_epoch(epoch + 1, train_epoch_metrics, metrics, current_lr)

        # 保存模型
        current_save_path = f"{config['save_path']}/{exp_name}/model_epoch_{epoch + 1}.pth"
        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # 建议加入这个
            'best_val_acc': best_val_acc
        }
        torch.save(checkpoint_dict, current_save_path)
        logger.log_file_only(f"Saved checkpoint to {current_save_path}")

        if metrics['Acc'] > best_val_acc:
            best_val_acc = metrics['Acc']
            logger.log_file_only(f"🔥 New Best Model saved with Acc: {best_val_acc:.4f} at Epoch [{epoch + 1}]")


def validate(model, val_loader, config):
    model.eval()
    evaluator = BinaryMetrics()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(config['device'])
            labels = labels.to(config['device']).float()
            # 这里的 _ 占位符数量要根据你的模型返回值匹配，这里假设是 5 个
            logits, z_sem, _, _, _, _, _, _ = model(imgs)
            evaluator.update(logits.squeeze(), labels)
    return evaluator.print_report()


if __name__ == "__main__":
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    train()
