import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPVisionModel
from tqdm import tqdm
import os
import yaml

# 导入你之前定义的模块
from models.tsf_net import TSFNet
from data.dataset import ForensicDataset
from losses.supcon_loss import SupConLoss


def train():
    # --- 1. 加载配置 ---
    # 实际项目中建议使用 yaml.safe_load(open("configs/config.yaml"))
    config = yaml.safe_load(open("../config/model_config.yaml", 'r', encoding='utf-8'))
    # config = {
    #     'clip_model': "openai/clip-vit-base-patch32",
    #     'batch_size': 32,  # 根据显存调整，SupCon 建议越大越好
    #     'lr': 1e-4,
    #     'epochs': 50,
    #     'temp': 0.07,
    #     'lambda_supcon': 0.5,
    #     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    #     'save_path': './checkpoints'
    # }
    os.makedirs(config['save_path'], exist_ok=True)

    # --- 2. 数据准备 ---
    # 定义基础增强（注意：不要过度增强以免破坏微观伪影）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))  # CLIP 默认归一化
    ])

    train_ds = ForensicDataset(root_dir='../data/train', transform=train_transform)
    val_ds = ForensicDataset(root_dir='../data/val', transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # --- 3. 模型初始化 ---
    # 冻结的 CLIP 引擎：仅作为特征提取器
    clip_v  = CLIPVisionModel.from_pretrained(config['clip_model']).to(config['device'])
    clip_v .eval()
    for param in clip_v .parameters():
        param.requires_grad = False

    # 我们的 TSF-Net
    model = TSFNet(config).to(config['device'])

    # --- 4. 优化器与损失函数 ---
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_supcon = SupConLoss(temperature=config['temp'])

    # --- 5. 训练循环 ---
    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        for imgs, labels in loop:
            imgs, labels = imgs.to(config['device']), labels.to(config['device']).float()

            # A. 预提取 CLIP 特征 (不计梯度)
            with torch.no_grad():
                clip_out = clip_v(pixel_values=imgs).pooler_output  # [B, 768]

            # B. 前向传播
            logits, z_sem, _ = model(imgs, clip_out)

            # C. 计算复合损失
            loss_bce = criterion_bce(logits.squeeze(), labels)
            loss_sc = criterion_supcon(z_sem, labels)
            total_loss = loss_bce + config['lambda_supcon'] * loss_sc

            # D. 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 统计
            train_loss += total_loss.item()
            preds = (torch.sigmoid(logits).squeeze() > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch + 1}/{config['epochs']}]")
            loop.set_postfix(loss=total_loss.item(), acc=correct / total)

        scheduler.step()

        # --- 6. 验证环节 ---
        val_acc = validate(model, clip_v, val_loader, config)
        print(f"Epoch {epoch + 1} Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{config['save_path']}/best_model.pth")


def validate(model, clip_v, val_loader, config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(config['device']), labels.to(config['device']).float()
            clip_out = clip_v(pixel_values=imgs).pooler_output
            logits, _, _ = model(imgs, clip_out)
            preds = (torch.sigmoid(logits).squeeze() > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    train()
