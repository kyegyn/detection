import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# 引入你的模型和配置
from models.tsf_net import TSFNet
from data.dataset import ForensicDataset
# 假设你有一个 get_val_transform 用于验证集的预处理
from data.transforms import get_val_transform

def calculate_eer(y_true, y_score):
    """计算 EER (Equal Error Rate)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def evaluate(config):
    device = config['device']

    # 1. 加载模型
    print(f"🔄 Loading model from {config['checkpoint_path']}...")
    model = TSFNet(config).to(device)
    checkpoint = torch.load(config['checkpoint_path'], map_location=device, weights_only=True)

    # 兼容处理：有些保存可能是整包，有些是 state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 2. 准备数据
    test_dataset = ForensicDataset(
        root_dir=config['test_data_dir'],
        transform=get_val_transform(config['input_size'])
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # 3. 推理循环
    y_true = []
    y_scores = [] # 记录概率值
    y_preds = []  # 记录 0/1 预测结果

    print("🚀 Starting Evaluation...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)

            # 假设你的 dataset 返回 label 0=Real, 1=Fake
            # 但模型输出通常是 [B, 2] 或者 [B, 1]
            # 这里假设模型输出 logits [B, 2]

            # 需要手动提取 clip_features 或者修改模型 forward 逻辑
            # 这里简化演示，假设 model 内部处理好了 clip 逻辑，或者你需要像 train.py 一样先过 clip
            # 注意：如果你的 model forward 需要 clip_emb，这里要补上 clip 提取代码

            # --- 伪代码：如果 model 包含 clip 预处理 ---
            logits, _, _ = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1] # 取出类别 1 (Fake) 的概率

            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            y_preds.extend(preds.cpu().numpy())

    # 4. 计算指标
    acc = accuracy_score(y_true, y_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='binary')
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer = calculate_eer(y_true, y_scores)

    cm = confusion_matrix(y_true, y_preds)

    print("\n" + "="*30)
    print(f"📊 Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {roc_auc:.4f}")
    print(f"EER      : {eer:.4f}") # 论文核心指标
    print(f"F1-Score : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("="*30)

    # 5. 绘制并保存 ROC 曲线 (写论文用)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('logs/roc_curve.png')
    print("🖼️ ROC Curve saved to logs/roc_curve.png")

if __name__ == "__main__":
    conf = {
        'device': 'cuda',
        'checkpoint_path': 'checkpoints/best_model.pth', # 指向你训练好的模型
        'test_data_dir': 'data/test',
        'batch_size': 64,
        'input_size': 224, # 根据你的 resize
        # ... 其他模型参数 ...
        'clip_model': "openai/clip-vit-base-patch32",
        'embed_dim': 256
    }
    evaluate(conf)
