import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt


class BinaryMetrics:
    """
    二分类指标计算器
    自动累积验证集的所有 Batch，并在 Epoch 结束时计算 AUC, EER, F1 等关键指标。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """清空缓存，开始新一轮评估"""
        self.preds = []   # 存储概率值 (0.0 - 1.0)
        self.targets = [] # 存储真实标签 (0 或 1)

    def update(self, logits, labels):
        """
        在每个 Batch 结束后调用
        Args:
            logits: 模型输出的原始 Logits [B]
            labels: 真实标签 [B]
        """
        # 1. Sigmoid 转换为概率
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # 2. 存入列表
        self.preds.extend(probs)
        self.targets.extend(labels)

    def compute(self):
        """
        计算所有累积数据的指标
        Returns:
            metrics_dict: 包含 acc, auc, eer, f1, precision, recall 的字典
        """
        y_true = np.array(self.targets)
        y_score = np.array(self.preds)

        # 默认阈值 0.5 用于计算硬分类指标
        y_pred = (y_score > 0.5).astype(int)

        # 1. 基础指标
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        # 2. AUC (Area Under Curve)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0 # 防止只有一个类别时报错

        # 3. EER (Equal Error Rate) - 取证核心指标
        eer, threshold_eer = self._calculate_eer(y_true, y_score)

        return {
            "Acc": acc,
            "AUC": auc,
            "EER": eer,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "Best_Thresh": threshold_eer # EER 对应的最佳阈值
        }

    def _calculate_eer(self, y_true, y_score):
        """
        计算 EER (Equal Error Rate)
        EER 是 FAR (False Acceptance Rate) 和 FRR (False Rejection Rate) 最接近时的值。
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

        # FRR = 1 - TPR
        frr = 1 - tpr

        # 找到 FAR 和 FRR 差值最小的点
        abs_diffs = np.abs(fpr - frr)
        min_index = np.argmin(abs_diffs)

        eer = (fpr[min_index] + frr[min_index]) / 2
        best_threshold = thresholds[min_index]

        return eer, best_threshold

    def print_report(self):
        """打印格式化的报告"""
        res = self.compute()
        print("-" * 30)
        print(f"📊 Evaluation Report:")
        print(f"Accuracy : {res['Acc']:.4f}")
        print(f"AUC      : {res['AUC']:.4f}")
        print(f"EER      : {res['EER']:.4f} (Lower is better)")
        print(f"F1-Score : {res['F1']:.4f}")
        print(f"Precision: {res['Precision']:.4f}")
        print(f"Recall   : {res['Recall']:.4f}")
        print("-" * 30)
        return res

    def plot_roc(self, save_path=None):
        """画出 ROC 曲线并保存 (可选)"""
        y_true = np.array(self.targets)
        y_score = np.array(self.preds)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
