import logging
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class ExperimentLogger:
    """
    统一日志管理类：同时处理 控制台输出、文本文件记录 和 TensorBoard 可视化
    """

    def __init__(self, log_dir, experiment_name=None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.save_dir = os.path.join(log_dir, experiment_name)
        self.tb_dir = os.path.join(self.save_dir, 'tensorboard')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # --- 1. 初始化主 Logger (输出到所有地方) ---
        self.logger = logging.getLogger("TSF-Net")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # 防止重复添加 handler

        # 格式：[时间] [级别] 消息
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Handler A: 输出到文件
        fh = logging.FileHandler(os.path.join(self.save_dir, 'train.log'), encoding='utf-8')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Handler B: 输出到控制台 (Stream)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # --- 【新增】初始化静默 Logger (只输出到文件) ---
        # 技巧：我们复用上面的 fh (FileHandler)，这样它们会写到同一个文件里，且格式一致
        self.file_logger = logging.getLogger("TSF-Net-FileOnly")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.handlers = [] # 清空默认
        self.file_logger.propagate = False # 关键：防止它把消息传给父级 logger 导致打印到控制台
        self.file_logger.addHandler(fh) # 只添加文件 Handler

        # --- 2. 初始化 TensorBoard ---
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        self.logger.info(f"🚀 Experiment initialized at: {self.save_dir}")

    # ... (log_hyperparams, log_step, log_epoch 方法保持不变) ...
    def log_hyperparams(self, config):
        """记录超参数配置"""
        self.logger.info("=== Hyperparameters ===")
        for k, v in config.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("=======================")

    def log_step(self, epoch, step, global_step, losses_dict):
        """
        Step 级别的日志 (通常只写 TensorBoard，防止文本日志爆炸)
        Args:
            losses_dict: {'total': 1.5, 'bce': 0.8, 'supcon': 0.7}
        """
        for name, value in losses_dict.items():
            self.writer.add_scalar(f'Train_Step/{name}', value, global_step)

    def log_epoch(self, epoch, train_metrics, val_metrics, lr):
        """
        Epoch 级别的日志 (记录到文本 + TensorBoard)
        """
        # 1. 记录文本
        msg = f"Epoch [{epoch}] | LR: {lr:.6f} | "
        msg += f"Train Loss: {train_metrics['loss']:.4f} | "
        msg += f"Val Acc: {val_metrics['Acc']:.4f} | Val AUC: {val_metrics['AUC']:.4f} | Val EER: {val_metrics['EER']:.4f}"
        self.logger.info(msg)

        # 2. 记录 TensorBoard - Train
        for k, v in train_metrics.items():
            self.writer.add_scalar(f'Train_Epoch/{k}', v, epoch)

        # 3. 记录 TensorBoard - Val
        for k, v in val_metrics.items():
            self.writer.add_scalar(f'Val_Epoch/{k}', v, epoch)

        # 4. 记录 LR
        self.writer.add_scalar('Hyperparams/Learning_Rate', lr, epoch)

    def log_info(self, msg):
        """通用 info 记录 (控制台 + 文件)"""
        self.logger.info(msg)

    # --- 【新增】只记录到文件的方法 ---
    def log_file_only(self, msg):
        """仅记录到文件，不干扰控制台进度条"""
        self.file_logger.info(msg)

    def close(self):
        """关闭资源"""
        self.writer.close()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        # 别忘了清理 file_logger 的 handler
        for handler in self.file_logger.handlers:
            handler.close()
            self.file_logger.removeHandler(handler)
