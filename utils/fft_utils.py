import random
import os
import numpy as np
import torch


def seed_everything(seed):
    """
    固定所有随机种子，确保实验可复现
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU

    # --- 牺牲一点点速度换取极致的确定性 (可选) ---
    # 如果追求绝对一致，把下面两行取消注释
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # 也就是通常为了速度，我们保持 benchmark=True (默认)，这会导致卷积算法选择有微小波动，但影响不大。
