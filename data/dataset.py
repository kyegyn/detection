import os
import random
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms


class ForensicDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['nature', 'ai']
        self.data = []

        for cls in self.classes:
            path = os.path.join(root_dir, cls)
            for img_name in os.listdir(path):
                self.data.append((os.path.join(path, img_name), self.classes.index(cls)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        try:
            # 尝试打开图片
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError, IOError) as e:
            # 【全栈技巧】捕获坏图异常，打印警告，但别让程序崩
            print(f"⚠️ Warning: Corrupted image found: {img_path}. Skipping...")

            # 策略：随机找一张新的代替 (递归调用自己)
            # 为了防止死循环（万一全是坏图），建议随机选一个别的索引
            new_idx = random.randint(0, len(self.data) - 1)
            return self.__getitem__(new_idx)

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))
    ])
    dataset = ForensicDataset(root_dir='/root/autodl-tmp/data/val', transform=transform)
    img_path, label = dataset.data[1]
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample {1}: Image shape: {img_path}, Label: {label}")
    # for i in range(5):
    #     img, label = dataset[i]
    #     print(f"Sample {i}: Image shape: {img.shape}, Label: {label}")