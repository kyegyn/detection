import os
from PIL import Image
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
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

