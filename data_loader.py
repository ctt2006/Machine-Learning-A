"""
数据加载和预处理模块
"""
import os
import glob
import numpy as np
from PIL import Image
import torch


class GlassDataset:
    """玻璃缺陷数据集"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.filenames = []
        
        # 加载图像和标签
        img_dir = os.path.join(data_dir, 'img')
        txt_dir = os.path.join(data_dir, 'txt')
        
        if not os.path.exists(img_dir):
            raise ValueError(f"图像目录不存在: {img_dir}")
        
        # 获取所有图像文件
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        for img_path in img_files:
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # 检查是否有对应的标签文件
            txt_path = os.path.join(txt_dir, name_without_ext + '.txt')
            has_defect = os.path.exists(txt_path)
            
            self.images.append(img_path)
            self.labels.append(1 if has_defect else 0)  # 1=有缺陷, 0=无缺陷
            self.filenames.append(name_without_ext)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.float32) / 255.0  # 归一化到[0,1]
        
        # 转换为CHW格式
        image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image)
        
        label = self.labels[idx]
        filename = self.filenames[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, filename


def create_data_loaders(train_dir, test_dir, batch_size=32):
    """创建训练和测试数据加载器"""
    train_dataset = GlassDataset(train_dir)
    test_dataset = GlassDataset(test_dir)
    
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    
    # 简单的数据加载器（不使用torch.utils.data.DataLoader，因为需要手动实现）
    class SimpleDataLoader:
        def __init__(self, dataset, batch_size, shuffle=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = list(range(len(dataset)))
            if shuffle:
                np.random.shuffle(self.indices)
        
        def __iter__(self):
            # 每次迭代时重新打乱（如果是训练模式）
            if self.shuffle:
                np.random.shuffle(self.indices)
            for i in range(0, len(self.dataset), self.batch_size):
                batch_indices = self.indices[i:i+self.batch_size]
                batch = [self.dataset[idx] for idx in batch_indices]
                yield collate_fn(batch)
        
        def __len__(self):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    train_loader = SimpleDataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = SimpleDataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_loader, test_loader, test_dataset
