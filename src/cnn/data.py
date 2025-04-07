"""
数据加载和处理模块

提供加载和预处理MNIST数据集的功能
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class MNISTDataset(Dataset):
    """自定义MNIST数据集类"""

    def __init__(self, data, labels=None, transform=None):
        """
        初始化MNIST数据集

        参数:
            data: 图像数据
            labels: 标签数据，如果为None则为测试集
            transform: 数据变换
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像并转换为28x28的图像格式
        image = self.data[idx].reshape(28, 28).astype(np.float32)

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 如果有标签，返回图像和标签；否则只返回图像
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image


def load_data(filepath, test_size=0.2, random_state=42):
    """
    加载MNIST数据集并划分训练集和验证集

    参数:
        filepath: CSV文件路径
        test_size: 验证集比例
        random_state: 随机种子

    返回:
        train_data, train_labels, val_data, val_labels
    """
    # 加载CSV数据
    print(f"正在从{filepath}加载MNIST数据...")
    df = pd.read_csv(filepath)

    # 分离标签和特征
    labels = df["label"].values
    data = df.drop("label", axis=1).values / 255.0  # 归一化

    # 划分训练集和验证集
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"数据加载完成: {len(train_data)}个训练样本, {len(val_data)}个验证样本")

    return train_data, train_labels, val_data, val_labels


def prepare_dataloaders(train_data, train_labels, val_data, val_labels, batch_size=64):
    """
    准备DataLoader用于模型训练和验证

    参数:
        train_data: 训练数据
        train_labels: 训练标签
        val_data: 验证数据
        val_labels: 验证标签
        batch_size: 批次大小

    返回:
        train_loader, val_loader
    """
    # 定义数据变换
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # 转换为张量并缩放到[0,1]
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST数据集的标准均值和标准差
        ]
    )

    # 创建数据集
    train_dataset = MNISTDataset(train_data, train_labels, transform=transform)
    val_dataset = MNISTDataset(val_data, val_labels, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader
