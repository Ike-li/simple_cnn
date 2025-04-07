"""
CNN模型定义模块

提供简单易懂的卷积神经网络模型类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    简单的卷积神经网络模型

    架构:
    1. 卷积层1: 1通道输入, 32通道输出, 3x3卷积核
    2. 最大池化: 2x2窗口
    3. 卷积层2: 32通道输入, 64通道输出, 3x3卷积核
    4. 最大池化: 2x2窗口
    5. 全连接层1: 7*7*64 -> 128
    6. 全连接层2: 128 -> 10 (10个数字类别)
    """

    def __init__(self):
        """初始化CNN模型的层"""
        super(CNN, self).__init__()

        # 第一卷积块
        self.conv1 = nn.Conv2d(
            in_channels=1,  # MNIST图像是灰度图，只有1个通道
            out_channels=32,  # 输出32个特征图
            kernel_size=3,  # 3x3卷积核
            padding=1,  # 填充以保持空间维度
        )

        # 第二卷积块
        self.conv2 = nn.Conv2d(
            in_channels=32,  # 前一层的输出通道数
            out_channels=64,  # 输出64个特征图
            kernel_size=3,  # 3x3卷积核
            padding=1,  # 填充以保持空间维度
        )

        # 全连接层
        self.fc1 = nn.Linear(7 * 7 * 64, 128)  # 7x7是最后卷积层输出的特征图大小
        self.fc2 = nn.Linear(128, 10)  # 10个输出类别（数字0-9）

        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        前向传播函数

        参数:
            x: 输入图像，形状为 [batch_size, 1, 28, 28]

        返回:
            输出预测，形状为 [batch_size, 10]
        """
        # 第一卷积块：卷积 -> ReLU激活 -> 最大池化
        x = self.conv1(x)  # [batch_size, 32, 28, 28]
        x = F.relu(x)  # 应用ReLU激活函数
        x = F.max_pool2d(x, 2)  # 最大池化，尺寸减半 [batch_size, 32, 14, 14]

        # 第二卷积块：卷积 -> ReLU激活 -> 最大池化
        x = self.conv2(x)  # [batch_size, 64, 14, 14]
        x = F.relu(x)  # 应用ReLU激活函数
        x = F.max_pool2d(x, 2)  # 最大池化，尺寸减半 [batch_size, 64, 7, 7]

        # 扁平化特征图，准备输入全连接层
        x = x.view(-1, 7 * 7 * 64)  # [batch_size, 7*7*64]

        # 全连接层1
        x = self.fc1(x)  # [batch_size, 128]
        x = F.relu(x)  # 应用ReLU激活函数
        x = self.dropout(x)  # 应用dropout防止过拟合

        # 全连接层2（输出层）
        x = self.fc2(x)  # [batch_size, 10]

        # 返回结果（不应用softmax，因为CrossEntropyLoss会内部处理）
        return x


# 简单的函数来打印模型结构和参数数量
def print_model_summary(model):
    """打印模型结构和参数数量"""
    print(f"模型结构:\n{model}\n")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")


if __name__ == "__main__":
    # 创建模型实例
    model = CNN()

    # 打印模型摘要
    print_model_summary(model)

    # 测试前向传播
    dummy_input = torch.randn(1, 1, 28, 28)  # 创建一个随机输入张量
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
