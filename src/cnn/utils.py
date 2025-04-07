"""
工具函数模块

提供用于可视化和其他辅助功能的函数
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns


def set_seed(seed=42):
    """
    设置随机种子，以确保结果可重现

    参数:
        seed: 随机种子值
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")


def get_device():
    """
    获取可用的计算设备

    返回:
        device: 可用设备 ('cuda' 或 'cpu')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")

    return device


def plot_training_history(history):
    """
    绘制训练历史曲线

    参数:
        history: 包含训练历史的字典，包括train_loss, train_acc, val_loss, val_acc
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="训练损失")
    plt.plot(history["val_loss"], label="验证损失")
    plt.xlabel("轮数")
    plt.ylabel("损失")
    plt.title("训练和验证损失")
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="训练准确率")
    plt.plot(history["val_acc"], label="验证准确率")
    plt.xlabel("轮数")
    plt.ylabel("准确率 (%)")
    plt.title("训练和验证准确率")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, data_loader, num_samples=10, device="cpu"):
    """
    可视化模型预测结果

    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        num_samples: 要可视化的样本数量
        device: 计算设备
    """
    # 确保模型处于评估模式
    model.eval()

    # 获取一批数据
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # 限制样本数量
    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    labels = labels[:num_samples]

    # 进行预测
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        _, predicted = torch.max(outputs, 1)

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 显示图像和预测结果
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        # 转换图像格式，从[1, 28, 28]转为[28, 28]
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap="gray")

        true_label = labels[i].item()
        pred_label = predicted[i].item()

        # 设置标题颜色，正确为绿色，错误为红色
        color = "green" if true_label == pred_label else "red"
        plt.title(f"预测: {pred_label}\n实际: {true_label}", color=color)

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    绘制混淆矩阵

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 如果没有提供类别名称，则使用数字作为类别名称
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    # 设置图像大小
    plt.figure(figsize=(10, 8))

    # 使用seaborn绘制热图
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    # 设置标题和标签
    plt.title("混淆矩阵")
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")

    plt.tight_layout()
    plt.show()


def show_sample_images(data_loader, num_samples=10):
    """
    显示数据集中的样本图像

    参数:
        data_loader: 数据加载器
        num_samples: 要显示的样本数量
    """
    # 获取一批数据
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # 限制样本数量
    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    labels = labels[:num_samples]

    # 创建图表
    plt.figure(figsize=(12, 4))

    # 显示图像
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        # 转换图像格式，从[1, 28, 28]转为[28, 28]
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap="gray")
        plt.title(f"标签: {labels[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
