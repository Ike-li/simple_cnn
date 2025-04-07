"""
模型训练和评估模块

提供用于训练和评估CNN模型的函数
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def train_model(
    model, train_loader, val_loader, epochs=10, learning_rate=0.001, device="cpu"
):
    """
    训练CNN模型

    参数:
        model: CNN模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备 ('cpu' or 'cuda')

    返回:
        history: 包含训练和验证损失、准确率的历史记录字典
    """
    # 移动模型到指定设备
    model = model.to(device)
    print(f"模型已加载到设备: {device}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 用于保存训练历史的字典
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 打印训练参数
    print(f"\n开始训练，共{epochs}轮，学习率={learning_rate}")
    print("-" * 60)

    # 训练循环
    for epoch in range(epochs):
        # 记录开始时间
        start_time = time.time()

        # 训练模式
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 遍历训练数据
        for batch_idx, (data, target) in enumerate(train_loader):
            # 移动数据到指定设备
            data, target = data.to(device), target.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            output = model(data)

            # 计算损失
            loss = criterion(output, target)

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            # 累加损失
            train_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            # 每100个批次打印一次进度
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"轮数: {epoch + 1}/{epochs} "
                    f"[{batch_idx + 1}/{len(train_loader)}] "
                    f"损失: {loss.item():.4f} "
                    f"准确率: {100 * train_correct / train_total:.2f}%"
                )

        # 计算平均训练损失和准确率
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # 验证模式
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # 不需要计算梯度
        with torch.no_grad():
            for data, target in val_loader:
                # 移动数据到指定设备
                data, target = data.to(device), target.to(device)

                # 前向传播
                output = model(data)

                # 计算损失
                loss = criterion(output, target)

                # 累加损失
                val_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        # 计算平均验证损失和准确率
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 计算训练所用时间
        elapsed_time = time.time() - start_time

        # 打印轮次结果
        print(
            f"轮数: {epoch + 1}/{epochs} - "
            f"用时: {elapsed_time:.2f}秒 - "
            f"训练损失: {train_loss:.4f} - "
            f"训练准确率: {train_acc:.2f}% - "
            f"验证损失: {val_loss:.4f} - "
            f"验证准确率: {val_acc:.2f}%"
        )
        print("-" * 60)

    print(f"训练完成！最终验证准确率: {val_acc:.2f}%")

    return history


def evaluate_model(model, test_loader, device="cpu"):
    """
    评估模型性能

    参数:
        model: 训练好的CNN模型
        test_loader: 测试数据加载器
        device: 评估设备 ('cpu' or 'cuda')

    返回:
        accuracy: 模型准确率
        y_true: 真实标签
        y_pred: 预测标签
    """
    # 移动模型到指定设备
    model = model.to(device)
    model.eval()

    # 初始化
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    # 不需要计算梯度
    with torch.no_grad():
        for data, target in test_loader:
            # 移动数据到指定设备
            data, target = data.to(device), target.to(device)

            # 前向传播
            output = model(data)

            # 获取预测结果
            _, predicted = torch.max(output.data, 1)

            # 累加统计
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 收集真实标签和预测标签用于后续分析
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 计算准确率
    accuracy = 100 * correct / total
    print(f"测试准确率: {accuracy:.2f}%")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, digits=4))

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵:")
    print(cm)

    return accuracy, np.array(y_true), np.array(y_pred)


def save_model(model, filepath):
    """
    保存模型到文件

    参数:
        model: 要保存的模型
        filepath: 保存路径
    """
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存到 {filepath}")


def load_model(model, filepath, device="cpu"):
    """
    从文件加载模型

    参数:
        model: 空模型实例
        filepath: 模型文件路径
        device: 加载设备

    返回:
        model: 加载了权重的模型
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型已从 {filepath} 加载")
    return model
