# 简单卷积神经网络 (CNN) 示例

这是一个简单易懂的卷积神经网络 (CNN) 项目，用于 MNIST 手写数字识别。该项目旨在展示 CNN 的基本原理和实现方式，适合深度学习入门者学习。

## 项目特点

- 简单清晰的CNN架构，易于理解
- 详细的代码注释，解释每一步的目的和作用
- 完整的训练、评估和可视化流程
- 基于PyTorch框架实现

## 项目结构

```
src/
├── cnn/                  # CNN模块
│   ├── __init__.py       # 初始化文件
│   ├── data.py           # 数据加载和处理
│   ├── model.py          # CNN模型定义
│   ├── train.py          # 训练和评估函数
│   └── utils.py          # 辅助函数
├── main.py               # 主程序入口
├── models/               # 保存训练模型的目录（运行时创建）
└── mnist-demo.csv        # MNIST数据集（CSV格式）
```

## 环境要求

- Python 3.10+
- PyTorch 2.0.0+
- torchvision
- matplotlib
- scikit-learn
- pandas

## 快速开始

### 安装依赖

使用Poetry安装依赖：

```bash
cd src
poetry install
```

或者使用pip：

```bash
pip install torch torchvision matplotlib scikit-learn pandas
```

### 训练模型

```bash
cd src
python main.py --mode train --epochs 5 --batch_size 64
```

### 测试模型

```bash
cd src
python main.py --mode test
```

### 可视化数据和预测

```bash
cd src
python main.py --mode visualize --visualize_samples 10
```

## 命令行参数

- `--data_path`: MNIST数据集文件路径（默认：'mnist-demo.csv'）
- `--test_size`: 验证集比例（默认：0.2）
- `--epochs`: 训练轮数（默认：5）
- `--batch_size`: 批次大小（默认：64）
- `--learning_rate`: 学习率（默认：0.001）
- `--seed`: 随机种子（默认：42）
- `--model_path`: 模型保存目录（默认：'models'）
- `--model_name`: 模型保存文件名（默认：'mnist_cnn.pth'）
- `--mode`: 运行模式（'train', 'test', 'visualize'）（默认：'train'）
- `--visualize_samples`: 可视化样本数量（默认：10）

## CNN 模型结构

该项目实现的CNN模型具有以下结构：

1. 第一个卷积层：1通道输入，32通道输出，3x3卷积核
2. ReLU激活函数
3. 最大池化：2x2窗口
4. 第二个卷积层：32通道输入，64通道输出，3x3卷积核
5. ReLU激活函数
6. 最大池化：2x2窗口
7. 全连接层1：7*7*64 → 128
8. ReLU激活函数
9. Dropout层（防止过拟合）
10. 全连接层2：128 → 10（10个数字类别）

## 参考资料

- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [卷积神经网络简介](https://cs231n.github.io/convolutional-networks/)
- [MNIST数据集](http://yann.lecun.com/exdb/mnist/)
