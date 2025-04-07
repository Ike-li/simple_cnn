"""
卷积神经网络 MNIST 手写数字识别

这是一个简单易懂的卷积神经网络(CNN)示例，用于演示如何使用PyTorch构建CNN进行手写数字识别
"""

import os
import argparse
import torch

from cnn.model import CNN, print_model_summary
from cnn.data import load_data, prepare_dataloaders
from cnn.train import train_model, evaluate_model, save_model, load_model
from cnn.utils import (
    set_seed, get_device, plot_training_history, 
    visualize_predictions, plot_confusion_matrix, show_sample_images
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CNN MNIST 手写数字识别')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='mnist-demo.csv',
                        help='MNIST数据集文件路径')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='验证集比例')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='mnist_cnn.pth',
                        help='模型保存文件名')
    
    # 运行模式
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'visualize'], 
                        default='train', help='运行模式：训练/测试/可视化')
    parser.add_argument('--visualize_samples', type=int, default=10,
                        help='可视化样本数量')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备
    device = get_device()
    
    # 加载数据
    train_data, train_labels, val_data, val_labels = load_data(
        args.data_path, test_size=args.test_size
    )
    
    # 准备数据加载器
    train_loader, val_loader = prepare_dataloaders(
        train_data, train_labels, val_data, val_labels, batch_size=args.batch_size
    )
    
    # 确保模型保存目录存在
    os.makedirs(args.model_path, exist_ok=True)
    model_path = os.path.join(args.model_path, args.model_name)
    
    # 运行相应模式
    if args.mode == 'train':
        # 创建模型
        model = CNN()
        print_model_summary(model)
        
        # 训练模型
        history = train_model(
            model, train_loader, val_loader, 
            epochs=args.epochs, learning_rate=args.learning_rate, device=device
        )
        
        # 保存模型
        save_model(model, model_path)
        
        # 绘制训练历史
        plot_training_history(history)
        
        # 在验证集上评估模型
        accuracy, y_true, y_pred = evaluate_model(model, val_loader, device=device)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(y_true, y_pred)
        
    elif args.mode == 'test':
        # 创建模型
        model = CNN()
        
        # 加载训练好的模型
        model = load_model(model, model_path, device=device)
        
        # 在验证集上评估模型
        accuracy, y_true, y_pred = evaluate_model(model, val_loader, device=device)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(y_true, y_pred)
        
        # 可视化一些预测结果
        visualize_predictions(model, val_loader, num_samples=args.visualize_samples, device=device)
        
    elif args.mode == 'visualize':
        # 显示一些样本图像
        show_sample_images(train_loader, num_samples=args.visualize_samples)
        
        # 如果模型文件存在
        if os.path.exists(model_path):
            # 创建模型
            model = CNN()
            
            # 加载训练好的模型
            model = load_model(model, model_path, device=device)
            
            # 可视化一些预测结果
            visualize_predictions(model, val_loader, num_samples=args.visualize_samples, device=device)
        else:
            print(f"模型文件 {model_path} 不存在，无法可视化预测结果")


if __name__ == "__main__":
    main() 