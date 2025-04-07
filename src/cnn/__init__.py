"""
简单卷积神经网络示例包

此包包含构建、训练和评估卷积神经网络的基础组件
"""

from cnn.model import CNN
from cnn.data import load_data, prepare_dataloaders
from cnn.train import train_model, evaluate_model
from cnn.utils import plot_training_history, visualize_predictions

__all__ = [
    'CNN',
    'load_data',
    'prepare_dataloaders',
    'train_model',
    'evaluate_model',
    'plot_training_history',
    'visualize_predictions'
] 