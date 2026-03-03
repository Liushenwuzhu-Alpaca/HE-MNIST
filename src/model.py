"""
神经网络模型模块 - PyTorch全连接网络用于MNIST手写数字识别
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json


class MNISTNet(nn.Module):
    """用于MNIST手写数字识别的全连接神经网络"""

    def __init__(
        self,
        input_size: int = 784,
        hidden1_size: int = 256,
        hidden2_size: int = 128,
        output_size: int = 10,
    ):
        """
        初始化网络结构

        Args:
            input_size: 输入维度 (MNIST 28x28 = 784)
            hidden1_size: 第一隐藏层维度
            hidden2_size: 第二隐藏层维度
            output_size: 输出维度 (数字0-9)
        """
        super(MNISTNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def get_weights(self) -> dict:
        """
        获取模型权重

        Returns:
            包含所有权重和偏置的字典
        """
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights

    def set_weights(self, weights: dict):
        """
        设置模型权重

        Args:
            weights: 权重字典
        """
        for name, param in self.named_parameters():
            if name in weights:
                param.data = torch.from_numpy(weights[name]).float()

    def get_weight_matrix(self, layer_name: str) -> np.ndarray:
        """获取指定层的权重矩阵"""
        return self.state_dict()[layer_name + ".weight"].cpu().numpy()

    def get_bias_vector(self, layer_name: str) -> np.ndarray:
        """获取指定层的偏置向量"""
        return self.state_dict()[layer_name + ".bias"].cpu().numpy()


class ModelTrainer:
    """模型训练器"""

    def __init__(
        self, model: nn.Module, learning_rate: float = 0.001, device: str = None
    ):
        """
        初始化训练器

        Args:
            model: 神经网络模型
            learning_rate: 学习率
            device: 计算设备
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def load_mnist(
        self,
        data_dir: str = "./data/mnist",
        batch_size: int = 64,
        download: bool = True,
    ):
        """加载MNIST数据集"""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=download, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=download, transform=transform
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        return self.train_loader, self.test_loader

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}"
                )

        return total_loss / len(self.train_loader)

    def evaluate(self) -> tuple:
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_loss = test_loss / len(self.test_loader)

        return accuracy, avg_loss

    def train(self, epochs: int = 10, save_path: str = "./models/mnist_net.pth"):
        """
        训练模型

        Args:
            epochs: 训练轮数
            save_path: 模型保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"开始训练，设备: {self.device}")
        print(f"训练轮数: {epochs}")

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            avg_loss = self.train_epoch()
            accuracy, test_loss = self.evaluate()

            print(f"  训练损失: {avg_loss:.4f}")
            print(f"  测试损失: {test_loss:.4f}")
            print(f"  准确率: {accuracy:.2f}%")

        torch.save(self.model.state_dict(), save_path)
        print(f"\n模型已保存至: {save_path}")

        return self.model

    def load(self, load_path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        return self.model


def train_model(epochs: int = 10, save_path: str = "./models/mnist_net.pth"):
    """
    便捷函数：训练MNIST模型

    Args:
        epochs: 训练轮数
        save_path: 模型保存路径
    """
    model = MNISTNet()
    trainer = ModelTrainer(model)
    trainer.load_mnist()
    trainer.train(epochs=epochs, save_path=save_path)
    return model


def get_model_weights(model: nn.Module) -> dict:
    """
    便捷函数：获取模型权重

    Args:
        model: 神经网络模型

    Returns:
        权重字典
    """
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    return weights


def create_model() -> MNISTNet:
    """创建模型实例"""
    return MNISTNet()


if __name__ == "__main__":
    train_model(epochs=5)
