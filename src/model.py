"""
神经网络模型 - 使用square激活函数
"""

import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MNISTNet(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, output=10):
        """
        初始化模型

        Args:
            input_size (int): 输入大小
            hidden1 (int): 隐藏层1大小
            hidden2 (int): 隐藏层2大小
            output (int): 输出大小
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output)

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 输出张量
        """
        x = x.view(x.size(0), -1)
        x = self.fc1(x) ** 2
        x = self.fc2(x) ** 2
        return self.fc3(x)


class Trainer:
    def __init__(self, model, lr=0.001):
        """
        初始化训练器

        Args:
            model (MNISTNet): 模型
            lr (float): 学习率
        """

        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def load_data(self, data_dir="./data/mnist", batch_size=64):
        """
        加载数据

        Args:
            data_dir (str): 数据目录
            batch_size (int): 批大小
        """
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_loader = DataLoader(
            torchvision.datasets.MNIST(
                root=data_dir, train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            torchvision.datasets.MNIST(
                root=data_dir, train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
        )

    def train(self, epochs, save_path="./models/mnist_net.pth"):
        """
        训练模型

        Args:
            epochs (int): 训练轮数
            save_path (str): 保存路径
        """

        for epoch in range(1, epochs + 1):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(data), target)
                loss.backward()
                self.optimizer.step()
            acc = self.evaluate()
            print(f"Epoch {epoch}/{epochs}: 准确率 {acc:.2f}%")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def evaluate(self):
        """
        评估模型

        Returns:
            float: 准确率
        """
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data).argmax(1)
                correct += (pred == target).sum().item()
        return 100 * correct / len(self.test_loader.dataset)


if __name__ == "__main__":
    model = MNISTNet()
    trainer = Trainer(model)
    trainer.load_data()
    trainer.train(epochs=10)
