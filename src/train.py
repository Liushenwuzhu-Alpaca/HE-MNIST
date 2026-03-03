"""
MNIST模型训练脚本 - 使用PyTorch下载MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os

os.makedirs('models', exist_ok=True)

print("下载MNIST数据集...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data/mnist', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data/mnist', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"训练集: {len(train_dataset)} 样本")
print(f"测试集: {len(test_dataset)} 样本")


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n开始训练...")
for epoch in range(3):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch+1}/3, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    acc = 100. * correct / total
    print(f'Epoch {epoch+1}/3 完成 - Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

acc = 100. * correct / total
print(f'\n测试准确率: {acc:.2f}%')

torch.save(model.state_dict(), 'models/mnist_net.pth')
print('模型已保存到 models/mnist_net.pth')
