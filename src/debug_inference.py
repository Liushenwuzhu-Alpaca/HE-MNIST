"""
密文推理调试脚本 - 使用square_()和add_plain_()
参考encrypted-evaluation-master项目
"""

import sys

sys.path.insert(0, "src")

import numpy as np
import tenseal as ts
import torch
from torchvision import datasets, transforms

from keygen import KeyGenerator


# 加载数据
print("加载MNIST数据...")
transform = transforms.ToTensor()
test_data = datasets.MNIST(
    root="./data/mnist", train=False, download=True, transform=transform
)

# 加载模型和密钥
print("加载模型和密钥...")
context = KeyGenerator.load_context("keys/context.bin")
context.auto_rescale = True
print(f"auto_rescale: {context.auto_rescale}")
print(f"global_scale: {context.global_scale}")

model = torch.load("models/mnist_net.pth", map_location="cpu")

weights = {
    "fc1.weight": model["fc1.weight"].numpy(),
    "fc1.bias": model["fc1.bias"].numpy(),
    "fc2.weight": model["fc2.weight"].numpy(),
    "fc2.bias": model["fc2.bias"].numpy(),
    "fc3.weight": model["fc3.weight"].numpy(),
    "fc3.bias": model["fc3.bias"].numpy(),
}

# 获取测试样本
img, label = test_data[0]
img_np = img.numpy().flatten().astype(np.float64)

print(f"\n测试样本: 真实标签 = {label}")

# ============ 明文推理 ============
print("\n" + "=" * 50)
print("明文推理 (ReLU)")
print("=" * 50)

x1 = np.matmul(img_np, weights["fc1.weight"].T) + weights["fc1.bias"]
x1 = np.maximum(x1, 0)
print(f"FC1输出范围: [{x1.min():.4f}, {x1.max():.4f}]")

x2 = np.matmul(x1, weights["fc2.weight"].T) + weights["fc2.bias"]
x2 = np.maximum(x2, 0)
print(f"FC2输出范围: [{x2.min():.4f}, {x2.max():.4f}]")

x3 = np.matmul(x2, weights["fc3.weight"].T) + weights["fc3.bias"]
print(f"明文预测: {np.argmax(x3)}")

# ============ 明文推理 (square激活) ============
print("\n" + "=" * 50)
print("明文推理 (square激活)")
print("=" * 50)

x1_sq = np.matmul(img_np, weights["fc1.weight"].T) + weights["fc1.bias"]
x1_sq = x1_sq**2
print(f"FC1输出范围: [{x1_sq.min():.4f}, {x1_sq.max():.4f}]")

x2_sq = np.matmul(x1_sq, weights["fc2.weight"].T) + weights["fc2.bias"]
x2_sq = x2_sq**2
print(f"FC2输出范围: [{x2_sq.min():.4f}, {x2_sq.max():.4f}]")

x3_sq = np.matmul(x2_sq, weights["fc3.weight"].T) + weights["fc3.bias"]
print(f"明文预测(square): {np.argmax(x3_sq)}")

# ============ 密文推理 ============
print("\n" + "=" * 50)
print("密文推理 (square激活)")
print("=" * 50)

# 加密输入
enc_x = ts.ckks_vector(context, img_np.tolist())

# FC1: enc_x.mm(weights) + bias -> square
print("\n--- FC1 ---")
enc_x.mm_(weights["fc1.weight"].T)
enc_x = enc_x + ts.ckks_vector(context, weights["fc1.bias"].tolist())
enc_x1_dec = np.array(enc_x.decrypt())
print(f"FC1输出范围: [{enc_x1_dec.min():.4f}, {enc_x1_dec.max():.4f}]")

# square激活
enc_x.square_()
enc_x1_sq_dec = np.array(enc_x.decrypt())
print(f"Square后范围: [{enc_x1_sq_dec.min():.4f}, {enc_x1_sq_dec.max():.4f}]")

# FC2: enc_x.mm(weights) + bias -> square
print("\n--- FC2 ---")
enc_x.mm_(weights["fc2.weight"].T)
enc_x = enc_x + ts.ckks_vector(context, weights["fc2.bias"].tolist())
enc_x2_dec = np.array(enc_x.decrypt())
print(f"FC2输出范围: [{enc_x2_dec.min():.4f}, {enc_x2_dec.max():.4f}]")

# square激活
enc_x.square_()
enc_x2_sq_dec = np.array(enc_x.decrypt())
print(f"Square后范围: [{enc_x2_sq_dec.min():.4f}, {enc_x2_sq_dec.max():.4f}]")

# FC3
print("\n--- FC3 ---")
enc_x.mm_(weights["fc3.weight"].T)
enc_x = enc_x + ts.ckks_vector(context, weights["fc3.bias"].tolist())
enc_x3_dec = np.array(enc_x.decrypt())

print(f"\n明文logits(ReLU): {x3[:5]}")
print(f"明文logits(square): {x3_sq[:5]}")
print(f"密文logits: {enc_x3_dec[:5]}")
print(f"密文预测: {np.argmax(enc_x3_dec)}")
