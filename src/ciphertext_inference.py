"""
密文推理模块 - 使用square激活函数
"""

import tenseal as ts
import numpy as np
from typing import List, Tuple


class CiphertextInference:
    """密文推理引擎 - 使用square激活函数"""

    def __init__(self, context: ts.Context, weights: dict):
        """
        初始化密文推理引擎

        Args:
            context: TenSEAL密钥上下文
            weights: 模型权重字典
        """
        self.context = context
        self.weights = weights
        self._load_weights()

    def _load_weights(self):
        """加载模型权重"""
        self.w1 = self.weights["fc1.weight"].numpy()
        self.b1 = self.weights["fc1.bias"].numpy()
        self.w2 = self.weights["fc2.weight"].numpy()
        self.b2 = self.weights["fc2.bias"].numpy()
        self.w3 = self.weights["fc3.weight"].numpy()
        self.b3 = self.weights["fc3.bias"].numpy()

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        """明文前向传播 (square激活)"""
        x = x.reshape(-1).astype(np.float64)

        # 归一化：与训练时相同
        x = (x - 0.1307) / 0.3081

        x1 = np.matmul(x, self.w1.T) + self.b1
        x1 = x1**2

        x2 = np.matmul(x1, self.w2.T) + self.b2
        x2 = x2**2

        x3 = np.matmul(x2, self.w3.T) + self.b3

        return x3

    def forward_encrypted(self, encrypted_x: ts.CKKSVector) -> ts.CKKSVector:
        """密文前向传播 (square激活)"""
        encrypted_x.mm_(self.w1.T)
        encrypted_x = encrypted_x + ts.ckks_vector(self.context, self.b1.tolist())
        encrypted_x.square_()

        encrypted_x.mm_(self.w2.T)
        encrypted_x = encrypted_x + ts.ckks_vector(self.context, self.b2.tolist())
        encrypted_x.square_()

        encrypted_x.mm_(self.w3.T)
        encrypted_x = encrypted_x + ts.ckks_vector(self.context, self.b3.tolist())

        return encrypted_x

    def predict_plain(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """明文预测"""
        logits = self.forward_plain(x)
        probs = self._softmax(logits)
        prediction = int(np.argmax(probs))
        return prediction, probs

    def predict_encrypted(self, encrypted_x: ts.CKKSVector) -> Tuple[int, np.ndarray]:
        """密文预测"""
        encrypted_logits = self.forward_encrypted(encrypted_x)
        logits = np.array(encrypted_logits.decrypt())
        probs = self._softmax(logits)
        prediction = int(np.argmax(probs))
        return prediction, probs

    @staticmethod
    def _softmax(x) -> np.ndarray:
        """Softmax函数"""
        if hasattr(x, "numpy"):
            x = x.numpy()
        x = np.array(x).flatten()
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from keygen import KeyGenerator
    import torch
    import torchvision.transforms as transforms
    from torchvision import datasets

    context = KeyGenerator.load_context("keys/context.bin")
    model = torch.load("models/mnist_net.pth", map_location="cpu")
    weights = model if isinstance(model, dict) else model.state_dict()

    inference = CiphertextInference(context, weights)

    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root="./data/mnist", train=False, download=True, transform=transform
    )

    img, label = test_data[0]
    img_np = img.numpy().flatten().astype(np.float64)

    plain_pred, plain_probs = inference.predict_plain(img_np)
    print(f"明文预测: {plain_pred}, 真实: {label}")

    enc_input = ts.ckks_vector(context, img_np.tolist())
    enc_pred, enc_probs = inference.predict_encrypted(enc_input)
    print(f"密文预测: {enc_pred}, 真实: {label}")
