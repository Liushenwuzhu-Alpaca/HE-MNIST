"""
密文推理模块 - 在加密数据上执行神经网络前向传播
"""

import tenseal as ts
import numpy as np
from typing import List, Tuple, Optional
import torch


class CiphertextInference:
    """密文推理引擎"""

    def __init__(self, context: ts.Context, weights: dict, scale: float = None):
        """
        初始化密文推理引擎

        Args:
            context: TenSEAL密钥上下文
            weights: 模型权重字典
            scale: CKKS缩放因子
        """
        self.context = context
        self.weights = weights
        self.scale = scale or (2**40)

        self._load_weights()

    def _load_weights(self):
        """加载模型权重"""
        self.w1 = self.weights["fc1.weight"]
        self.b1 = self.weights["fc1.bias"]
        self.w2 = self.weights["fc2.weight"]
        self.b2 = self.weights["fc2.bias"]
        self.w3 = self.weights["fc3.weight"]
        self.b3 = self.weights["fc3.bias"]

    def _plain_relu(self, x: np.ndarray) -> np.ndarray:
        """
        明文ReLU（简化版：仅用于权重预热后的近似）

        注意：密文推理中ReLU是难点，这里采用方案：
        方案1：在解密后计算ReLU（推荐用于演示）
        方案2：使用密文多项式近似
        """
        return np.maximum(x, 0)

    def _encrypted_relu_approximate(
        self, encrypted_x: ts.CKKSVector, degree: int = 3
    ) -> ts.CKKSVector:
        """
        使用多项式近似密文ReLU

        ReLU(x) ≈ 0.5 * x + 0.5 * |x|
        或使用切比雪夫多项式逼近

        这里使用简化近似：ReLU(x) ≈ (x + |x|) / 2
        """
        encrypted_abs = encrypted_x * encrypted_x
        encrypted_abs = encrypted_abs.sqrt()

        return (encrypted_x + encrypted_abs) * 0.5

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        """
        明文前向传播（用于对比）

        Args:
            x: 输入向量

        Returns:
            预测结果（ logits）
        """
        x = x.reshape(-1)

        x1 = np.dot(x, self.w1.T) + self.b1
        x1 = self._plain_relu(x1)

        x2 = np.dot(x1, self.w2.T) + self.b2
        x2 = self._plain_relu(x2)

        x3 = np.dot(x2, self.w3.T) + self.b3

        return x3

    def forward_encrypted(self, encrypted_x: ts.CKKSVector) -> ts.CKKSVector:
        """
        密文前向传播

        Args:
            encrypted_x: 加密的输入向量

        Returns:
            加密的输出向量
        """
        encrypted_x1 = encrypted_x.mm(self.w1.T) + self.b1
        encrypted_x1 = self._encrypted_relu_approximate(encrypted_x1)

        encrypted_x2 = encrypted_x1.mm(self.w2.T) + self.b2
        encrypted_x2 = self._encrypted_relu_approximate(encrypted_x2)

        encrypted_x3 = encrypted_x2.mm(self.w3.T) + self.b3

        return encrypted_x3

    def predict_plain(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        明文预测

        Args:
            x: 输入向量

        Returns:
            (预测类别, 概率分布)
        """
        logits = self.forward_plain(x)
        probs = self._softmax(logits)
        prediction = np.argmax(probs)
        return prediction, probs

    def predict_encrypted(self, encrypted_x: ts.CKKSVector) -> Tuple[int, np.ndarray]:
        """
        密文预测

        Args:
            encrypted_x: 加密的输入向量

        Returns:
            (预测类别, 概率分布)
        """
        encrypted_logits = self.forward_encrypted(encrypted_x)
        logits = encrypted_logits.decrypt()

        probs = self._softmax(logits)
        prediction = np.argmax(probs)
        return prediction, probs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class BatchCiphertextInference:
    """批量密文推理"""

    def __init__(self, context: ts.Context, weights: dict):
        """
        初始化批量推理引擎

        Args:
            context: TenSEAL密钥上下文
            weights: 模型权重字典
        """
        self.context = context
        self.weights = weights
        self._load_weights()

    def _load_weights(self):
        """加载权重"""
        self.w1 = self.weights["fc1.weight"]
        self.b1 = self.weights["fc1.bias"]
        self.w2 = self.weights["fc2.weight"]
        self.b2 = self.weights["fc2.bias"]
        self.w3 = self.weights["fc3.weight"]
        self.b3 = self.weights["fc3.bias"]

    def forward_batch(
        self, encrypted_batch: List[ts.CKKSVector]
    ) -> List[ts.CKKSVector]:
        """
        批量密文前向传播

        Args:
            encrypted_batch: 加密的输入向量列表

        Returns:
            加密的输出向量列表
        """
        results = []

        for encrypted_x in encrypted_batch:
            encrypted_x1 = encrypted_x.mm(self.w1.T) + self.b1
            encrypted_x1 = self._relu_approx(encrypted_x1)

            encrypted_x2 = encrypted_x1.mm(self.w2.T) + self.b2
            encrypted_x2 = self._relu_approx(encrypted_x2)

            encrypted_x3 = encrypted_x2.mm(self.w3.T) + self.b3
            results.append(encrypted_x3)

        return results

    def _relu_approx(self, x: ts.CKKSVector) -> ts.CKKSVector:
        """ReLU近似"""
        return (x + x.square().sqrt()) * 0.5

    def decrypt_batch(self, encrypted_results: List[ts.CKKSVector]) -> np.ndarray:
        """批量解密"""
        return np.array([result.decrypt() for result in encrypted_results])


def create_inference_engine(
    context_path: str = "./keys/context.bin",
    weights_path: str = "./models/mnist_net.pth",
):
    """
    便捷函数：创建推理引擎

    Args:
        context_path: 密钥上下文路径
        weights_path: 模型权重路径

    Returns:
        CiphertextInference实例
    """
    from keygen import KeyGenerator
    import torch

    context = KeyGenerator.load_context(context_path)

    model = torch.load(weights_path, map_location="cpu")
    if isinstance(model, dict):
        weights = model
    else:
        weights = model.state_dict()

    return CiphertextInference(context, weights)


def compare_plain_encrypted(
    context: ts.Context, weights: dict, test_data: np.ndarray, test_label: int = None
):
    """
    对比明文和密文推理结果

    Args:
        context: TenSEAL上下文
        weights: 模型权重
        test_data: 测试数据
        test_label: 真实标签
    """
    inference = CiphertextInference(context, weights)

    plain_pred, plain_probs = inference.predict_plain(test_data)

    encrypted_input = ts.ckks_vector(context, test_data.tolist())
    enc_pred, enc_probs = inference.predict_encrypted(encrypted_input)

    print(f"真实标签: {test_label}")
    print(f"明文推理: 预测={plain_pred}, 概率={plain_probs}")
    print(f"密文推理: 预测={enc_pred}, 概率={enc_probs}")
    print(f"预测一致: {plain_pred == enc_pred}")

    return plain_pred, enc_pred


if __name__ == "__main__":
    import sys

    sys.path.append("..")

    from keygen import KeyGenerator
    import torch

    keygen = KeyGenerator()
    keygen.generate()

    model = torch.load("../models/mnist_net.pth", map_location="cpu")
    if hasattr(model, "state_dict"):
        weights = model.state_dict()
    else:
        weights = model

    inference = CiphertextInference(keygen.context, weights)

    test_input = np.random.randn(784)
    plain_pred, plain_probs = inference.predict_plain(test_input)

    print(f"明文预测: {plain_pred}")
    print(f"概率分布: {plain_probs}")
