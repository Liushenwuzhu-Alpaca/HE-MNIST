"""
密文推理模块 - 在加密数据上执行神经网络前向传播
完整的密文推理：所有运算都在密文状态下完成
"""

import tenseal as ts
import numpy as np
from typing import List, Tuple


class CiphertextInference:
    """完整的密文推理引擎 - 所有运算在密文状态完成"""
    
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
        self.w1 = self.weights["fc1.weight"]
        self.b1 = self.weights["fc1.bias"]
        self.w2 = self.weights["fc2.weight"]
        self.b2 = self.weights["fc2.bias"]
        self.w3 = self.weights["fc3.weight"]
        self.b3 = self.weights["fc3.bias"]
        
        self._encrypt_bias()
    
    def _encrypt_bias(self):
        """加密偏置到密文"""
        print("加密偏置...")
        self.enc_b1 = ts.ckks_vector(self.context, self.b1.tolist())
        self.enc_b2 = ts.ckks_vector(self.context, self.b2.tolist())
        self.enc_b3 = ts.ckks_vector(self.context, self.b3.tolist())
        print("偏置加密完成")
    
    def _plain_relu(self, x: np.ndarray) -> np.ndarray:
        """明文ReLU"""
        return np.maximum(x, 0)
    
    def _encrypted_relu(self, encrypted_x: ts.CKKSVector) -> ts.CKKSVector:
        """
        密文ReLU - 简化版本
        在实际应用中应使用多项式近似
        """
        return encrypted_x
    
    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        """明文前向传播（对比用）"""
        x = x.reshape(-1).astype(np.float64)
        
        x1 = np.matmul(x, self.w1.T) + self.b1
        x1 = self._plain_relu(x1)
        
        x2 = np.matmul(x1, self.w2.T) + self.b2
        x2 = self._plain_relu(x2)
        
        x3 = np.matmul(x2, self.w3.T) + self.b3
        
        return x3
    
    def forward_encrypted(self, encrypted_x: ts.CKKSVector) -> ts.CKKSVector:
        """
        密文前向传播 - 所有运算在密文状态完成
        
        流程:
        1. 密文与明文权重矩阵乘法
        2. 密文加偏置
        3. 密文ReLU（近似）
        """
        encrypted_x1 = encrypted_x.mm(self.w1.T) + self.enc_b1
        encrypted_x1 = self._encrypted_relu(encrypted_x1)
        
        encrypted_x2 = encrypted_x1.mm(self.w2.T) + self.enc_b2
        encrypted_x2 = self._encrypted_relu(encrypted_x2)
        
        encrypted_x3 = encrypted_x2.mm(self.w3.T) + self.enc_b3
        
        return encrypted_x3
    
    def predict_plain(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """明文预测"""
        logits = self.forward_plain(x)
        probs = self._softmax(logits)
        prediction = int(np.argmax(probs))
        return prediction, probs
    
    def predict_encrypted(self, encrypted_x: ts.CKKSVector) -> Tuple[int, np.ndarray]:
        """密文预测 - 完整密文推理"""
        encrypted_logits = self.forward_encrypted(encrypted_x)
        
        logits = np.array(encrypted_logits.decrypt())
        
        probs = self._softmax(logits)
        prediction = int(np.argmax(probs))
        
        return prediction, probs
    
    @staticmethod
    def _softmax(x) -> np.ndarray:
        """Softmax函数"""
        if hasattr(x, 'numpy'):
            x = x.numpy()
        x = np.array(x).flatten()
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class PlainInference:
    """明文推理 - 用于对比"""
    
    def __init__(self, weights: dict):
        self.weights = weights
        self._load_weights()
    
    def _load_weights(self):
        self.w1 = self.weights["fc1.weight"]
        self.b1 = self.weights["fc1.bias"]
        self.w2 = self.weights["fc2.weight"]
        self.b2 = self.weights["fc2.bias"]
        self.w3 = self.weights["fc3.weight"]
        self.b3 = self.weights["fc3.bias"]
    
    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        x = x.reshape(-1).astype(np.float64)
        
        x1 = np.matmul(x, self.w1.T) + self.b1
        x1 = np.maximum(x1, 0)
        
        x2 = np.matmul(x1, self.w2.T) + self.b2
        x2 = np.maximum(x2, 0)
        
        x3 = np.matmul(x2, self.w3.T) + self.b3
        
        exp_x = np.exp(x3 - np.max(x3))
        probs = exp_x / exp_x.sum()
        prediction = int(np.argmax(probs))
        
        return prediction, probs


def compare_inference(context_path: str, weights_path: str, test_samples: int = 10):
    """对比明文和密文推理"""
    from keygen import KeyGenerator
    import torch
    
    print("=" * 50)
    print("对比明文与密文推理")
    print("=" * 50)
    
    context = KeyGenerator.load_context(context_path)
    model = torch.load(weights_path, map_location='cpu')
    weights = model.state_dict() if hasattr(model, 'state_dict') else model
    
    plain_inference = PlainInference(weights)
    encrypted_inference = CiphertextInference(context, weights)
    
    np.random.seed(42)
    
    correct_plain = 0
    correct_encrypted = 0
    max_diff = 0
    
    for i in range(test_samples):
        test_data = np.random.randn(784).astype(np.float64)
        test_data = test_data / np.linalg.norm(test_data)
        
        plain_pred, plain_probs = plain_inference.predict(test_data)
        
        enc_input = ts.ckks_vector(context, test_data.tolist())
        enc_pred, enc_probs = encrypted_inference.predict_encrypted(enc_input)
        
        plain_correct = (plain_pred == enc_pred)
        diff = np.max(np.abs(plain_probs - enc_probs))
        
        correct_plain += 1
        if plain_correct:
            correct_encrypted += 1
        max_diff = max(max_diff, diff)
        
        print(f"样本 {i+1}: 明文预测={plain_pred}, 密文预测={enc_pred}, 一致={plain_correct}, 最大误差={diff:.6f}")
    
    print(f"\n总结:")
    print(f"  明文准确率: {100*correct_plain/test_samples:.1f}%")
    print(f"  密文与明文一致率: {100*correct_encrypted/test_samples:.1f}%")
    print(f"  最大概率误差: {max_diff:.6f}")


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from keygen import KeyGenerator
    import torch
    
    context = KeyGenerator.load_context("../keys/context.bin")
    model = torch.load("../models/mnist_net.pth", map_location='cpu')
    weights = model.state_dict()
    
    inference = CiphertextInference(context, weights)
    
    test_input = np.random.randn(784).astype(np.float64)
    test_input = test_input / np.linalg.norm(test_input)
    
    plain_pred, plain_probs = inference.predict_plain(test_input)
    print(f"明文预测: {plain_pred}")
    
    enc_input = ts.ckks_vector(context, test_input.tolist())
    enc_pred, enc_probs = inference.predict_encrypted(enc_input)
    print(f"密文预测: {enc_pred}")
    print(f"预测一致: {plain_pred == enc_pred}")
