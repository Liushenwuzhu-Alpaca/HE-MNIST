"""
密文推理模块 - 使用square激活函数
"""

from typing import Tuple

import numpy as np
import tenseal as ts


class CiphertextInference:
    """密文推理引擎"""

    MEAN = 0.1307
    STD = 0.3081
    """
    在加密之前输入应该被正规化为平均值为0.1307与方差为0.3081
    由MNIST训练数据计算得来
    """

    def __init__(self, context: ts.Context, weights: dict):
        """
        初始化密文推理引擎

        Args:
            context (ts.Context): Tenseal上下文
            weights (dict): 模型权重
        """
        self.context = context
        self.w1 = weights["fc1.weight"].numpy()
        self.b1 = weights["fc1.bias"].numpy()
        self.w2 = weights["fc2.weight"].numpy()
        self.b2 = weights["fc2.bias"].numpy()
        self.w3 = weights["fc3.weight"].numpy()
        self.b3 = weights["fc3.bias"].numpy()

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        """
        明文前向传播

        Args:
            x (np.ndarray): 输入数据

        Returns:
            np.ndarray: 输出数据
        """
        x = (x.reshape(-1).astype(np.float64) - self.MEAN) / self.STD
        x1 = np.matmul(x, self.w1.T) + self.b1
        x2 = np.matmul(x1**2, self.w2.T) + self.b2
        x3 = np.matmul(x2**2, self.w3.T) + self.b3
        return x3

    def forward_encrypted(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
        """
        密文前向传播

        Args:
            enc_x (ts.CKKSVector): 输入密文

        Returns:
            ts.CKKSVector: 输出密文
        """
        enc_x.mm_(self.w1.T)
        enc_x = enc_x + ts.ckks_vector(self.context, self.b1.tolist())
        enc_x.square_()
        enc_x.mm_(self.w2.T)
        enc_x = enc_x + ts.ckks_vector(self.context, self.b2.tolist())
        enc_x.square_()
        enc_x.mm_(self.w3.T)
        enc_x = enc_x + ts.ckks_vector(self.context, self.b3.tolist())
        return enc_x

    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        明文预测
        Args:
            x (np.ndarray): 输入数据
        Returns:
            Tuple[int, np.ndarray]: 预测结果和概率分布
        """
        logits = self.forward_plain(x)
        probs = self._softmax(logits)
        return int(np.argmax(probs)), probs

    def predict_encrypted(self, enc_x: ts.CKKSVector) -> Tuple[int, np.ndarray]:
        """
        密文预测-需要重新解密
        Args:
            enc_x (ts.CKKSVector): 输入密文
        Returns:
            Tuple[int, np.ndarray]: 预测结果和概率分布
        """
        logits = np.array(self.forward_encrypted(enc_x).decrypt())
        probs = self._softmax(logits)
        return int(np.argmax(probs)), probs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """
        softmax函数，用于将向量转换为概率分布
        Args:
            x (np.ndarray): 输入向量
        Returns:
            np.ndarray: 概率分布
        """
        x = np.array(x).flatten()
        return np.exp(x - x.max()) / np.exp(x - x.max()).sum()
