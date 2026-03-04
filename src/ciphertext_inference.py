"""
密文推理模块 - 使用square激活函数
"""

import tenseal as ts
import numpy as np
from typing import Tuple


class CiphertextInference:
    """密文推理引擎"""

    MEAN = 0.1307
    STD = 0.3081

    def __init__(self, context: ts.Context, weights: dict):
        self.context = context
        self.w1 = weights["fc1.weight"].numpy()
        self.b1 = weights["fc1.bias"].numpy()
        self.w2 = weights["fc2.weight"].numpy()
        self.b2 = weights["fc2.bias"].numpy()
        self.w3 = weights["fc3.weight"].numpy()
        self.b3 = weights["fc3.bias"].numpy()

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        x = ((x.reshape(-1).astype(np.float64) - self.MEAN) / self.STD)
        x1 = np.matmul(x, self.w1.T) + self.b1
        x2 = np.matmul(x1**2, self.w2.T) + self.b2
        x3 = np.matmul(x2**2, self.w3.T) + self.b3
        return x3

    def forward_encrypted(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
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
        logits = self.forward_plain(x)
        probs = self._softmax(logits)
        return int(np.argmax(probs)), probs

    def predict_encrypted(self, enc_x: ts.CKKSVector) -> Tuple[int, np.ndarray]:
        logits = np.array(self.forward_encrypted(enc_x).decrypt())
        probs = self._softmax(logits)
        return int(np.argmax(probs)), probs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.array(x).flatten()
        return np.exp(x - x.max()) / np.exp(x - x.max()).sum()
