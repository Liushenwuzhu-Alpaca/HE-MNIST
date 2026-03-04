"""
数据加密模块 - 基于TenSEAL CKKS的数据加密和解密
"""

from typing import List, Union

import numpy as np
import tenseal as ts


class Encryptor:
    """CKKS加密器"""

    def __init__(self, context: ts.Context):
        """
        初始化加密器

        Args:
            context: TenSEAL密钥上下文
        """
        self.context = context

    def encrypt_vector(self, data: Union[List[float], np.ndarray]) -> ts.CKKSVector:
        """
        加密向量

        Args:
            data: 待加密的浮点数向量

        Returns:
            加密的CKKS向量
        """
        if isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        elif not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)

        return ts.ckks_vector(self.context, data.tolist())

    def encrypt_matrix(
        self, matrix: Union[List[List[float]], np.ndarray]
    ) -> List[ts.CKKSVector]:
        """
        加密矩阵（按行加密）

        Args:
            matrix: 待加密的二维数组

        Returns:
            加密的CKKS向量列表
        """
        if isinstance(matrix, list):
            matrix = np.array(matrix, dtype=np.float64)

        return [self.encrypt_vector(row) for row in matrix]

    def decrypt_vector(self, encrypted_vector: ts.CKKSVector) -> np.ndarray:
        """
        解密向量

        Args:
            encrypted_vector: 加密的CKKS向量

        Returns:
            解密后的numpy数组
        """
        return np.array(encrypted_vector.decrypt())

    def decrypt_matrix(self, encrypted_matrix: List[ts.CKKSVector]) -> np.ndarray:
        """
        解密矩阵

        Args:
            encrypted_matrix: 加密的CKKS向量列表

        Returns:
            解密后的numpy二维数组
        """
        return np.array([self.decrypt_vector(vec) for vec in encrypted_matrix])


class Encoder:
    """数据编码器 - 将数据编码为适合CKKS加密的格式"""

    @staticmethod
    def normalize(
        data: np.ndarray, target_min: float = -1.0, target_max: float = 1.0
    ) -> np.ndarray:
        """
        将数据归一化到目标范围

        Args:
            data: 输入数据
            target_min: 目标最小值
            target_max: 目标最大值

        Returns:
            归一化后的数据
        }
        """
        data_min = data.min()
        data_max = data.max()

        if data_max == data_min:
            return np.zeros_like(data)

        normalized = (data - data_min) / (data_max - data_min)
        return normalized * (target_max - target_min) + target_min

    @staticmethod
    def encode_image(image: np.ndarray, flatten: bool = True) -> np.ndarray:
        """
        编码MNIST图像

        Args:
            image: MNIST图像 (28x28或784)
            flatten: 是否展平为一维

        Returns:
            编码后的向量
        """
        if flatten and image.ndim == 2:
            image = image.flatten()

        return Encoder.normalize(image.astype(np.float64))

    @staticmethod
    def encode_batch(images: np.ndarray) -> np.ndarray:
        """
        批量编码图像

        Args:
            images: 图像批量 (N, 28, 28) 或 (N, 784)

        Returns:
            编码后的矩阵
        """
        if images.ndim == 3:
            images = images.reshape(images.shape[0], -1)

        encoded = np.array([Encoder.encode_image(img, flatten=False) for img in images])
        return encoded.astype(np.float64)


def encrypt_weights(weights: np.ndarray, context: ts.Context) -> List[ts.CKKSVector]:
    """
    便捷函数：加密神经网络权重

    Args:
        weights: 权重矩阵
        context: TenSEAL上下文

    Returns:
        加密的权重列表
    """
    encoder = Encoder()
    encryptor = Encryptor(context)

    if weights.ndim == 2:
        return encryptor.encrypt_matrix(weights)
    else:
        return [encryptor.encrypt_vector(weights)]


def decrypt_weights(encrypted_weights: List[ts.CKKSVector]) -> np.ndarray:
    """
    便捷函数：解密神经网络权重

    Args:
        encrypted_weights: 加密的权重列表

    Returns:
        解密后的权重矩阵
    """
    if len(encrypted_weights) == 1:
        return encrypted_weights[0].decrypt()

    encryptor = Encryptor(encrypted_weights[0].context)
    return encryptor.decrypt_matrix(encrypted_weights)


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from keygen import KeyGenerator

    keygen = KeyGenerator()
    keygen.generate()

    encryptor = Encryptor(keygen.context)
    encoder = Encoder()

    test_data = np.random.randn(10)
    print(f"原始数据: {test_data}")

    encrypted = encryptor.encrypt_vector(test_data)
    print(f"加密后类型: {type(encrypted)}")

    decrypted = encryptor.decrypt_vector(encrypted)
    print(f"解密后数据: {decrypted}")
    print(f"误差: {np.abs(test_data - decrypted).max()}")
