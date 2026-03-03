"""
密钥生成模块 - TenSEAL CKKS密钥上下文生成
"""

import tenseal as ts
import json
import os


class KeyGenerator:
    """TenSEAL CKKS密钥生成器"""

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list = None,
        global_scale: int = 2**40,
    ):
        """
        初始化密钥生成器

        Args:
            poly_modulus_degree: 多项式模数次数 (影响安全性)
            coeff_mod_bit_sizes: 系数模数位长列表
            global_scale: 全局缩放因子
        """
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.global_scale = global_scale
        self.context = None
        self.public_key = None
        self.secret_key = None
        self.galois_keys = None

    def generate(self):
        """生成CKKS密钥上下文和密钥对"""
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            global_scale=self.global_scale,
        )

        self.context.generate_galois_keys()
        self.public_key = self.context.public_key()
        self.secret_key = self.context.secret_key()
        self.galois_keys = self.context.galois_keys()

        return self.context

    def get_context_bytes(self) -> bytes:
        """获取密钥上下文字节（可序列化）"""
        if self.context is None:
            raise ValueError("密钥上下文未生成")
        return self.context.serialize()

    def get_public_key_bytes(self) -> bytes:
        """获取公钥字节"""
        if self.public_key is None:
            raise ValueError("公钥未生成")
        return self.public_key.serialize()

    def get_secret_key_bytes(self) -> bytes:
        """获取私钥字节"""
        if self.secret_key is None:
            raise ValueError("私钥未生成")
        return self.secret_key.serialize()

    def get_galois_keys_bytes(self) -> bytes:
        """获取Galois密钥字节"""
        if self.galois_keys is None:
            raise ValueError("Galois密钥未生成")
        return self.galois_keys.serialize()

    def save_keys(
        self,
        public_key_path: str,
        secret_key_path: str,
        context_path: str = None,
        galois_keys_path: str = None,
    ):
        """保存密钥到文件"""
        with open(public_key_path, "wb") as f:
            f.write(self.get_public_key_bytes())

        with open(secret_key_path, "wb") as f:
            f.write(self.get_secret_key_bytes())

        if context_path:
            with open(context_path, "wb") as f:
                f.write(self.get_context_bytes())

        if galois_keys_path:
            with open(galois_keys_path, "wb") as f:
                f.write(self.get_galois_keys_bytes())

        print(f"密钥已保存:")
        print(f"  公钥: {public_key_path}")
        print(f"  私钥: {secret_key_path}")
        if context_path:
            print(f"  上下文: {context_path}")
        if galois_keys_path:
            print(f"  Galois密钥: {galois_keys_path}")

    @staticmethod
    def load_context(context_path: str) -> ts.Context:
        """从文件加载密钥上下文"""
        with open(context_path, "rb") as f:
            return ts.context_from(f.read())

    @staticmethod
    def load_public_key(public_key_path: str) -> bytes:
        """从文件加载公钥"""
        with open(public_key_path, "rb") as f:
            return f.read()

    @staticmethod
    def load_secret_key(secret_key_path: str) -> bytes:
        """从文件加载私钥"""
        with open(secret_key_path, "rb") as f:
            return f.read()


def generate_keys(key_size: int = 8192, save_dir: str = "keys"):
    """
    便捷函数：生成并保存密钥

    Args:
        key_size: 密钥大小
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    keygen = KeyGenerator(poly_modulus_degree=key_size)
    keygen.generate()

    keygen.save_keys(
        public_key_path=os.path.join(save_dir, "public_key.bin"),
        secret_key_path=os.path.join(save_dir, "secret_key.bin"),
        context_path=os.path.join(save_dir, "context.bin"),
        galois_keys_path=os.path.join(save_dir, "galois_keys.bin"),
    )

    return keygen


if __name__ == "__main__":
    generate_keys()
