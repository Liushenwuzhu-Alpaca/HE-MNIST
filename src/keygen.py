"""
密钥生成模块 - TenSEAL CKKS密钥上下文生成
"""

import tenseal as ts
import os


class KeyGenerator:
    """TenSEAL CKKS密钥生成器"""

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list = None,
        global_scale: int = 2**40,
    ):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.global_scale = global_scale
        self.context = None

    def generate(self):
        """生成CKKS密钥上下文和密钥对"""
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )

        self.context.global_scale = self.global_scale
        self.context.auto_rescale = True
        self.context.generate_galois_keys()

        return self.context

    def get_context_bytes(self) -> bytes:
        """获取密钥上下文字节（可序列化）"""
        if self.context is None:
            raise ValueError("密钥上下文未生成")
        return self.context.serialize()

    def save_keys(
        self,
        public_key_path: str,
        secret_key_path: str,
        context_path: str = None,
        galois_keys_path: str = None,
    ):
        """保存密钥到文件 - 保存整个context"""
        if context_path:
            with open(context_path, "wb") as f:
                f.write(self.get_context_bytes())
            print(f"密钥上下文已保存: {context_path}")

        # 创建不含私钥的公开context
        public_context = self.context.copy()
        public_context.make_context_public()

        with open(public_key_path, "wb") as f:
            f.write(public_context.serialize())
        print(f"公钥已保存: {public_key_path}")

        # 保存含私钥的完整context
        with open(secret_key_path, "wb") as f:
            f.write(self.get_context_bytes())
        print(f"私钥已保存: {secret_key_path}")

    @staticmethod
    def load_context(context_path: str) -> ts.Context:
        """从文件加载密钥上下文（含私钥）"""
        with open(context_path, "rb") as f:
            return ts.context_from(f.read())

    @staticmethod
    def load_public_context(public_key_path: str) -> ts.Context:
        """从文件加载公钥上下文（不含私钥）"""
        with open(public_key_path, "rb") as f:
            return ts.context_from(f.read())


def generate_keys(key_size: int = 8192, save_dir: str = "keys"):
    """便捷函数：生成并保存密钥"""
    os.makedirs(save_dir, exist_ok=True)

    keygen = KeyGenerator(poly_modulus_degree=key_size)
    keygen.generate()

    keygen.save_keys(
        public_key_path=os.path.join(save_dir, "public_key.bin"),
        secret_key_path=os.path.join(save_dir, "secret_key.bin"),
        context_path=os.path.join(save_dir, "context.bin"),
    )

    return keygen


if __name__ == "__main__":
    generate_keys()
