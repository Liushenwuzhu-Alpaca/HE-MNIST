"""
主程序 - 基于同态加密的手写数字识别
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(__file__))

from keygen import KeyGenerator, generate_keys
from model import train_model, MNISTNet, ModelTrainer
from ciphertext_inference import CiphertextInference, PlainInference, compare_inference
from encrypt import Encryptor, Encoder
import tenseal as ts
import numpy as np
import torch


def setup_environment():
    """设置环境，创建必要的目录"""
    directories = ["keys", "models", "data/mnist", "results"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ 环境设置完成")


def step1_generate_keys():
    """步骤1：生成同态加密密钥"""
    print("\n" + "=" * 50)
    print("步骤1: 生成TenSEAL CKKS密钥")
    print("=" * 50)

    os.makedirs("../keys", exist_ok=True)

    keygen = KeyGenerator(poly_modulus_degree=8192)
    keygen.generate()

    keygen.save_keys(
        public_key_path="../keys/public_key.bin",
        secret_key_path="../keys/secret_key.bin",
        context_path="../keys/context.bin",
        galois_keys_path="../keys/galois_keys.bin",
    )

    print("✓ 密钥生成完成")
    return keygen


def step2_train_model(epochs: int = 10):
    """步骤2：训练神经网络模型"""
    print("\n" + "=" * 50)
    print("步骤2: 训练MNIST神经网络模型")
    print("=" * 50)

    model = MNISTNet(input_size=784, hidden1_size=256, hidden2_size=128, output_size=10)
    trainer = ModelTrainer(model)
    trainer.load_mnist(data_dir="./data/mnist", batch_size=64, download=True)
    trainer.train(epochs=epochs, save_path="./models/mnist_net.pth")

    print("✓ 模型训练完成")
    return model


def step3_test_inference():
    """步骤3：测试密文推理"""
    print("\n" + "=" * 50)
    print("步骤3: 测试明文与密文推理一致性")
    print("=" * 50)

    context = KeyGenerator.load_context("../keys/context.bin")

    model = torch.load("../models/mnist_net.pth", map_location="cpu")
    if hasattr(model, "state_dict"):
        weights = model.state_dict()
    else:
        weights = model

    inference = CiphertextInference(context, weights)

    # 使用MNIST格式的测试数据（归一化到0-1）
    test_data = np.random.rand(784).astype(np.float64)

    plain_pred, plain_probs = inference.predict_plain(test_data)

    encrypted_input = ts.ckks_vector(context, test_data.tolist())
    enc_pred, enc_probs = inference.predict_encrypted(encrypted_input)

    print(f"\n明文预测: {plain_pred}")
    print(f"密文预测: {enc_pred}")
    print(f"明文概率: {plain_probs}")
    print(f"密文概率: {enc_probs}")
    print(f"\n预测一致: {plain_pred == enc_pred}")
    print(f"概率最大误差: {np.max(np.abs(plain_probs - enc_probs))}")

    print("\n✓ 密文推理测试完成")


def step4_demo():
    """步骤4：运行演示"""
    print("\n" + "=" * 50)
    print("步骤4: 运行Web演示")
    print("=" * 50)
    print("启动Flask服务器...")
    print("访问 http://localhost:5000")

    from app import app

    app.run(debug=False, host="0.0.0.0", port=5000)


def run_full_pipeline(epochs: int = 10):
    """运行完整流程"""
    print("=" * 60)
    print("  基于同态加密的手写数字识别 - 完整流程")
    print("=" * 60)

    setup_environment()

    keygen = step1_generate_keys()

    step2_train_model(epochs=epochs)

    step3_test_inference()

    print("\n" + "=" * 60)
    print("  所有步骤完成！")
    print("=" * 60)
    print("\n运行以下命令启动Web演示:")
    print("  python src/main.py --demo")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于同态加密的手写数字识别")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        help="运行指定步骤 (1:密钥生成, 2:模型训练, 3:测试推理, 4:Web演示)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--demo", action="store_true", help="启动Web演示")

    args = parser.parse_args()

    if args.demo:
        step4_demo()
        return

    if args.step:
        if args.step == 1:
            step1_generate_keys()
        elif args.step == 2:
            step2_train_model(epochs=args.epochs)
        elif args.step == 3:
            step3_test_inference()
        elif args.step == 4:
            step4_demo()
    else:
        run_full_pipeline(epochs=args.epochs)


if __name__ == "__main__":
    main()
