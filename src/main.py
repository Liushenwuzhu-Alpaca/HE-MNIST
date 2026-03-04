"""
主程序 - 基于同态加密的手写数字识别
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np
import tenseal as ts
import torch

from ciphertext_inference import CiphertextInference
from encrypt import Encoder, Encryptor
from keygen import KeyGenerator, generate_keys
from model import MNISTNet, Trainer


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

    os.makedirs("keys", exist_ok=True)

    keygen = KeyGenerator(poly_modulus_degree=8192)
    keygen.generate()

    keygen.save_keys(
        public_key_path="keys/public_key.bin",
        secret_key_path="keys/secret_key.bin",
        context_path="keys/context.bin",
        galois_keys_path="keys/galois_keys.bin",
    )

    print("✓ 密钥生成完成")
    return keygen


def step2_train_model(epochs: int = 10):
    print("\n" + "=" * 50)
    print("步骤2: 训练MNIST神经网络模型")
    print("=" * 50)
    model = MNISTNet()
    trainer = Trainer(model)
    trainer.load_data(data_dir="./data/mnist", batch_size=64)
    trainer.train(epochs=epochs, save_path="./models/mnist_net.pth")
    print("✓ 模型训练完成")
    return model


def step3_test_inference():
    """步骤3：测试密文推理"""
    print("\n" + "=" * 50)
    print("步骤3: 测试明文与密文推理一致性")
    print("=" * 50)

    context = KeyGenerator.load_context("keys/context.bin")

    model = torch.load("models/mnist_net.pth", map_location="cpu")
    if hasattr(model, "state_dict"):
        weights = model.state_dict()
    else:
        weights = model

    inference = CiphertextInference(context, weights)

    # 使用MNIST测试数据
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root="./data/mnist", train=False, download=True, transform=transform
    )

    correct_plain = 0
    correct_enc = 0
    total = 10

    for i in range(total):
        img, label = test_data[i]
        img_np = img.numpy().flatten().astype(np.float64)

        plain_pred, plain_probs = inference.predict(img_np)
        encrypted_input = ts.ckks_vector(context, img_np.tolist())
        enc_pred, enc_probs = inference.predict_encrypted(encrypted_input)

        if plain_pred == label:
            correct_plain += 1
        if enc_pred == label:
            correct_enc += 1

        print(f"样本{i}: 真实={label}, 明文={plain_pred}, 密文={enc_pred}")

    print(f"\n明文准确率: {correct_plain}/{total}")
    print(f"密文准确率: {correct_enc}/{total}")
    print("\n✓ 密文推理测试完成")


def step4_demo():
    """步骤4：运行演示"""
    print("\n" + "=" * 50)
    print("步骤4: 运行Web演示")
    print("=" * 50)
    print("启动Flask服务器...")
    print("访问 http://localhost:5000")

    from app import app

    app.run(debug=False, host="0.0.0.0", port=5001)


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
