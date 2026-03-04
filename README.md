# 基于同态加密的手写数字识别

大二密码学大作业 - 使用TenSEAL CKKS同态加密实现保护隐私的MNIST数字识别

## 项目简介

本项目实现了一个**保护隐私的手写数字识别系统**：
- 用户将手写数字图片加密后上传
- 服务器在密文状态下完成神经网络推理
- 仅返回预测结果（0-9），服务器无法获知原始图片内容

## 技术栈

| 组件 | 技术 |
|------|------|
| 同态加密 | TenSEAL CKKS |
| 深度学习 | PyTorch |
| Web框架 | Flask |
| 数据集 | MNIST |

## 系统架构

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  客户端   │     │    服务器    │     │   客户端    │
│ 1.加密图片│────▶│ 2.密文推理   │────▶│ 4.解密结果  │
│  (CKKS)  │     │ (神经网络)   │     │             │
└──────────┘     └──────────────┘     └──────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整流程

```bash
cd src
python main.py
```

这将自动执行：
1. 生成TenSEAL CKKS密钥
2. 训练MNIST神经网络模型
3. 测试明文/密文推理一致性

### 3. 启动Web演示

```bash
python src/main.py --demo
```

访问 http://localhost:5000

## 项目结构

```
HE-MNIST/
├── src/
│   ├── __init__.py
│   ├── keygen.py              # 密钥生成
│   ├── encrypt.py             # 数据加密/解密
│   ├── model.py               # 神经网络模型 (square激活)
│   ├── ciphertext_inference.py # 密文推理
│   ├── app.py                 # Flask Web应用
│   └── main.py                # 主程序
├── templates/
│   └── index.html             # Web界面
├── models/                    # 训练好的模型
├── keys/                      # 密钥文件
├── data/                      # MNIST数据
└── requirements.txt           # 依赖列表
```

## 核心实现

### 同态加密配置

- **方案**: CKKS (Cheon-Kim-Kim-Song)
- **多项式度数**: 8192
- **系数模数**: [40, 21, 21, 21, 21, 21, 21, 40]
- **全局缩放因子**: 2^21

### 神经网络结构

- 输入层: 784维 (28×28 MNIST图像)
- 隐藏层1: 256神经元 + **平方激活(x²)**
- 隐藏层2: 128神经元 + **平方激活(x²)**
- 输出层: 10神经元 (数字0-9)

### 密文推理流程

1. 加密输入图像 → CKKS密文向量
2. 密文矩阵乘法 + 密文加法 → 密文向量
3. 密文平方激活(x²) → 密文向量
4. 重复至输出层
5. 解密得到预测结果

**注意**: 由于CKKS同态加密的特性，密文推理使用**平方函数(x²)**替代ReLU激活函数，这是encrypted-evaluation项目的标准做法。

## 使用说明

### 命令行操作

```bash
# 仅生成密钥
python src/main.py --step 1

# 仅训练模型
python src/main.py --step 2 --epochs 10

# 仅测试推理
python src/main.py --step 3

# 启动Web演示
python src/main.py --step 4
# 或
python src/main.py --demo
```

## 注意事项

1. **密钥安全**: 私钥文件(`keys/secret_key.bin`)应妥善保管，不要上传到服务器
2. **性能**: 密文推理比明文推理慢约100-1000倍，这是正常现象
3. **精度**: CKKS同态加密有约10^-5的精度损失
4. **激活函数**: 使用平方函数(x²)替代ReLU，这是同态加密中的常见做法

## 参考资料

- [TenSEAL GitHub](https://github.com/OpenMined/TenSEAL)
- [encrypted-evaluation项目](https://github.com/youben11/encrypted-evaluation)
- [CKKS原理](https://eprint.iacr.org/2019/016)
- [PyTorch MNIST教程](https://pytorch.org/tutorials/)

## 许可证

MIT License
