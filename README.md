# 基于同态加密的手写数字识别

大二密码学大作业 - 使用TenSEAL CKKS同态加密实现保护隐私的MNIST数字识别

## 项目简介

本项目实现了一个保护隐私的手写数字识别系统：
- 用户将手写数字图片加密后上传
- 服务器在密文状态下完成神经网络推理
- 仅返回预测结果（0-9），服务器无法获知原始图片内容

## 技术栈

| 组件 | 技术 |
|------|------|
| 同态加密 | TenSEAL (CKKS方案) |
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

### 1. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行完整流程

```bash
cd src
python main.py
```

这将自动执行：
1. 生成TenSEAL CKKS密钥
2. 训练MNIST神经网络模型
3. 测试明文/密文推理一致性

### 4. 启动Web演示

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
│   ├── model.py               # 神经网络模型
│   ├── ciphertext_inference.py # 密文推理
│   ├── app.py                 # Flask Web应用
│   └── main.py                # 主程序
├── templates/
│   └── index.html             # Web界面
├── models/                    # 训练好的模型
├── keys/                      # 密钥文件
├── data/                      # MNIST数据
├── requirements.txt           # 依赖列表
└── PROJECT_PLAN.md           # 项目计划
```

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

## 神经网络结构

- 输入层: 784维 (28×28 MNIST图像)
- 隐藏层1: 256神经元 + ReLU + Dropout(0.2)
- 隐藏层2: 128神经元 + ReLU + Dropout(0.2)
- 输出层: 10神经元 (数字0-9)

## 注意事项

1. **密钥安全**: 私钥文件(`keys/secret_key.bin`)应妥善保管，不要上传到服务器
2. **性能**: 密文推理比明文推理慢约100-1000倍，这是正常现象
3. **精度**: CKKS同态加密有约10^-5的精度损失

## 参考资料

- [TenSEAL文档](https://microsoft.github.io/SEAL/)
- [CKKS原理](https://eprint.iacr.org/2019/016)
- [PyTorch MNIST教程](https://pytorch.org/tutorials/)

## 许可证

MIT License
