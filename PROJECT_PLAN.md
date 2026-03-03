# 基于同态加密的手写数字识别 - 大作业项目计划

## 一、项目概述

实现一个保护隐私的手写数字识别系统：用户将手写数字图片加密后上传，服务器在密文状态下完成神经网络推理，仅返回预测结果（0-9）。

## 二、技术栈

| 组件 | 选择 | 安装命令 |
|------|------|----------|
| 同态加密库 | **TenSEAL** | `pip install tenseal` |
| 深度学习框架 | **PyTorch** | `pip install torch torchvision` |
| Web框架 | Flask | `pip install flask` |
| 同态方案 | CKKS | TenSEAL内置 |
| 数据集 | MNIST | torchvision自带 |

## 三、系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    系统架构图                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │ 客户端   │     │    服务器    │     │   客户端    │   │
│   │          │     │              │     │              │   │
│   │ 1.加密图片│────▶│ 2.密文推理   │────▶│ 4.解密结果  │   │
│   │ (CKKS)   │     │ (神经网络)   │     │             │   │
│   └──────────┘     └──────────────┘     └──────────────┘   │
│                                                             │
│   密钥: 服务器生成 → 公钥发给客户端 → 私钥保留在客户端     │
└─────────────────────────────────────────────────────────────┘
```

## 四、项目结构

```
HE-MNIST/
├── src/
│   ├── __init__.py
│   ├── keygen.py              # TenSEAL密钥上下文生成
│   ├── encrypt.py             # 数据加密/解密
│   ├── model.py               # PyTorch神经网络定义与训练
│   ├── ciphertext_inference.py # 密文推理
│   ├── app.py                 # Flask Web界面
│   └── main.py                # 主程序
├── data/
│   └── mnist/                 # MNIST数据集
├── models/                    # 训练好的模型权重
├── templates/
│   └── index.html             # Web界面模板
├── report/
│   └── 大作业报告.md           # 课程报告(5000-8000字)
├── requirements.txt
└── README.md
```

## 五、神经网络设计

### 架构：简单全连接网络 (Fully Connected Network)

```python
# MNIST输入: 28x28 = 784维向量
# 网络结构:
# Input(784) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(10) → Output
```

| 层 | 输出维度 | 参数量 |
|---|---------|-------|
| Input | 784 | 0 |
| Dense1 | 256 | 784×256 + 256 = 200,960 |
| ReLU | 256 | 0 |
| Dense2 | 128 | 256×128 + 128 = 32,896 |
| ReLU | 128 | 0 |
| Dense3 | 10 | 128×10 + 10 = 1,290 |
| **总计** | | **~235K** |

## 六、密文推理原理

### 6.1 CKKS方案特点

- **加法同态**：`Enc(a) + Enc(b) = Enc(a + b)`
- **乘法同态**：`Enc(a) × Enc(b) = Enc(a × b)`
- **密文-明文乘法**：`Enc(a) × b = Enc(a × b)`
- 有精度损失（约10^-5），适合浮点数运算

### 6.2 密文推理流程

```
明文:     x          → 加密 → Enc(x)
权重:     W1, W2    → 明文   (无需加密，服务端已有)
偏置:     b1, b2   → 明文

密文推理:
Enc(y) = Enc(x) @ W1 + b1
       → ReLU(Enc(y))  (使用近似多项式)
       → Enc(z) = Enc(y1) @ W2 + b2
       → 解密后 softmax → 预测结果
```

## 七、实施计划

### 第1周：环境搭建

- [ ] 安装TenSEAL、PyTorch、Flask
- [ ] 下载MNIST数据集
- [ ] 理解CKKS原理

### 第2周：模型训练

- [ ] 定义PyTorch全连接网络
- [ ] 训练模型（准确率目标：>95%）
- [ ] 保存模型权重

### 第3周：同态加密实现

- [ ] 生成TenSEAL CKKS密钥上下文
- [ ] 实现图片向量加密/解密
- [ ] 实现密文矩阵乘法

### 第4周：系统集成

- [ ] 密文前向传播
- [ ] Flask Web界面开发
- [ ] 端到端测试

### 第5周：报告撰写

- [ ] 技术报告（5000-8000字）
- [ ] 代码整理

## 八、关键代码流程

```python
# 1. 生成密钥上下文
import tenseal as ts
context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

# 2. 加密输入
encrypted_input = ts.ckks_vector(context, image_pixels)

# 3. 密文推理
encrypted_hidden = encrypted_input.mm(weight1) + bias1  # 密文矩阵乘法
# ReLU需要特殊处理（多项式近似）...
encrypted_output = encrypted_hidden.mm(weight2) + bias2

# 4. 解密
result = encrypted_output.decrypt()
prediction = softmax(result)
```

## 九、潜在挑战与解决方案

| 挑战 | 解决方案 |
|------|----------|
| 密文ReLU难以实现 | 使用密文多项式近似或明文ReLU |
| 密文Softmax计算复杂 | 在解密后明文计算 |
| CKKS精度损失 | 选择合适的scale参数 |
| 密文推理速度慢 | 批处理优化 |

## 十、验收标准

- [ ] 神经网络训练准确率 > 95%
- [ ] 密文推理结果与明文一致
- [ ] Web界面可演示
- [ ] 技术报告（5000-8000字）

---

## 附录：依赖包

```
tenseal>=0.3.0
torch>=2.0.0
torchvision>=0.15.0
flask>=2.0.0
numpy>=1.24.0
Pillow>=9.0.0
```
