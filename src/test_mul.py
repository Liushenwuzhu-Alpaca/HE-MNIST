import sys
sys.path.insert(0, "src")

import numpy as np
import tenseal as ts
import tenseal.operations as ops

# 检查 tenseal 的操作模块
print("=== tenseal 可用操作 ===")
print(dir(ops))

# 直接用新 context 测试
print("\n=== 直接测试 CKKS ===")

# 创建新 context
ctx = ts.context(ts.SCHEME_TYPE.CKKS, 16384, global_scale=2**40)
ctx.generate_galois_keys()
ctx.auto_rescale = True

print(f"ctx.global_scale: {ctx.global_scale}")
print(f"ctx.auto_rescale: {ctx.auto_rescale}")

enc_a = ts.ckks_vector(ctx, [2.0])
enc_b = ts.ckks_vector(ctx, [3.0])

print(f"加密 2.0: {np.array(enc_a.decrypt())}")
print(f"加密 3.0: {np.array(enc_b.decrypt())}")

result = enc_a * enc_b
print(f"乘法结果: {np.array(result.decrypt())}")
print(f"期望: 6.0")
