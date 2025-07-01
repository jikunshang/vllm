import pickle
import torch
import habana_frameworks.torch as htorch

# 定义一个示例 FP8_e4m3 类型的张量
# 注意：FP8_e4m3 是一种低精度浮点格式，通常用于深度学习中的混合精度训练
# 在 PyTorch 中，可以通过 torch.float8_e4m3 创建 FP8 类型的张量
# 这里我们先创建一个浮点数张量，然后将其转换为 FP8_e4m3 类型
fp8_tensor = torch.randn(2, 3).to(torch.float8_e4m3fn)

print("原始 FP8_e4m3 张量:")
print(fp8_tensor)
print("数据类型:", fp8_tensor.dtype)
print("-" * 50)

# 使用 pickle 序列化 FP8 张量
with open("fp8_tensor.pkl", "wb") as f:
    pickle.dump(fp8_tensor, f)
    print("FP8 张量已序列化并保存到 fp8_tensor.pkl 文件中")

print("-" * 50)

# 使用 pickle 反序列化 FP8 张量
with open("fp8_tensor.pkl", "rb") as f:
    loaded_fp8_tensor = pickle.load(f)
    print("反序列化后的 FP8 张量:")
    print(loaded_fp8_tensor)
    print("数据类型:", loaded_fp8_tensor.dtype)

# 验证序列化和反序列化是否成功
assert torch.allclose(fp8_tensor.float(), loaded_fp8_tensor.float()), "序列化和反序列化不一致"
print("-" * 50)
print("序列化和反序列化成功，数据一致")