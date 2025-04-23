import torch

# 检查 CUDA 是否可用
print("CUDA Available:", torch.cuda.is_available())

# 查看可见的 GPU 数量
print("GPU Count:", torch.cuda.device_count())

# 查看当前 CUDA 设备
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# 打印 CUDA 版本信息（如果有保存）
print("CUDA Version:", torch.version.cuda)

print(torch.__version__)         # 应输出 1.8.0
print(torch.cuda.is_available())