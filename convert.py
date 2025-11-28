import torch
import torch.nn as nn
from my_models.MobileNetV3 import MobileNetV3

# 假设你的ResNet定义在resnet.py文件中
model = MobileNetV3()

# 加载权重
checkpoint = torch.load('./checkpoints/MobileNetV3/V3_fine_tune_2/best.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)  # 使用strict=False可以忽略不匹配的键

model.eval()
# 将模型转换为TorchScript
example_input = torch.rand(1, 3, 224, 224)  # 修改这里以匹配你的模型输入尺寸
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("model.pt")