import torch
import torch.nn as nn
import torchvision.models as models


# Build the model
class MobileNetV2Modified(nn.Module):
    def __init__(self, num_classes=26):
        super(MobileNetV2Modified, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        # Freeze Params
        for param in model.parameters():
            param.requires_grad = False
        # Modify Head
        in_features = model.classifier[1].in_features
        hidden_dim = 512
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),  # MobileNetV2 原本自带的 Dropout
            nn.Linear(in_features, hidden_dim),  # 第一个全连接层 (1280 -> 512)
            nn.ReLU(),  # 非线性激活函数 (关键!)
            nn.BatchNorm1d(hidden_dim),  # (推荐) 批标准化，加速收敛
            nn.Dropout(p=0.5),  # (关键) 添加更强的 Dropout 来防止"头"过拟合
            nn.Linear(hidden_dim, num_classes)  # 输出层 (512 -> 26)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)

