import torch
import torch.nn as nn
import torchvision.models as models


# Build the model
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=26):
        super(MobileNetV2, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        # Freeze Params
        for param in model.parameters():
            param.requires_grad = False
        # Modify Head
        in_feature = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feature, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    test_model = MobileNetV2()
    print(test_model)
    x = torch.randn(1, 3, 224, 224)
    y = test_model(x)
    print(y)
