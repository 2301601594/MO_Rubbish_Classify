import torch
import torch.nn as nn
import torchvision.models as models


# Build the model
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=26):
        super(MobileNetV3, self).__init__()
        model = models.mobilenet_v3_large(pretrained=True)
        # Freeze Params
        for param in model.features.parameters():
            param.requires_grad = False
        # Modify Head
        in_features = 1280

        model.classifier[3] = nn.Linear(in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = MobileNetV3()
    print(model)