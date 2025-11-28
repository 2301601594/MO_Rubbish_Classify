import torch.nn as nn
import timm
from safetensors.torch import load_file

weights_path = '/home/dongj/python_proj/Rubbish Classify/pretrain_weights/efficientnetv2.safetensors'
model_name = 'efficientnetv2_rw_s.ra2_in1k'

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=26):
        super(EfficientNetV2, self).__init__()

        model = timm.create_model(
            model_name,
            pretrained=False,  # <-- 设置为 False
        )

        static_dict = load_file(weights_path)
        model.load_state_dict(static_dict)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model.classifier.requires_grad = True
        self.model = model

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = EfficientNetV2(num_classes=26)