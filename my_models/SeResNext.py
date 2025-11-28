import torch.nn as nn
import timm
from safetensors.torch import load_file

weights_path = '/home/dongj/python_proj/Rubbish Classify/pretrain_weights/SeResNext.safetensors'
model_name = 'seresnext50_32x4d.racm_in1k'

class SeResNext(nn.Module):
    def __init__(self, num_classes=26):
        super(SeResNext, self).__init__()

        model = timm.create_model(
            model_name,
            pretrained=False,  # <-- 设置为 False
        )

        static_dict = load_file(weights_path)
        model.load_state_dict(static_dict)
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.model = model

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = SeResNext(num_classes=26)
    print(model)