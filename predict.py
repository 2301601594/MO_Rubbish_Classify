import torch
from torch.nn import BatchNorm1d
from torchvision import models
from torchvision import transforms as T
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import ttach as tta
from tqdm import tqdm

from utils import get_loader, get_transforms, get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = 'SeResNext'


model1 = get_model(MODEL).to(device)
model1.load_state_dict(torch.load("./checkpoints/SeResNext.pth"))
model1.eval()

model2 = get_model(MODEL).to(device)
model2.load_state_dict(torch.load('./checkpoints/SeResNext.pth'))
model2.eval()

index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14,
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}

inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle',
            6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror',
            14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery',
            20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image):

    tta_model_1 = tta.ClassificationTTAWrapper(model1, tta.aliases.flip_transform(), merge_mode='mean')
    tta_model_2 = tta.ClassificationTTAWrapper(model2, tta.aliases.flip_transform(), merge_mode='mean')

    with torch.no_grad():
        logits_1 = tta_model_1(image.to(device))  #预测
        logits_2 = tta_model_2(image.to(device))
        logits = logits_1 + logits_2
        return inverted[logits.argmax(dim=-1).cpu().numpy().item()]

if __name__ == '__main__':
    total_correct = 0
    wrong_preds = []
    train_transform, val_transform = get_transforms()
    _, val_loader = get_loader(data_path='./garbage_26x100', train_transform=train_transform,
                                          val_transform=val_transform, batch_size=1, num_workers=2)

    loop = tqdm(val_loader)
    for image, label in loop:
        res = predict(image)
        if res == inverted[label.item()]:
            total_correct += 1
        else:
            wrong_preds.append(f'Correct:{inverted[label.item()]}->Wrong:{res}')


    print(total_correct/len(val_loader)* 100)
    print(wrong_preds)

