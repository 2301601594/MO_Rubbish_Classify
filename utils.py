import os
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from dataset import RubbishDataset
from torch.optim import lr_scheduler

from my_models.MobileNetV2 import MobileNetV2
from my_models.MobileNetV2_modified import MobileNetV2Modified
from my_models.MobileNetV3 import MobileNetV3
from my_models.MobileNetV4 import MobileNetV4
from my_models.EfficientNetV2 import EfficientNetV2
from my_models.EfficientNetV2_modified import EfficientNetV2Modified
from my_models.SeResNext import SeResNext


# Get Dataloader
def get_loader(data_path='./garbage_26x100', train_transform=None, val_transform=None, batch_size=32, num_workers=2):
    train_set = RubbishDataset(os.path.join(data_path, 'train'), transform=train_transform)
    val_set = RubbishDataset(os.path.join(data_path, 'val'), transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# Get Transform
def get_transforms():
    train_transform = T.Compose([
        # 1. 缩放与裁剪 (核心增强)
        # T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        # 2. 几何变换 (模拟不同角度)
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),  # 垃圾是可能倒置的
        T.RandomRotation(degrees=30),  # 旋转±30度
        # 3. 颜色与光照 (模拟不同环境)
        # T.ColorJitter 是一个非常强大的工具
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # 4. 格式转换 (必须)
        T.ToTensor(),
        # 5. 标准化 (必须)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


# Get Model
def get_model(model):
    res_model = None
    if model == 'MobileNetV2':
        res_model = MobileNetV2()
    if model == 'MobileNetV2_modified':
        res_model = MobileNetV2Modified()
    if model == 'MobileNetV3':
        res_model = MobileNetV3()
    if model == 'MobileNetV4':
        res_model = MobileNetV4()
    if model == 'EfficientNetV2':
        res_model = EfficientNetV2()
    if model == 'EfficientNetV2_modified':
        res_model = EfficientNetV2Modified()
    if model == 'SeResNext':
        res_model = SeResNext()
    return res_model


# Get Scheduler
def get_scheduler(optimizer, scheduler, epoch):
    if scheduler == 'ReduceLROnPlateau':
        s = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return s
    if scheduler == 'CosineAnnealingLR':
        s = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-8)
        return s
    if scheduler == 'CosineAnnealingWarmRestarts':
        s = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return s


def train_fn(model, loss_fn, train_loader, optimizer, device):
    model.train()
    loop = tqdm(train_loader)
    total_loss = 0
    for image, label in loop:
        image = image.to(device)
        label = label.to(device)
        pred = model(image)
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    return total_loss / len(train_loader)


def eval_fn(model, loss_fn, scheduler, val_loader, device):
    model.eval()
    loop = tqdm(val_loader)
    total_loss = 0
    all_labels = []
    all_preds = []
    for image, label in loop:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model(image)
            # evaluate loss
            loss = loss_fn(pred, label)
            total_loss += loss.item()
            # eval acc and f1
            pred_index = pred.argmax(dim=1)
            all_preds.extend(pred_index.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, acc, f1


if __name__ == '__main__':
    data_path = './garbage_26x100'
    model = get_model('MobileNetV3')
    print(model)
