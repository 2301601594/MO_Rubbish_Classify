# Fine-tune the model
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *
from torch.optim import Adam, SGD, AdamW

# HyperParams
TASK_NAME = 'SE_finetune_3'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = 'SeResNext'
SCHEDULER = 'CosineAnnealingLR'
DATA_PATH = './garbage_26x100'
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 100


def main():
    # Load Data
    train_transform, val_transform = get_transforms()
    train_loader, val_loader = get_loader(data_path=DATA_PATH, train_transform=train_transform,
                                          val_transform=val_transform, batch_size=64, num_workers=NUM_WORKERS)

    # Load Model
    model = get_model(model=MODEL)
    model.load_state_dict(torch.load(f'/home/dongj/python_proj/Rubbish Classify/checkpoints/SeResNext/SE_2/last.pth'))
    model = model.to(DEVICE)

    # Unfrozen Params
    for param in model.model.layer4.parameters():
        param.requires_grad = True

    for param in model.model.layer3.parameters():
        param.requires_grad = True

    # --- 3. 设置 阶段二 的优化器 (差异化 LR) ---
    # (确保您新加的头部 'model.fc' 也是可训练的)
    params_head = model.model.fc.parameters()

    # 将所有解冻的骨干层放在一起
    params_backbone_unfrozen = list(model.model.layer3.parameters()) + list(model.model.layer4.parameters())

    # 找出并冻结其余层 (保险起见)
    params_backbone_frozen = (
            list(model.model.conv1.parameters()) +
            list(model.model.bn1.parameters()) +
            list(model.model.layer1.parameters()) +
            list(model.model.layer2.parameters())
    )
    for param in params_backbone_frozen:
        param.requires_grad = False

    # 定义差异化学习率的优化器
    optimizer = AdamW([
        {'params': params_backbone_unfrozen, 'lr': 1e-6},  # 骨干用极低 LR (1e-6)
        {'params': params_head, 'lr': 1e-5}  # 头部用稍高 LR (1e-5)
    ], lr=1e-6, weight_decay=1e-3)  # 仍然需要强权重衰减


    # Init Train
    if not os.path.exists(f'./checkpoints/{MODEL}'):
        os.makedirs(f'./checkpoints/{MODEL}')
    if not os.path.exists(f'./checkpoints/{MODEL}/{TASK_NAME}'):
        os.mkdir(f'./checkpoints/{MODEL}/{TASK_NAME}')
    best_f1 = 0

    # Start Training
    writer = SummaryWriter(log_dir=f'./logs/{MODEL}/{TASK_NAME}')
    # optimizer = Adam(model.parameters(), lr=LR_RATE, weight_decay=1e-3)
    # optimizer = SGD(model.parameters(), lr=LR_RATE, momentum=0.9, weight_decay=1e-3)
    if SCHEDULER is not None:
        scheduler = get_scheduler(scheduler=SCHEDULER, optimizer=optimizer, epoch=EPOCHS)
    else:
        scheduler = None

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(1, EPOCHS + 1):
        # Train
        loss = train_fn(model=model, train_loader=train_loader, optimizer=optimizer,
                        loss_fn=loss_fn, device=DEVICE)
        writer.add_scalar(f'Train_Loss/{TASK_NAME}', loss, epoch)
        # Eval
        eval_loss, acc, f1 = eval_fn(model=model, val_loader=val_loader, loss_fn=loss_fn, scheduler=scheduler,
                                     device=DEVICE)
        writer.add_scalar(f'Eval_Loss/{TASK_NAME}', eval_loss, epoch)
        writer.add_scalar(f'Accuracy/{TASK_NAME}', acc, epoch)
        writer.add_scalar(f'F1_Score/{TASK_NAME}', f1, epoch)

        if SCHEDULER is not None:
            scheduler.step()

        if f1 > best_f1:
            torch.save(model.state_dict(), f'./checkpoints/{MODEL}/{TASK_NAME}/best.pth')
            best_f1 = f1

        torch.save(model.state_dict(), f'./checkpoints/{MODEL}/{TASK_NAME}/last.pth')

    writer.close()


if __name__ == '__main__':
    main()
