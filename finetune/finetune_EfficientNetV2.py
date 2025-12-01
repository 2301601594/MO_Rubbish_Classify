# Fine-tune the model
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *
from torch.optim import Adam, SGD, AdamW

# HyperParams
TASK_NAME = 'EV2_finetune_1'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = 'EfficientNetV2_modified'
SCHEDULER = 'CosineAnnealingLR'
DATA_PATH = '../garbage_26x100'
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 10


def main():
    # Load Data
    train_transform, val_transform = get_transforms()
    train_loader, val_loader = get_loader(data_path=DATA_PATH, train_transform=train_transform,
                                          val_transform=val_transform, batch_size=64, num_workers=NUM_WORKERS)

    # Load Model
    model = get_model(model=MODEL)
    model.load_state_dict(torch.load(f'../checkpoints/EfficientNetV2_modified/E_M_init/last.pth'))
    model = model.to(DEVICE)

    # Unfrozen Params
    for param in model.model.blocks[-2:].parameters():
        param.requires_grad = True

    # 同样解冻 'conv_head' 和它后面的 'bn2'
    for param in model.model.conv_head.parameters():
        param.requires_grad = True
    for param in model.model.bn2.parameters():
        param.requires_grad = True

    # --- 3. 设置 阶段二 的优化器 (差异化 LR) ---
    # 头部
    params_head = model.model.classifier.parameters()

    # 解冻的骨干层
    params_backbone_unfrozen = (
            list(model.model.blocks[-2:].parameters()) +
            list(model.model.conv_head.parameters()) +
            list(model.model.bn2.parameters())
    )

    # 找出并冻结其余层 (保险起见)
    # (这包括 blocks[0]...blocks[3], conv_stem, bn1)
    params_backbone_frozen = (
            list(model.model.conv_stem.parameters()) +
            list(model.model.bn1.parameters()) +
            list(model.model.blocks[:-2].parameters())
    )
    for param in params_backbone_frozen:
        param.requires_grad = False

    # 定义差异化学习率的优化器
    optimizer = AdamW([
        {'params': params_backbone_unfrozen, 'lr': 1e-6},  # 骨干
        {'params': params_head, 'lr': 1e-5}  # 头部
    ], lr=1e-6, weight_decay=1e-2)  # 强权重衰减


    # Init Train
    if not os.path.exists(f'../checkpoints/{MODEL}'):
        os.makedirs(f'../checkpoints/{MODEL}')
    if not os.path.exists(f'../checkpoints/{MODEL}/{TASK_NAME}'):
        os.mkdir(f'../checkpoints/{MODEL}/{TASK_NAME}')
    best_f1 = 0

    # Start Training
    writer = SummaryWriter(log_dir=f'../logs/{MODEL}/{TASK_NAME}')
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
            torch.save(model.state_dict(), f'../checkpoints/{MODEL}/{TASK_NAME}/best.pth')
            best_f1 = f1
        torch.save(model.state_dict(), f'../checkpoints/{MODEL}/{TASK_NAME}/last.pth')

    writer.close()


if __name__ == '__main__':
    main()
