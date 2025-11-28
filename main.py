import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *
from torch.optim import Adam, SGD, AdamW

# HyperParams
TASK_NAME = 'T-6-1'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = 'MobileNetV3'
SCHEDULER = 'CosineAnnealingLR'
DATA_PATH = './garbage_26x100'
BATCH_SIZE = 32
NUM_WORKERS = 2
LR_RATE = 1e-2
EPOCHS = 150


def main():
    # Load Data
    train_transform, val_transform = get_transforms()
    train_loader, val_loader = get_loader(data_path=DATA_PATH, train_transform=train_transform,
                                          val_transform=val_transform, batch_size=64, num_workers=NUM_WORKERS)

    # Load Model
    model = get_model(model=MODEL)
    model = model.to(DEVICE)

    # Init Train
    if not os.path.exists(f'./checkpoints/{MODEL}'):
        os.makedirs(f'./checkpoints/{MODEL}')
    if not os.path.exists(f'./checkpoints/{MODEL}/{TASK_NAME}'):
        os.mkdir(f'./checkpoints/{MODEL}/{TASK_NAME}')
    best_f1 = 0

    # Start Training
    writer = SummaryWriter(log_dir=f'./logs/{MODEL}/{TASK_NAME}')
    optimizer = AdamW(model.parameters(), lr=LR_RATE, weight_decay=1e-3)
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
