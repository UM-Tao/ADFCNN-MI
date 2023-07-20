import torch
import torch.optim as optim

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def get_criterion():
    
    criterion = torch.nn.CrossEntropyLoss()
    
    return criterion


def get_optimizer(model, args):
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    return optimizer


def get_scheduler(optimizer, args):
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS, eta_min=0)
    
    return scheduler


def get_checkpoint_callback(fold: int, monitor: str, args):
    if monitor == 'val_acc':
        return ModelCheckpoint(monitor=monitor,
                                dirpath=f'{args.CKPT_PATH}/{args.LOG_NAME}/fold_{fold + 1}',
                                filename=f'{args.task}_S{args.target_subject:02d}_' + '{epoch:02d}-{val_acc:.3f}',
                                save_top_k=3,
                                mode='max')
    elif monitor == 'val_loss':
        return ModelCheckpoint(monitor=monitor,
                                dirpath=f'{args.CKPT_PATH}/{args.LOG_NAME}/fold_{fold + 1}',
                                filename=f'{args.task}_S{args.target_subject:02d}_' + '{epoch:02d}-{val_loss:.3f}',
                                save_top_k=3,
                                mode='min')
    else:
        return None


def get_callbacks(fold: int, monitor: str, args):
    checkpoint_callback = get_checkpoint_callback(fold=fold, monitor=monitor, args=args)
    lr_logger=LearningRateMonitor(logging_interval='epoch')
    return [checkpoint_callback, lr_logger]

