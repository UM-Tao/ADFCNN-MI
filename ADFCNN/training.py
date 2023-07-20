'''Import libraires'''
import os, yaml
from datetime import datetime
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import KFold
from dataloader.bci_compet import get_dataset
from model.litmodel import get_litmodel
from utils.setup_utils import (
    get_device,
    get_log_name,
)
from utils.training_utils import get_callbacks

'''Argparse'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--subject_num', type=int, default=17)
parser.add_argument('--fold_num', type=int, default=0)
parser.add_argument('--gpu_num', type=str, default='0')
parser.add_argument('--config_name', type=str, default='bcicompet2b_config')
aargs = parser.parse_args()


# Config setting
with open('/home/user_taowei/PycharmProjects/ADFCNN/configs/bcicompet2b_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    args = EasyDict(config)


#### Set SEED ####
seed_everything(args.SEED)


#### Set Device ####
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = aargs.gpu_num
args['device'] = get_device(aargs.gpu_num)
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deterministic = True


#### Set Log ####
args['current_time'] = datetime.now().strftime('%Y%m%d')
args['LOG_NAME'] = get_log_name(args)


#### Update configs ####
args.lr = float(args.lr)
if args.downsampling != 0: args['sampling_rate'] = args.downsampling


'''Training''' # 
for num_subject in range(args.num_subjects):
    # if num_subject != aargs.subject_num: continue
    args['target_subject'] = num_subject
    
    dataset = get_dataset(aargs.config_name, args)

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.SEED)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        # if fold != aargs.fold_num: continue

        ### Set dataloader ###
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_dataloader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    pin_memory=False,
                                    num_workers=args.num_workers,
                                    sampler=train_subsampler)
        val_dataloader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    pin_memory=False,
                                    num_workers=args.num_workers,
                                    sampler=val_subsampler)

        model = get_litmodel(args)
        logger = TensorBoardLogger(args.LOG_PATH,
                                    name='/data/home/yc07466/Project/DHCNN/{args.LOG_NAME}/S{args.target_subject:02d}_fold{fold + 1}')
        callbacks = get_callbacks(fold=fold, monitor='val_acc', args=args)

        trainer = Trainer(enable_progress_bar=True,
            max_epochs=args.EPOCHS,
            gpus=[int(aargs.gpu_num)],
            callbacks=callbacks,
            default_root_dir=args.CKPT_PATH,
            logger=logger,
        )

        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)

        torch.cuda.empty_cache()

