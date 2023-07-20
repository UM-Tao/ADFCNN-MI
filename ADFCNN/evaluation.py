'''Import libraires'''
import os, yaml
from datetime import datetime
from easydict import EasyDict
from glob import glob

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

from dataloader.bci_compet import get_dataset
from model.litmodel import get_litmodel
from utils.setup_utils import (
    get_device,
)


'''Argparse'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='bcicompet2a_config')
parser.add_argument('--ckpt_path', type=str, default='20230720_task_BCICompet2a_batch_16_lr_1e-3_window_3_IFNet')
parser.add_argument('--subject_num', type=int, default=1)
aargs = parser.parse_args()


### Set confings
config_name = aargs.config_name

with open(f'configs/{config_name}.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    args = EasyDict(config)
    
args['current_time'] = datetime.now().strftime('%Y%m%d')

#### Set Checkpoint ####
args['LOG_NAME'] = aargs.ckpt_path

#### Set Device ####
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_NUM
args['device'] = get_device(args.GPU_NUM)
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deterministic = True

#### Set SEED ####
seed_everything(args.SEED)

#### Update configs ####
if args.downsampling != 0: args['sampling_rate'] = args.downsampling


### Evaluation
total_results = []
total_kappas = []
preds = []
preds_label = []
labels = []

args.is_test = True
for num_subject in range(args.num_subjects):
    # if num_subject != aargs.subject_num: continue
    args['target_subject'] = num_subject
    
    dataset = get_dataset(config_name, args)
        
    results = np.zeros((dataset.data.shape[0], args.num_classes))
    
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.SEED)

    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        test_dataloader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    pin_memory=False,
                                    num_workers=args.num_workers)
        
        ckpt_path = sorted(glob(f'{args.CKPT_PATH}/{args.LOG_NAME}/fold_{fold + 1}/*S{num_subject:02d}*'))[-1]
        print(ckpt_path)
        model = get_litmodel(args)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device)['state_dict'], strict=False)
        
        trainer = Trainer(
            gpus=[int(args.GPU_NUM)],
        )
        
        logits = trainer.predict(model, dataloaders=test_dataloader)
        result = torch.cat(logits, dim=0).argmax(axis=1)
        result = F.one_hot(result, num_classes=args.num_classes).detach().cpu().numpy()
        results += result     
        
        torch.cuda.empty_cache()
        
    results /= 10
        
    preds.append(results)
    results = results.argmax(axis=1)
    preds_label.append(results)
    total_results.append(accuracy_score(results, dataset.label))
    total_kappas.append(cohen_kappa_score(results, dataset.label))
    labels.append(dataset.label)


### Accuracy
acc_result_df = pd.DataFrame(total_results)
acc_result_df.index = [f'S{idx + 1}' for idx in range(args.num_subjects)]
acc_result_df.loc['Avg.'] = acc_result_df.mean()


### Kappa
kappa_result_df = pd.DataFrame(total_kappas)
kappa_result_df.index = [f'S{idx + 1}' for idx in range(args.num_subjects)]
kappa_result_df.loc['Avg.'] = kappa_result_df.mean()

result_df = pd.merge(acc_result_df, kappa_result_df, left_index=True, right_index=True, how='inner')
result_df.columns = ['Acc.', 'Kappa']

print('\n\n')
print('='*24)
print('='*7, ' Result ', '='*7)
print(result_df)
print('='*24)
print('='*24)
print('\n\n')
