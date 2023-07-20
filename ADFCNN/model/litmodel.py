import torch

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from utils.training_utils import get_criterion, get_scheduler, get_optimizer

from model.ADFCNN import get_model
# from model.FBCNet import get_model
# from model.ShallowConvNet import get_model
# from model.DeepConvNet import get_model
# from model.EEGNet import get_model
# from model.IFNetV2 import get_model

class LitModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        
        self.save_hyperparameters()
        self.model = get_model(args)
        self.criterion = get_criterion()
        self.args = args
    
    
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            self.sample_batch = batch['data'].type(torch.float)

        inputs = batch['data'].type(torch.float)
        labels = batch['label'].type(torch.long) if batch['label'].dim() == 1 else batch['label']
        
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        
        acc = accuracy(preds, labels if labels.dim() == 1 else torch.argmax(labels, dim=1))
        self.log('train_loss', loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True)
        self.log('train_acc', acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True)
        return {'loss': loss}

    
    def training_epoch_end(self, outputs):
        if self.current_epoch == 0:
            self.logger.experiment.add_graph(self.model, self.sample_batch)
    
    
    def evaluate(self, batch, stage=None):
        inputs = batch['data'].type(torch.float)
        labels = batch['label'].type(torch.long) if batch['label'].dim() == 1 else batch['label']
        
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy(preds, labels if labels.dim() == 1 else torch.argmax(labels, dim=1))
        
        if stage:
            self.log(f'{stage}_loss', loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True)
            self.log(f'{stage}_acc', acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True)
    
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage='val')
    
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')
        
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch['data'].type(torch.float)
        return self(inputs)
    
    
    def configure_optimizers(self):
        self.__init_optimizers()
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler_config}
    
    
    def __init_optimizers(self):
        self.optimizer = get_optimizer(self, self.args)
        self.lr_scheduler = get_scheduler(self.optimizer, self.args)
        self.lr_scheduler_config = {
            'scheduler': self.lr_scheduler,
            'internal': 'epoch'
        }


def get_litmodel(args):
    
    model = LitModel(args)
    
    return model

