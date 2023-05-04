# Essential PyTorch
import torch
import torchaudio
import torch.nn.functional as F

# Other modules used in this notebook
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import yaml
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from torchmetrics.classification import Accuracy, F1Score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning import loggers as pl_loggers
import torchvision.transforms as transforms
# from torchmetrics.functional import dice_score
from torchmetrics import Dice
from torchmetrics.classification import MulticlassMatthewsCorrCoef
import segmentation_models_pytorch as smp
import math

# Custom packages
from gaugan_pytorch.gaugan_fat.fat_data.dataloader import FATDataset
from gaugan_pytorch.gaugan_fat.utils.utils import get_preprocessing
from mediseg import UnetPlusPlus

def convert_to_rgb(mask, path):
    """Function to convert typemaps to rgb masks"""
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    white = mask == 1

    rgb_mask[white] = [255,255,255]
    cv2.imwrite(path, rgb_mask[:,:,[2,1,0]])

class MyLearner(pl.LightningModule):

    def __init__(self, learning_rate=1e-4, train_batches = None, val_batches = None, in_channels=3, num_classes=2, weight_decay = 1e-2):

        super().__init__()
        self.learning_rate = learning_rate
        self.model = UnetPlusPlus(encoder_name="tu-hrnet_w30",
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )
        print('Model', self.model)
        # self.model = smp.UnetPlusPlus(
        # encoder_name="tu-hrnet_w30",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7efficientnet-b7
        # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        # in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        # classes=num_classes,                      # model output channels (number of classes in your dataset)
        # )   # Set model
        
        self.weight_decay = weight_decay
        self.classes = num_classes
        self.train_epoch_loss = torch.tensor(0)
        self.val_epoch_loss = torch.tensor(0)
        self.num_train_batches = torch.tensor(math.ceil(train_batches))
        self.num_val_batches = torch.tensor(math.ceil(val_batches))
        # self.val_accuracy = 0
        # self.val_F1Score = 0
        self.iou_score = torch.tensor(0)
        #print('IOu score intialization', self.iou_score)
        self.f1score = torch.tensor(0)
        #self.f2_score = 0
        self.accuracy = torch.tensor(0)
        #self.recall = 0
        self.dice_score = torch.tensor(0)
        self.mcc = torch.tensor(0)

        print('Number of train batches:', self.num_train_batches)
        print('Number of val batches:', self.num_val_batches)

    def forward(self, x):
        x = self.model(x)    # Apply activation in the step funcs
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits,y.long()) #, weight=weight
        self.log('Training loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.train_epoch_loss = self.train_epoch_loss.to(self.device) + loss.to(self.device)
        return loss

    # def training_epoch_end(self, outs):
    #     print('train epoch loss', self.train_epoch_loss) 
    #     print(type(self.train_epoch_loss))
    #     print(type(self.num_train_batches))
    #     self.log(f'Training_loss', self.train_epoch_loss.to(self.device)/self.num_train_batches.to(self.device))#.cpu().detach().numpy()
    #     print(f'Training_loss', self.train_epoch_loss.to(self.device)/self.num_train_batches.to(self.device))
    #     #Resetting the metrics for upcoming epoch
    #     self.train_epoch_loss = torch.tensor(0)

    def validation_step(self, batch, batch_idx, split='val'):
        x, y = batch
        logits = self(x)
        #print(logits.shape)
        loss = F.cross_entropy(logits,y.long()) #, weight=weight
        #self.val_epoch_loss = self.val_epoch_loss.to(self.device) + loss.to(self.device)
        logits = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        preds_to_save = preds.cpu().detach().numpy()
        # print('preds shape', preds.shape)

        # for ind in range(x.shape[0]):
        #     print(x[ind].permute(1,2,0).cpu().detach().numpy().shape)
        #     convert_to_rgb(x[ind].permute(1,2,0).cpu().detach().numpy(), str(ind)+'.png')
        #     #cv2.imwrite(os.path.join('/truba/home/isahin/gaugan_pytorch/gaugan_fat/val_imgs',str(ind)+'.png'), x[ind].permute(1,2,0).cpu().detach().numpy())

        for ind in range(y.shape[0]):
            convert_to_rgb(y[ind].cpu().detach().numpy(), os.path.join('/truba/home/isahin/gaugan_pytorch/gaugan_fat/val_masks',str(ind)+'.png'))
            #cv2.imwrite(os.path.join('/truba/home/isahin/gaugan_pytorch/gaugan_fat/val_masks',str(ind)+'.png'), y[ind].cpu().detach().numpy())

        for ind in range(preds.shape[0]):
            convert_to_rgb(preds_to_save[ind], os.path.join('/truba/home/isahin/gaugan_pytorch/gaugan_fat/val_pred',str(ind)+'.png'))
            # cv2.imwrite(os.path.join('/truba/home/isahin/gaugan_pytorch/gaugan_fat/val_pred',str(ind)+'.png'), preds_to_save[ind])

        # Compute confusion matrix stats
        tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), y.long(), mode='binary', num_classes=2)
        self.iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").to(self.device)
        self.log('Validation Loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('Validation IoU', self.iou_score, on_epoch=True, prog_bar=True, logger=True)
        self.f1score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro").to(self.device)
        self.log('Validation F1scrore', self.f1score, on_epoch=True, prog_bar=True, logger=True)
        self.accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").to(self.device)
        self.log('Validation Accuracy', self.accuracy, on_epoch=True, prog_bar=True, logger=True)
        dice = Dice(num_classes=2, average='macro').to(self.device)
        self.dice_score = dice(preds, y)
        self.log('Validation Dice', self.dice_score, on_epoch=True, prog_bar=True, logger=True)
        metric = MulticlassMatthewsCorrCoef(num_classes=2).to(self.device)
        self.mcc =metric(preds, y)
        self.log('Validation MCC', self.mcc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, split='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        # scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,  T_0 = 1060, verbose=True) # T_0 = number of steps in ten epochs
        scheduler = CosineAnnealingLR(optimizer, 30, eta_min=1e-8, last_epoch=- 1, verbose=True) # T_0 = number of steps in ten epochs
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, threshold=0.001, 
        #                                                        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}#, "monitor": "Validation Loss"
        # return optimizer

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return valid_loader

    # def test_dataloader(self):
    #     return test_loader