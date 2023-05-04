# Essential PyTorch
import torch
import torchaudio
import torch.nn.functional as F
import cv2

# Other modules used in this notebook
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import yaml
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
import torchvision
import math

# Custom packages
# from dataloader import FATDataset
# from utils import get_preprocessing
from .mediseg import UnetPlusPlus

class MyLearner(pl.LightningModule):

    def __init__(self, learning_rate=1e-4, train_batches = 0, val_batches = 0, in_channels=3, num_classes=2, weight_decay = 1e-2):

        super().__init__()
        self.learning_rate = learning_rate
        self.model = UnetPlusPlus(encoder_name="tu-hrnet_w30",
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )
        # print('Model', self.model)
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


    def validation_step(self, batch, batch_idx, split='val'):
        # print('batch number', batch_idx)
        # print('Inital iou score', self.iou_score)
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits,y.long()) #, weight=weight
        self.val_epoch_loss = self.val_epoch_loss.to(self.device) + loss.to(self.device)
        logits = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
    
        # Compute confusion matrix stats
        tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), y.long(), mode='multilabel', num_classes=2)
        self.iou_score = self.iou_score.to(self.device) + smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").to(self.device)
        self.log('Validation Loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('Validation IoU', self.iou_score, on_epoch=True, prog_bar=True, logger=True)
        self.f1score = self.f1score.to(self.device) + smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro").to(self.device)
        self.log('Validation F1scrore', self.f1score, on_epoch=True, prog_bar=True, logger=True)
        self.accuracy = self.accuracy.to(self.device) + smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").to(self.device)
        self.log('Validation Accuracy', self.accuracy, on_epoch=True, prog_bar=True, logger=True)
        dice = Dice(num_classes=2, average='macro').to(self.device)
        self.dice_score = self.dice_score.to(self.device) + dice(preds, y)#.to("cuda:0")
        self.log('Validation Dice', self.dice_score, on_epoch=True, prog_bar=True, logger=True)
        metric = MulticlassMatthewsCorrCoef(num_classes=2).to(self.device)
        self.mcc = self.mcc.to(self.device) + metric(preds, y)
        self.log('Validation MCC', self.mcc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, split='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        # scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,  T_0 = 1060, verbose=True) # T_0 = number of steps in ten epochs
        scheduler = CosineAnnealingLR(optimizer, 30, eta_min=1e-8, last_epoch=- 1, verbose=True) # T_0 = number of steps in ten epochs
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return optimizer

    # def train_dataloader(self):
    #     return train_loader

    # def val_dataloader(self):
    #     return valid_loader

    # def test_dataloader(self):
    #     return test_loader


def convert_to_rgb(mask, img_name):
    """Function to convert typemaps to rgb masks"""
    print('RGB FUNC')
    mask = torch.squeeze(mask)
    print(mask.shape)
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    white = mask == 1
    print(white)
    try:
        rgb_mask[white] = [255,255,255]
    except Exception as e:
        print(e)

    print(rgb_mask.shape)
    print(cv2.imwrite(os.path.join('/truba/home/isahin/gaugan_pytorch/gaugan_tubule_test/real_preds', img_name), rgb_mask[:,:,[2,1,0]]))


def eval_acc(debug_name=None, num_classes=2, in_channels = 3):
    ENCODER = "tu-hrnet_w30"
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = MyLearner(num_classes=num_classes, in_channels=in_channels).load_from_checkpoint("/truba/home/isahin/gaugan_pytorch/gaugan_tubule/lightning_logs/lightning_logs/version_7/checkpoints/epoch=110-step=43401.ckpt")
    model = model.eval().to('cuda')
    # Loading data
    # test_batch_size = 16
    #load_data_and_labels(r"/home/farhan/audio_classification/audio_classification_dir/ali_test_data",test_lst, test_labels)
    #test_dataset = FATDataset(img_path = "/truba/home/isahin/gaugan_pytorch/gaugan_tubule/tubule_data/tubule/train/images", mask_path = '/truba/home/isahin/gaugan_pytorch/gaugan_tubule/tubule_data/tubule/train/masks', mode = "Test")#, preprocessing=get_preprocessing(preprocessing_fn))
    #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, pin_memory=True)
    
    with torch.no_grad():
            img = cv2.imread("/truba/home/isahin/gaugan_pytorch/gaugan_tubule/tubule_data/tubule/train/images/155.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #mask = cv2.imread("/truba/home/isahin/gaugan_pytorch/gaugan_tubule/tubule_data/tubule/train/masks/105.png",cv2.IMREAD_UNCHANGED)

            img = torchvision.transforms.functional.to_tensor(img)
            img = img[None,:]
            print(img.shape)
            logits = model(img.to(device))
            preds = torch.argmax(logits, dim=1).cpu()
            print(preds.shape)
            convert_to_rgb(preds, '155.png')

if __name__ == '__main__':
    # Check evaluation accuracy 
    eval_acc(debug_name=True, in_channels=2)