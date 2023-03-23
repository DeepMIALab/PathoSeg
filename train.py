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
from dataloader import FATDataset
from utils import get_preprocessing
from mediseg import UnetPlusPlus

class MyLearner(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, train_batches = None, val_batches = None, in_channels=3, num_classes=2):

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

    # def validation_epoch_end(self, outs): 
    #     print(self.val_epoch_loss/self.num_val_batches.to(self.device))
    #     self.log(f'Validation Loss', self.val_epoch_loss/self.num_val_batches.to(self.device), prog_bar=True)       #.cpu().detach().numpy()
    #     self.log(f'Valid IoU', self.iou_score/self.num_val_batches.to(self.device), prog_bar=True)#.cpu().detach().numpy()
    #     self.log(f'Valid Dice', self.dice_score/self.num_val_batches.to(self.device), prog_bar=True) #.cpu().detach().numpy()
    #     self.log(f'Valid F1 score', self.f1score/self.num_val_batches.to(self.device), prog_bar=True)
    #     #self.log(f'Valid F2 score', self.f2_score.cpu().detach().numpy()/self.num_val_batches, prog_bar=True)
    #     self.log(f'Valid Accuracy', self.accuracy/self.num_val_batches.to(self.device), prog_bar=True)
    #     self.log(f'Valid MCC', self.mcc/self.num_val_batches.to(self.device), prog_bar=True)
    #     #self.log(f'Valid Recall', self.recall.cpu().detach().numpy()/self.num_val_batches, prog_bar=True)
        
    #     # Resetting the metrics for upcoming epoch
    #     self.iou_score = torch.tensor(0)
    #     self.f1score = torch.tensor(0)
    #     #self.f2_score = torch.tensor(0)
    #     self.accuracy = torch.tensor(0)
    #     #self.recall = torch.tensor(0)
    #     self.dice_score = torch.tensor(0)
    #     self.mcc = torch.tensor(0)
    #     self.val_epoch_loss = torch.tensor(0)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, split='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,  T_0 = 1060, verbose=True) # T_0 = number of steps in ten epochs
        scheduler = CosineAnnealingLR(optimizer, 20, eta_min=1e-8, last_epoch=- 1, verbose=True) # T_0 = number of steps in ten epochs
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        #return optimizer

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return valid_loader

    # def test_dataloader(self):
    #     return test_loader


# def eval_acc(debug_name=None):
#     device='cuda' if torch.cuda.is_available() else 'cpu'
#     # model = Net(9)
#     model.load_state_dict(torch.load("/home/farhan/audio_classification/audio_classification_dir/lightning_logs/version_0/checkpoints/epoch=41-step=6090.ckpt")["state_dict"], strict=False)
#     #model = torch.load("lightning_logs/version_0/checkpoints/epoch=41-step=6090.ckpt")
#     model.eval()
#     model = model.to(device).eval()
#     count = correct = 0

#     # Loading data
#     test_lst = []
#     test_labels = []
#     val_batch_size = 16
#     load_data_and_labels(r"/home/farhan/audio_classification/audio_classification_dir/ali_test_data",test_lst, test_labels)
#     test_dataset = MyDataset(root = r"/home/farhan/audio_classification/audio_classification_dir/data/test", filenames=test_lst, labels = test_labels, transforms=False)
#     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size, pin_memory=True)
    

#     for X, gt in test_dataloader:
#         logits = model(X.to(device))
#         preds = torch.argmax(logits, dim=1)
#         print(preds)
#         print(gt)
#         correct += sum(preds.cpu() == gt)
#         count += len(gt)
#     acc = correct/count
#     if debug_name:
#         print(f'{debug_name} acc = {acc:.4f}')
#     return acc

if __name__ == '__main__':
    
    ENCODER = "tu-hrnet_w30"
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Initializing the dataset objects for train, valid, and test cohorts
    train_dataset = FATDataset(img_path = "fat_data/fat_detection/img_dir/train", mode = "Training", preprocessing=get_preprocessing(preprocessing_fn))
    val_dataset = FATDataset(img_path = "fat_data/fat_detection/img_dir/val", mode = "Validation", preprocessing=get_preprocessing(preprocessing_fn))
    # test_dataset = MyDataset(root = r"/home/farhan/audio_classification/audio_classification_dir/data/test", filenames=test_lst, labels = test_labels, transforms=False)


    #Initializing the data loaders
    train_batch_size = 32
    val_batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size, pin_memory=True)

    PARAMS = {"learning_rate": 1e-3, "optimizer": "Adam", "batch size": "=32, 16", "Loss func":"Weighted cross entropy", "Dropouts": "No", "Callbacks":"Checkpoint,  Lr_Monitor, Early_Stopping", "scheduler":'Yes', 'encoder':'tu-hrnet_w30(mediseg)'}

    with open("hyperparams.json", "w") as outfile:
        json.dump(PARAMS, outfile)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs")

    # Initializing the pytorch lightning learner
    learner = MyLearner(train_batches = len(train_dataset)/train_batch_size, val_batches = len(val_dataset)/val_batch_size)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='Validation Loss')
    early_stopping = EarlyStopping(monitor = 'Validation Loss', patience=20, verbose = True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator='gpu', devices = 2, strategy = 'dp', max_epochs=700, callbacks=[checkpoint, lr_monitor, early_stopping], log_every_n_steps= (len(train_dataset)/train_batch_size+len(val_dataset)/val_batch_size), logger=tb_logger) #,  logger=neptune_logger,  resume_from_checkpoint="/home/farhan/audio_classification/audio_classification_dir/lightning_logs/lightning_logs/version_2/checkpoints/epoch=138-step=10147.ckpt"
    trainer.fit(learner)

    # # Check evaluation accuracy 
    # eval_acc(debug_name=True)