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
import argparse
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
from dataloader import GenericDataset
from utils.utils import get_preprocessing
from pathoseg import UnetPlusPlus
from Pytorch_learner import MyLearner

def convert_to_rgb(mask, path):
    """Function to convert typemaps to rgb masks"""
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    white = mask == 1

    rgb_mask[white] = [255,255,255]
    cv2.imwrite(path, rgb_mask[:,:,[2,1,0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PathoSeg ')

    parser.add_argument('--encoder', type=str, default='tu-hrnet_w30', help='encoder to use for the segmentation model')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='weight to intialize the encoder')
    parser.add_argument('--data_path', type=str, required=True, help='path to the base dataset folder')
    parser.add_argument('--train_batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--validation_batchsize', type=int, default=16, help='validation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='the iteration to start training')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes in the dataset')
    parser.add_argument('--max_epochs', type=int, default=700, help='total epochs for training')
    parser.add_argument('--gpus', type=int, default=3, help='number of gpus')

    args = parser.parse_args()
    print(args)
    
    ENCODER = "tu-hrnet_w30"
    ENCODER_WEIGHTS = 'imagenet'
    # Loading the preprocessing function for the encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)

    # Training augmentations 
    train_transform = A.Compose([
    A.HorizontalFlip(p=0.40),
    A.augmentations.geometric.transforms.VerticalFlip(p=0.25),
    ])

    # Validation transformations
    # valid_transform = A.Compose([
    # A.HorizontalFlip(p=0.3),
    # # A.RandomBrightnessContrast(p=0.3)
    # ])

    # Initializing the dataset objects for train, valid, and test cohorts
    train_dataset = GenericDataset(img_path = args.data_path+"/train", mode = "Training", transform=train_transform)#, preprocessing=get_preprocessing(preprocessing_fn))
    val_dataset = GenericDataset(img_path = args.data_path+"/val", mode = "Validation")#,preprocessing=get_preprocessing(preprocessing_fn))
    # test_dataset = MyDataset(root = r"/home/farhan/audio_classification/audio_classification_dir/data/test", filenames=test_lst, labels = test_labels, transforms=False)


    #Initializing the data loaders
    train_batch_size = args.train_batchsize
    val_batch_size = args.validation_batchsize
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size, pin_memory=True)

    # PARAMS = {"learning_rate": args.learning_rate, "optimizer": "Adam", "batch size": str(args.training_batchsize)+','+ str(args.validation_batchsize), "Loss func":"Cross entropy", "Dropouts": "No", "Callbacks":"Checkpoint,  Lr_Monitor, Early_Stopping", "scheduler":'Yes(CosineAnnealing)', 'encoder':'tu-hrnet_w30(mediseg)', 'weight_decay':'1e-2', 'augs':' A.HorizontalFlip, A.VerticalFlip', 'Comments':  'Batch 8 and 4 '}

    # with open("hyperparams.json", "w") as outfile:
    #     json.dump(PARAMS, outfile)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs")

    # Initializing the pytorch lightning learner
    learner = MyLearner(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, train_batches = len(train_dataset)/train_batch_size, val_batches = len(val_dataset)/val_batch_size, num_classes=args.num_classes)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='Validation Loss')
    early_stopping = EarlyStopping(monitor = 'Validation Loss', patience=30, verbose = True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator='gpu', devices = args.gpus, strategy = 'dp', max_epochs=args.max_epochs, callbacks=[checkpoint, lr_monitor, early_stopping], log_every_n_steps= (len(train_dataset)/train_batch_size+len(val_dataset)/val_batch_size), logger=tb_logger) #,  logger=neptune_logger,  resume_from_checkpoint="/home/farhan/audio_classification/audio_classification_dir/lightning_logs/lightning_logs/version_2/checkpoints/epoch=138-step=10147.ckpt"
    trainer.fit(learner, train_loader, valid_loader)
