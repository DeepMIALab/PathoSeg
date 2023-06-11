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
import argparse
import math

# Custom packages
# from dataloader import FATDataset
# from utils import get_preprocessing
from .model import UnetPlusPlus
from .learner import MyLearner

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


def eval_acc(debug_name=None, encoder_name, encoder_weights, num_classes=2, in_channels = 3):
    ENCODER = "tu-hrnet_w30"
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = MyLearner(num_classes=num_classes, in_channels=in_channels).load_from_checkpoint("/truba/home/isahin/gaugan_pytorch/gaugan_tubule/lightning_logs/lightning_logs/version_7/checkpoints/epoch=110-step=43401.ckpt")
    model = model.eval().to('cuda')

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
    parser = argparse.ArgumentParser(description='PathoSeg ')

    parser.add_argument('--encoder', type=str, default='tu-hrnet_w30', help='encoder to use for the segmentation model')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='weight to intialize the encoder')
    parser.add_argument('--data_path', type=str, required=True, help='path to the base dataset folder')
    parser.add_argument('--train_batchsize', type=int, default=64, help='training batch size')
    parser.add_argument('--validation_batchsize', type=int, default=32, help='validation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='the iteration to start training')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes in the dataset')
    parser.add_argument('--max_epochs', type=int, default=700, help='total epochs for training')
    parser.add_argument('--gpus', type=int, default=8, help='number of gpus')
    # Check evaluation accuracy 
    eval_acc(debug_name=True, in_channels=2)