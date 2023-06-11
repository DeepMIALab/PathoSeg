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
from utils import get_preprocessing
from .pathoseg import UnetPlusPlus
from .Pytorch_learner import MyLearner

def convert_to_rgb(mask, img_name, inference_path, dataset_name):
    """Function to convert typemaps to rgb masks"""
    if dataset_name == 'Prostate':
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
        red = mask == 1
        blue = mask == 2
        green = mask == 3
        yellow = mask == 4

        rgb_mask[red] = [255,0,0]
        rgb_mask[blue] = [0,0,255]
        rgb_mask[green] = [0,255,0]
        rgb_mask[yellow] = [255,255,0]
        cv2.imwrite(cv2.imwrite(os.path.join(inference_path+'/'+dataset_name, img_name), rgb_mask[:,:,[2,1,0]]))

    else:
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
        print(cv2.imwrite(os.path.join(inference_path+'/'+dataset_name, img_name), rgb_mask[:,:,[2,1,0]]))


def eval_acc(debug_name=None, args=None):
    ENCODER = args.encoder
    ENCODER_WEIGHTS = args.encoder_weights
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = MyLearner(num_classes=args.num_classes, in_channels= args.in_channels).load_from_checkpoint(args.ckpt)
    model = model.eval().to('cuda')
    test_imgs = os.listdir(args.data_path)

    for img in test_imgs:
        with torch.no_grad():
                img = cv2.imread(os.path.join(args.data_path, img), cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = torchvision.transforms.functional.to_tensor(img)
                img = img[None,:]
                print(img.shape)
                logits = model(img.to(device))
                preds = torch.argmax(logits, dim=1).cpu()
                print(preds.shape)
                convert_to_rgb(preds, img, args.inference_path, args.dataset_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PathoSeg')

    parser.add_argument('--encoder', type=str, default='tu-hrnet_w30', help='encoder to use for the segmentation model')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='weight to intialize the encoder')
    parser.add_argument('--data_path', type=str, required=True, help='path to the base dataset folder')
    parser.add_argument('--train_batchsize', type=int, default=64, help='training batch size')
    parser.add_argument('--validation_batchsize', type=int, default=32, help='validation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='the iteration to start training')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes in the dataset')
    parser.add_argument('--max_epochs', type=int, default=700, help='total epochs for training')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint to perform inference on the test data')
    parser.add_argument('--gpus', type=int, default=8, help='number of gpus')
    parser.add_argument('--in_channels', type=int, default=1, help='Channels of the input image to the model')
    parser.add_argument('--inference_path', type=str, required=True, help='Directory to store the model inference')
    parser.add_argument('--dataset_type', type=str, choices=['Prostate', 'Fat', 'Tubule'], required=True, help='Dataset type')

    args = parser.parse_args()
    print(args)

    # Check evaluation accuracy 
    eval_acc(debug_name=True, args=args)