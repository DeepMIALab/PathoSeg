import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
import glob2
import torchvision


def normalize_input_tensor(t, img_path):
    """Normalize the input tensor for the network"""
    # eps = 0.001
    for channel in range(3):
        print(t[0].shape)
        mean, std, var = torch.mean(t[channel]), torch.std(t[channel]), torch.var(t[channel])
        print("Mean, Std and Var before Normalize:\n", 
          mean, std, var)
        if std<=0.0:
            print(img_path)
        # Step 4: Normalizing the tensor
        t[channel]  = (t[channel]-mean)/(std) #+eps
        # Step 5: Again compute the mean, std and variance
        # after Normalize
        mean, std, var = torch.mean(t[channel]), torch.std(t[channel]), torch.var(t[channel])
        print("Mean, std and Var after normalize:\n", 
          mean, std, var)
        # print("Tensor",t.shape)
    return t

def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

class GenericDataset(Dataset):
    def __init__(self, img_path, transform = None, mode ='Training', preprocessing=None):
        self.img_list = []                # List containing img paths 
        self.mask_list = []               # List containing mask paths 
        self.img_path = img_path
        self.mode = mode.strip()
        self.transform = transform
        self.preprocessing = preprocessing

        if self.mode=="Training":
            for image_path in glob2.glob(self.img_path+"/*.png"):  #data/fat_detection/img_dir/train/*.png
                self.img_list.append(image_path)
                mask_path = image_path.replace("img_dir","ann_dir")
                self.mask_list.append(mask_path)
                
        elif self.mode=="Validation":
            for image_path in glob2.glob(os.path.join(self.img_path, "*.png")):
                self.img_list.append(image_path)
                mask_path = image_path.replace("img_dir","ann_dir")
                self.mask_list.append(mask_path)

        elif self.mode=="Test":
            for image_path in glob2.glob(os.path.join(self.img_path, "*.png")):
                self.img_list.append(image_path)
                mask_path = image_path.replace("img_dir","ann_dir")
                self.mask_list.append(mask_path)

        print("Data Distribution For "+mode+" Phase")
        print("Images", len(self.img_list))
        print("Masks", len(self.mask_list))


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.img_list[index].split("/")[-1]      # Return the name of the image and mask i.e 10.png
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)

        # img = Image.open(img_path).convert('RGB')
        # mask = Image.open(mask_path).convert('L')
        # print(img.size)

        if self.transform:
            state = torch.get_rng_state()
            sample = self.transform(image = img, mask=mask)
            img, mask = sample['image'], sample['mask']
            torch.set_rng_state(state)
            #mask = self.transform(mask)
            
        img = torchvision.transforms.functional.to_tensor(img)
        return img, mask