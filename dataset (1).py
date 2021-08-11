from torch.utils.data import DataLoader, Dataset, sampler
from albumentations import (HorizontalFlip, RandomBrightnessContrast,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise, RandomCrop)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from mask_funcs import mask2rle, make_mask
import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
class AnadoluDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, image_id+".jpeg")
        img = cv2.imread(image_path)
        augmented = self.transforms(image=np.array(img), mask=np.array(mask))
        img = augmented['image']
        mask = augmented['mask'] # 1x256x1600x4
        mask = mask.permute(2, 0, 1) # 4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
#     normalize = Normalize(mean=mean, std=std)
    if phase == "train":
        list_transforms.extend(
            [
                Resize(512,512),
                HorizontalFlip(p=0.5),
                A.augmentations.transforms.RandomSnow(),
                A.ElasticTransform(),
                A.RandomRotate90(p=0.5),
                A.RandomGridShuffle(p=0.5),
                A.Cutout(num_holes=12, p=0.5),
#                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ToTensorV2(),
            ]
        )
    list_transforms.extend(
        [
            Resize(512,512),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms