import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from matplotlib import image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as sm
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from loss_functions import DiceBCELoss
from mask_funcs import mask2rle, make_mask
from trainer import Trainer
from dataset import AnadoluDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', help="name of the output model file", type= str)
parser.add_argument('--arc', '-a', help="architecture of the input model file, e.g:resnet18", type= str)
parser.add_argument('--epochs', '-e', help="output submission file number", type= int)
parser.add_argument('--batch_size', '-bs', help="output submission file number", type= int)
parser.add_argument('--iter', '-i', help="output submission file number", type= int)
args = parser.parse_args()
print(args.epochs)

#path definitions
sample_submission_path = 'data/sample_submission.csv'
train_df_path = 'data/training.csv'
data_folder = 'data/Training_Imgs/Training_Imgs'
test_data_folder =  'data/Testing_Imgs/Testing_Imgs'

# create dataframe
tdf = pd.read_csv(train_df_path)
df = tdf.copy()
df['ImageId'], df['ClassId'] = zip(*df['filename_class'].str.split('_'))
df['ClassId'] = df['ClassId'].apply(lambda x: str(x)[-1]).astype(int)
df = df.pivot(index='ImageId',columns='ClassId',values='encoded_mask')
df['defects'] = df.count(axis=1)
df['image_height'] = tdf.iloc[::4, :]['image_height'].to_list()
df['image_width'] = tdf.iloc[::4, :]['image_width'].to_list()

#model from segmentationmodels
model = sm.Unet(args.arc, encoder_weights="imagenet", classes=4) 
#initialize training
model_trainer = Trainer(model, args.epochs, int(args.batch_size), args.arc+args.name+str(args.iter))
model_trainer.start()


# PLOT TRAINING
losses = model_trainer.losses
dice_scores = model_trainer.dice_scores # overall dice
iou_scores = model_trainer.iou_scores

print("dice: ", dice_scores)
print("iou: ", iou_scores)
print("losses: ", losses)
# def plot(scores, name):
#     plt.figure(figsize=(15,5))
#     plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
#     plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
#     plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');
#     plt.legend(); 
#     plt.show()


# plot(losses, "BCE+Dice loss")
# plot(dice_scores, "Dice score")
# plot(iou_scores, "IoU score")