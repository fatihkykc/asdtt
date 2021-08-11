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
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, RandomBrightnessContrast,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise, RandomCrop)
import albumentations as A
from albumentations.pytorch import ToTensorV2



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--final', '-f', help="isFinal? t or f", type=str)
parser.add_argument('--name', '-n', help="name of the input model file", type= str)
parser.add_argument('--arc', '-a', help="architecture of the input model file, e.g:resnet18", type= str)
parser.add_argument('--iter', '-i', help="output submission file number", type= int)
# parser.add_argument('--name', '-n', help="name of the input model file", type= str)
args = parser.parse_args()

unet_rn18_final = sm.Unet(args.arc, classes=4, activation=None, in_channels=3)

# Initialize mode and load trained weights
def weight_loader(model,path):
    ckpt_path = path
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    return model

unet_rn18_final = weight_loader(unet_rn18_final,args.name+'.pth')
unet_rn18_final.eval()

class Model:
    def __init__(self, models):
        self.models = models
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        x = torch.flip(x,dims = [-1])
        with torch.no_grad():
            for m in self.models:
                flipped_mask = m(x)
                mask = torch.flip(flipped_mask,dims = [-1])
                res.append(mask)
        res = torch.stack(res)
        return torch.sigmoid(torch.mean(res, dim=0))

model = Model([unet_rn18_final])

if args.final=='t':
    sample_submission_path = 'data/sample_submission_2nd.csv'
    test_data_folder =  "data/Testing_2nd_Imgs/Testing_2nd_Imgs"
else:
    sample_submission_path = 'data/sample_submission.csv'
    test_data_folder =  "data/Testing_Imgs/Testing_Imgs"

class TestDataset(Dataset, ):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std, resize=False):
        self.root = root
        df['ImageId'] = df['filename_class'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        if resize:
            self.transform = Compose(
            [   
                Resize(512,512),
                ToTensorV2(),
            ]
        )
        else:
            self.transform = Compose(
                [   
                    ToTensorV2(),
                ]
            )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname+".jpeg")
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples
    
df = pd.read_csv(sample_submission_path)
ts = DataLoader(
    TestDataset(test_data_folder, df, None, None, False),
    batch_size=1,
    shuffle=False,
    num_workers=10,
    pin_memory=True
)
orgsizes=[]
for i in ts:
    orgsizes.append((i[1].shape[3], i[1].shape[2]))
orsz = [val for val in orgsizes for _ in range(4)]

def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((512, 512), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num



best_threshold = 0.8
num_workers = 10
batch_size = 1
min_size = 400
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean=None, std=None, resize=True),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

from mask_funcs import rle2mask, mask2rle


def null_augment   (input): return input
def flip_lr_augment(input): return torch.flip(input, dims=[2])
def flip_ud_augment(input): return torch.flip(input, dims=[3])

def null_inverse_augment   (logit): return logit
def flip_lr_inverse_augment(logit): return torch.flip(logit, dims=[2])
def flip_ud_inverse_augment(logit): return torch.flip(logit, dims=[3])

augment = (
        (null_augment,   null_inverse_augment   ),
        (flip_lr_augment,flip_lr_inverse_augment),
        (flip_ud_augment,flip_ud_inverse_augment),
    )

model = unet_rn18_final
device='cuda'
predictions = []
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch
    #print('images', images.shape)
    images = images.cuda()
    batch_preds = 0
    probabilities = []
#     model = model.cuda()
    for k, (a, inv_a) in enumerate(augment):
#         print(a(images).shape)
        logit = model(a(images).float())
        p = inv_a(torch.sigmoid(logit))

        if k ==0:
            probability  = p**0.5
        else:
            probability += p**0.5
    probability = probability/len(augment)
    probabilities.append(probability)

    batch_preds+=probability
    
    batch_preds = batch_preds.data.cpu().numpy()
    #print(batch_preds.shape)
    for fname, preds in zip(fnames, batch_preds):
        for cls, pred in enumerate(preds):
            #print(cls)
            pred, num = post_process(pred, best_threshold, min_size)
            rle = mask2rle(pred)
            name = fname + f"_{cls+1}"
            predictions.append([name, rle])


df_segmentation = pd.DataFrame(predictions, columns=['filename_class', 'encoded_mask'])


pred_masks = []
for i in range(len(predictions)):
    pred_masks.append(rle2mask(predictions[i][1], (512,512)))
    
pred_rle = []
for idx, p in enumerate(pred_masks): 
    img = cv2.resize(p, orsz[idx])
    pred_rle.append(mask2rle(img))
    
masks = []
for idx, p in enumerate(pred_masks): 
    img = cv2.resize(p, orsz[idx])
    masks.append(img)

rles = []
for idx, p in enumerate(masks): 
    rles.append(mask2rle(p))
df_segmentation['encoded_mask'] = rles

submit = pd.read_csv(sample_submission_path, converters={'encoded_mask': lambda e: ' '})
df_segmentation['filename_class'] = df_segmentation['filename_class'].apply(lambda x: x.split('_')[0]+"_id"+x.split('_')[1])
dfs = df_segmentation.merge(submit, on='filename_class', how='right', suffixes=['', 'a'])
dfs = dfs.drop('encoded_maska', axis=1)

dfs.to_csv(args.final+'sub'+str(i)+args.name+'.csv', index=False)