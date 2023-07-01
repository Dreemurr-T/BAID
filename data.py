import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from common import *
import cv2 as cv
import time
import os
from tqdm import tqdm
import pandas as pd

mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]


class PretrainingDataset(Dataset):
    def __init__(self, path_csv=None, opt_csv=None) -> None:
        print("Start loading data into Mem")
        paths = pd.read_csv(path_csv)
        opts = pd.read_csv(opt_csv)
        self.original_paths = paths['origin'].values.tolist() * 3
        self.degraded_paths = paths['opt1'].values.tolist() + paths['opt2'].values.tolist() + paths['opt3'].values.tolist()
        self.trp_paths = paths['trp1_1'].values.tolist() + paths['trp2_1'].values.tolist() + paths['trp3_1'].values.tolist()
        self.trp2_paths = paths['trp1_2'].values.tolist() + paths['trp2_2'].values.tolist() + paths['trp3_2'].values.tolist()
        self.opt_list = opts['opt1'].values.tolist() + opts['opt2'].values.tolist() + opts['opt3'].values.tolist()

        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ]
        )

        print("Data Loaded!")

    def __len__(self):
        return len(self.original_paths)

    def __getitem__(self, index):
        original_image = cv.imread(self.original_paths[index])
        degraded_image = cv.imread(self.degraded_paths[index])
        trp_1 = cv.imread(self.trp_paths[index])
        trp_2 = cv.imread(self.trp2_paths[index])

        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        degraded_image = cv.cvtColor(degraded_image, cv.COLOR_BGR2RGB)
        trp_1 = cv.cvtColor(trp_1, cv.COLOR_BGR2RGB)
        trp_2 = cv.cvtColor(trp_2, cv.COLOR_BGR2RGB)

        opts = self.opt_list[index]

        original_image = self.transformer(original_image)
        degraded_image = self.transformer(degraded_image)
        trp_1 = self.transformer(trp_1)
        trp_2 = self.transformer(trp_2)

        return original_image, opts, degraded_image, trp_1, trp_2


class BBDataset(Dataset):
    def __init__(self, file_dir='dataset', type='train', test=False):
        self.if_test = test
        self.train_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.test_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.images = []
        self.pic_paths = []
        self.labels = []

        if type == 'train':
            DATA = pd.read_csv(os.path.join(file_dir, 'train_set.csv'))
        elif type == 'validation':
            DATA = pd.read_csv(os.path.join(file_dir, 'val_set.csv'))
        elif type == 'test':
            DATA = pd.read_csv(os.path.join(file_dir, 'test_set.csv'))

        labels = DATA['score'].values.tolist()
        pic_paths = DATA['image'].values.tolist()
        for i in tqdm(range(len(pic_paths))):
            pic_path = os.path.join('images', pic_paths[i])
            label = float(labels[i] / 10)
            self.pic_paths.append(pic_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.pic_paths)

    def __getitem__(self, index):
        pic_path = self.pic_paths[index]
        img = cv.imread(pic_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.if_test:
            img = self.test_transformer(img)
        else:
            img = self.train_transformer(img)

        return img, self.labels[index]