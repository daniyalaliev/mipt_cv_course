import os
from typing import Any
import torch
import torchvision
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as T
#import matplotlib.pyplot as plt
from tqdm import tqdm


df = pd.read_csv('tests/00_test_img_input/train/gt.csv')

print(np.array(df.values.tolist())[:, 1:])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.maxpool3 = nn.MaxPool2d(2)
        self.linear1 = None
        self.linear2 = None
        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(256)

    
    def forward(self, img):
        res = self.conv1(img)
        res = self.bn1(res)
        res = self.relu(res)
        # res = self.dropout(res)
        res = self.maxpool1(res)
        res = self.conv2(res)
        #res = self.bn2(res)
        res = self.relu(res)
        # res = self.dropout(res)
        res = self.maxpool2(res)
        res = self.conv3(res)
        #res = self.bn3(res)
        res = self.relu(res)
        res = self.dropout(res)
        res = self.maxpool3(res)
        res = torch.flatten(res, start_dim=1)
        if not self.linear1:
            self.linear1 = nn.Linear(res.shape[1], 64)
            self.linear2 = nn.Linear(64, 28)
        res = self.linear1(res)
        res = self.relu(res)
        res = self.linear2(res)
        return res

class Faces_Dataset(Dataset):
    def __init__(self, train_gt, train_img_dir):
        super().__init__()
        self.res = np.array(train_gt.values.tolist())[:, 1:]
        self.res = torch.from_numpy(self.res.astype(np.double)).float()
        self.train_img_dir = train_img_dir
        self.resize = T.Resize(size=(100, 100))

    def __len__(self):
        return len(self.res)

    def __getitem__(self, index):
        img = torchvision.io.read_image(os.path.join(self.train_img_dir, \
                                                     '0' * (5 - len(str(index))) + str(index) + '.jpg'))
        points = self.res[index]
        points[::2] = points[::2] * 100/img.shape[-2]
        points[1::2] = points[1::2] * 100/img.shape[-1]
        img = self.resize(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img.float(), points

train_dir = 'tests/00_test_img_input/train'
train_gt = pd.read_csv(os.path.join(train_dir, 'gt.csv'))
train_img_dir = os.path.join(train_dir, 'images')
# dataset = Faces_Dataset(train_gt, train_img_dir)
# plt.imshow(dataset[5][0].permute(1, 2, 0))
# plt.show()

def train_detector(train_gt, train_img_dir, fast_train=True):
    model = Model()
    dataset = Faces_Dataset(train_gt, train_img_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    loss = torch.nn.MSELoss()
    for _ in range(10):
        average = 0
        for image, target in tqdm(train_dataloader):
            optimizer.zero_grad()
            output = model(image)
            ls = loss(output, target)
            ls.backward()
            optimizer.step()
            average += ls.item()
            print(ls.item()/train_dataloader.batch_size)
        print(average/len(train_dataloader))

train_detector(train_gt, train_img_dir)
    