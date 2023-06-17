import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import cv2


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('data/train_data'))

    def __getitem__(self, idx):
        img_name = os.listdir('data/train_data')[idx]
        imgA = cv2.imread('data/train_data/'+img_name)
        imgA = cv2.resize(imgA, (160, 160))
        imgB = cv2.imread('data/label_data/'+img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)
        return imgA, imgB

bag = BagDataset(transform)

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
