from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import cv2
import glob
import albumentations as album
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class ImageDataset(Dataset):
    def __init__(self, files, csv_file, transform):
        self.files = files
        self.csv = csv_file
        self.transform = transform
        self.as_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        data = self.csv
        img = cv2.imread(img_file, 0)
        perf = np.nanmedian(data[data['name']==os.path.basename(img_file)[:10]]['Resist']).astype(np.float32)

        if 'T' in os.path.basename(img_file):
            img = ((img - 138.8) * 0.6 + 142.7).astype('uint8')
        
        augments = self.transform(image=img)
        img = self.as_tensor(augments['image'])
        
        return os.path.basename(img_file), img, perf
    
class MakeLoader():
    def __init__(self):
        self.files = glob.glob('/workspace/dataset/rubber_data_2021/*/*.tif')
        self.csv_path = '/workspace/dataset/rubber_data_2021/func_data.csv'
    
    def loader(self, batch_size=1):
        csv_file = pd.read_csv(self.csv_path)

        train_transform = album.Compose([
            album.VerticalFlip(p=0.5),
            album.Rotate(limit=[-10, 10]),
            album.CenterCrop(height=512, width=256),
            album.RandomCrop(height=224, width=224),
        ])
        val_transform = album.Compose([
            album.CenterCrop(height=224, width=224),
        ])

        train_files, val_files = train_test_split(self.files, test_size=0.2)

        train_data = ImageDataset(files=train_files, csv_file=csv_file, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)    
        val_data = ImageDataset(files=val_files, csv_file=csv_file, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=4)
        return train_loader, val_loader