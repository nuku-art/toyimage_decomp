import torch
import torchvision
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from torchvision.io import read_image
from torch.utils.data import Dataset
    
class inference():
    def __init__(self, model, device, dataset):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.r2 = 0
        
    def accuracy(self, file_path):
        self.model.eval()  
        # prepare to count predictions for each class
        numerator = 0
        denominator = 0
        coefficient = 0
        average = 0
        
        # accuracy
        with torch.no_grad():
            for data in self.dataset:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                images = torch.permute(images, (0, 2, 1, 3))
                outputs = self.model(images)
                for output, label in zip(outputs, labels):
                    numerator += (label - output)**2
                    coefficient += 1
                    average += label
            
            # print(f'outputs: {outputs.shape}')
            # print(f'labels: {labels.shape}')
            # print(f'average: {average.shape}')
            # print(f'numertor: {numerator.shape}')

            average = average / coefficient

            for data in self.dataset:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                for label2 in labels:
                    denominator += (label2 - average)**2
            # print(f'denominator: {denominator.shape}')
        
        self.r2 = 1 - numerator.item() / denominator.item()
        
        with open(file_path, 'a') as f:
            f.write(f'R2: {self.r2}\n')

class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        # print(f'__getitem__ image: {image.shape}')
        image = np.array(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def main():
    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.vgg16()
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    model.classifier[3] = torch.nn.Linear(4096, 1000)
    model.classifier[6] = torch.nn.Linear(1000, 1)
    param_path = './output/checkpoints_0430/vgg16_param.pth'
    model.load_state_dict(torch.load(param_path))
    model.to(device)
    
    # load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 16
    
    train_annotations_file = './dataset/train/label.csv'
    train_img_dir = './dataset/train'
    train_data = MyDataset(annotations_file=train_annotations_file, img_dir=train_img_dir,
                             transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                              shuffle=True)
    
    test_annotations_file = './dataset/test/label.csv'
    test_img_dir = './dataset/test'
    test_data = MyDataset(annotations_file=test_annotations_file, img_dir=test_img_dir,
                             transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # inference
    result_path = './output/accuracy'
    os.makedirs(result_path, exist_ok=True)
    result_file = f'{result_path}/accuracy.txt'
    with open(result_file, 'w') as f:
        f.write('train accuracy\n')
    train_inf = inference(model, device, trainloader)
    train_inf.accuracy(result_file)
    with open(result_file, 'a') as f:
        f.write('\ntest accuracy\n')
    test_inf = inference(model, device, testloader)
    test_inf.accuracy(result_file)
    
    


if __name__=='__main__':
    main()
