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

class trainning():
    def __init__(self, model, trainloader, device):
        self.model = model
        self.trainloader = trainloader
        self.device = device
        
    def optim(self):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=1e-9, momentum=0.5, weight_decay=1e-4)
        
        for epoch in range(100):
        
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels = data
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                labels = labels.to(torch.float)
                # print(f'type_labels: {type(labels)}')
                # labels = [labels]
                # print(f'labels: {labels}')

                # print(f'inputs: {type(inputs)}')
                inputs = torch.permute(inputs, (0, 2, 1, 3))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                # outputs = outputs[0:2]
                # print(f'outputs: {outputs}')
                loss = criterion(outputs, labels)
                # print(f'outputs.dtype: {outputs.dtype}')
                # print(f'labels.dtype: {labels.dtype}')
                # print(f'loss.dtype; {loss.dtype}')
                
                # l1_lambda = 1e-4  #L1正則化の係数
                l2_lambda = 1e-7  # L2正則化の係数
                # l1_regularization = sum(torch.norm(param, p = 1) for param in self.model.parameters())
                l2_regularization = sum(torch.norm(param, p=2) for param in self.model.parameters())
                # loss += l1_lambda * l1_regularization
                loss += l2_lambda * l2_regularization
                
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                    print()
                    running_loss = 0.0

class inference():
    def __init__(self, model, device, dataset):
        self.model = model
        self.device = device
        self.dataset = dataset
        
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
                    average += labels
            average = average / coefficient
            for data in self.dataset:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                for label2 in labels:
                    denominator += (label2 - average)**2
        
        r2_score = 1 - numerator / denominator
        
        with open(file_path, 'a') as f:
            f.write(f'R2: {r2_score}')
        

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
    
    # train
    train = trainning(model, trainloader, device)
    train.optim()
    
    # save param
    checkpoints_path = './output/checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)
    param_path = f'{checkpoints_path}/vgg16_param.pth'
    torch.save(model.state_dict(), param_path)
    
    # inference
    result_path = './output/accuracy'
    os.makedirs(result_path, exist_ok=True)
    result_file = f'{result_path}/accuracy.txt'
    with open(result_file, 'w') as f:
        f.write('train accuracy')
    train_inf = inference(model, device, trainloader)
    train_inf.accuracy(result_file)
    with open(result_file, 'a') as f:
        f.write('test accuracy')
    test_inf = inference(model, device, testloader)
    test_inf.accuracy(result_file)
    
    


if __name__=='__main__':
    main()
