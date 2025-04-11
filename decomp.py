import torch
import torchvision
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import datetime

from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
from captum.attr import IntegratedGradients
from src.devide_latent import process_latent
from src.image_decomposition import ImageDecomp

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
    
YMD = str(datetime.date.today()).replace('-','')
base_dir = f'/workspace/output/{YMD}'
os.makedirs(base_dir, exist_ok=True)

log_file_path = f'{base_dir}/inference_log.txt'
with open(log_file_path, mode='w') as f:
  f.write('test')
    
# load data
transform = transforms.Compose(
    [transforms.ToTensor(),
      transforms.Normalize(0.5, 0.5)])
batch_size = 1
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

# define model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = torchvision.models.vgg16()
net.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
net.classifier[3] = torch.nn.Linear(4096, 1000)
net.classifier[6] = torch.nn.Linear(1000, 1)
net.to(device)

# load param
param_path = '/workspace/output/checkpoints_0430/vgg16_param.pth'
net.load_state_dict(torch.load(param_path, weights_only=True))
net.eval()
top_model = nn.Sequential(*list(net.features.children()))
avg_model = nn.Sequential(*list(net.avgpool.children()))
bottom_model = nn.Sequential(*list(net.classifier.children()))

# pick data
for i, data in enumerate(testloader, 0):
  image, label = data[0].to(device), data[1].to(device)
  break
image = torch.permute(image, (0, 2, 1, 3))
with torch.no_grad():
  output = net(image)
with open(log_file_path, mode='a') as f:
    f.write(f'output: {output}')
    f.write(f'label: {label}')

detach_image = image.clone().detach().to('cpu').squeeze()
detach_image = detach_image.mul_(0.5).add_(0.5) * 255
np_image = detach_image.numpy().astype(np.uint8)
im = Image.fromarray(np_image)
im.save(f'{base_dir}/base_image.png')

with torch.no_grad():
  latent_vector = top_model(image)
  bottom_input = avg_model(latent_vector.clone())
  flatten_input = torch.flatten(bottom_input.clone(), 1)
  pred = bottom_model(flatten_input.clone())
with open(log_file_path, mode='a') as f:
    f.write(f'latent_vector: {latent_vector.shape}')
    f.write(f'decomp pred: {pred}')

# calcurate attribution
avg_latent = avg_model(latent_vector.clone())
bottom_input = torch.flatten(avg_latent, 1)
with torch.no_grad():
  pred = bottom_model(bottom_input.clone())

base_line = torch.zeros(bottom_input.shape).to(device)
ig = IntegratedGradients(bottom_model)
attributions, approximation_error = ig.attribute(bottom_input, baselines=base_line,method='gausslegendre', return_convergence_delta=True)
attributions = torch.reshape(attributions, latent_vector.shape)
channel_attribution = attributions.sum(dim=(2,3)).to('cpu').detach().squeeze().numpy()
attribution_horizontal = np.arange(channel_attribution.shape[0])
plt.bar(attribution_horizontal, channel_attribution)
plt.savefig(f'{base_dir}/attribution.png')

# devide latent
component_num = 1

process = process_latent(device)
latent_tensor = process.devide(latent_vector, channel_attribution, component_num)

sum_latent = torch.zeros(latent_vector.shape)
for i in latent_tensor:
  sum_latent += i
devide_loss = torch.norm(latent_vector.clone().to('cpu').detach()-sum_latent)
with open(log_file_path, mode='a') as f:
    print(f'devide loss: {devide_loss}')
    print(latent_vector.shape) 
    print(latent_tensor.shape)

image_dir = f'{base_dir}/decomp_image'
os.makedirs(image_dir, exist_ok=True)
decomposer = ImageDecomp(top_model, device)
decomposer.optim(image, latent_tensor, base_dir, image_dir)