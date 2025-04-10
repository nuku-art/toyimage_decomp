import torch
import torchvision
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import csv

from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset

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
    

class DataLoder():
    def __init__(self, batch_size, annotations_file, img_dir) -> None:
        self.batch_size = batch_size
        self.annotations_file = annotations_file
        self.img_dir = img_dir

    def load(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)])
        data = MyDataset(annotations_file=self.annotations_file, img_dir=self.img_dir,
                             transform=transform)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True)

        return loader

class save_output:
    def __init__(self, model, target_layer) -> None:
        self.model = model
        self.layer_output = []
        self.layer_grad = []
        self.feature_handle = target_layer.register_forward_hook(self.feature)
        self.grad_handle = target_layer.register_forward_hook(self.gradient)
    
    def feature(self, model, input, output):
        activation = output
        self.layer_output.append(activation.to('cpu').detach())
        
    def gradient(self, model, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        def _hook(grad):
            self.layer_grad.append(grad.to('cpu').detach())
        
        output.register_hook(_hook)
        
    def release(self):
        self.feature_handle.remove()
        self.grad_handle.remove()
    
class re_trans():
    def __init__(self) -> None:
        self.std = [0.5]
        self.mean = [0.5]

    def transform(self, image):  # image: np.ndarray
        index = image.shape
        for i in range(index[0]):
            for j in range(index[1]):
                for k in range(index[2]):
                    image[i, j, k] = image[i, j, k] * self.std[i] + self.mean[i]
        return_image = image

        return return_image

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    model = torchvision.models.vgg16()
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    model.classifier[3] = torch.nn.Linear(4096, 1000)
    model.classifier[6] = torch.nn.Linear(1000, 1)
    param_path = './output/checkpoints_0430/vgg16_param.pth'
    model.load_state_dict(torch.load(param_path))
    model.eval().to(device)
    
    # register hook
    target_layer = 28
    save = save_output(model=model, target_layer=model.features[target_layer])

    # load data
    test_annotations_file = './output/latent_image_pred1303.238525390625/image/label.csv'
    test_img_dir = './output/latent_image_pred1303.238525390625/image'
    batch_size = 1
    Loader = DataLoder(batch_size=batch_size, annotations_file=test_annotations_file,
                       img_dir=test_img_dir)
    testloader = Loader.load()
    
    # data sampling
    use_data = iter(testloader)
    image, label = next(use_data)
    # image = torch.permute(image, (0, 2, 1, 3))      # first time
    image = torch.permute(image, (0, 2, 3, 1))        # after
    output = model(image.to(device))
    

    path = f'./output/latent_image_pred{output.item()}'
    # os.makedirs(path, exist_ok=True)
    '''
    image_dir = f'{path}/image'
    os.makedirs(image_dir, exist_ok=True)
    
    # savebase image
    np_image = np.array(image)
    back_trans = re_trans()
    base_array = back_trans.transform(np_image[0])
    base_array = np.squeeze(base_array)
    # print(f'base_array: {base_array.shape}')
    base_im = Image.fromarray(np.uint8(base_array*255))
    base_path = f'{image_dir}/base_image.png'
    base_im.save(base_path)
    
    # save csv
    image_csv = f'{image_dir}/label.csv'
    with open(image_csv, 'w') as trf:
        writer = csv.writer(trf)
        writer.writerow(['id', 'label'])
    with open(image_csv, 'a') as tef:
                writer = csv.writer(tef)
                writer.writerow([base_path, label])
    '''

    # show latent image
    latent_bectors = np.array(save.layer_output[0].squeeze())
    # print(f'latent_bector: {latent_bector.shape}')
    latent_dir_path = f'{path}/layer{target_layer}'
    os.makedirs(latent_dir_path, exist_ok=True)
    for i in range(len(latent_bectors)):
        channel_num = i
        latent_bector = latent_bectors[channel_num]
        latent_bector = latent_bector / np.amax(latent_bector)
        # print(f'latent_bector: {latent_bector.shape}')
        latent_image = Image.fromarray(np.uint8(latent_bector*255))
        os.makedirs(path, exist_ok=True)
        im_path = f'{latent_dir_path}/image_channel{channel_num}.png'
        latent_image.save(im_path)


    
if __name__=='__main__':
    main()
    






