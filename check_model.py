import torchvision
import torch
import torch.nn as nn

def main():
    
    model = torchvision.models.vgg16()
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    model.classifier[3] = torch.nn.Linear(4096, 1000)
    model.classifier[6] = torch.nn.Linear(1000, 1)
    print(model)
    '''
tensor = [[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
          [9, 10, 11]]
tensor2 = tensor[1:3]
print(tensor2)
'''

if __name__=='__main__':
    main() 