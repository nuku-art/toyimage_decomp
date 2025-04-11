import torch
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

class ImageDecomp():
    def __init__(self, model, device, W=None, clamp=True, epoch=10000):
        self.model = model
        self.device = device
        self.W = W
        self.clamp = clamp
        self.epoch = epoch
        self.mean = 0.5
        self.var = 0.5
        
        
    def visualize(self, image_tensor, path):  ## path: image save directry
        natural_sum = np.zeros([image_tensor.shape[2],image_tensor.shape[3]])
        pixel_sum = np.zeros([image_tensor.shape[2],image_tensor.shape[3]])
        for i in range(image_tensor.shape[0]):
            image = image_tensor[i].clone().squeeze()
            image = image.to('cpu').detach()
            image = image.numpy()
            print(f'numpy image: {image}')
            natural_sum += image.copy()
            image = image * 255
            image = np.clip(image, 0, 255)
            pixel_sum += image.copy()
            im = Image.fromarray(image.astype(np.uint8))
            im.save(f'{path}/x{i}_opt.png', 'PNG')
            del image
        sum_image = natural_sum.copy() * 255
        sum_image = np.clip(sum_image, 0, 255)
        sum_im = Image.fromarray(sum_image.astype(np.uint8))
        sum_im.save(f'{path}/natural_sum_image.png', 'PNG')
        print(f'pixel_sum: {pixel_sum}')
        pixel_sum_im = Image.fromarray(pixel_sum.astype(np.uint8))
        pixel_sum_im.save(f'{path}/pixel_sum_image.png', 'PNG')

    def non_clop_visualize(self, image_tensor, path):
        natural_sum = np.zeros([image_tensor.shape[2],image_tensor.shape[3]])
        max_val = torch.max(torch.abs(image_tensor)).item()
        for i in range(image_tensor.shape[0]):
            image = image_tensor[i].clone().squeeze()
            image = image.to('cpu').detach()
            image = image.numpy()
            natural_sum += image.copy()
            max_val = np.max(np.abs(image.copy()))
            scaled_image = image.copy() / max_val
            colormap = cm.coolwarm((scaled_image.copy() + 1) / 2)
            colormap[:, :, :3] *= 2.0
            im = Image.fromarray((colormap[:, :, :3] * 255).astype(np.uint8))
            im.save(f'{path}/noncrop_x{i}_opt.png', 'PNG')
        sum_image = natural_sum.copy() * 255
        sum_image = np.clip(sum_image, 0, 255)
        sum_im = Image.fromarray(sum_image.astype(np.uint8))
        sum_im.save(f'{path}/noncrop_natural_sum_image.png', 'PNG')


    def optim(self, image, latent_tensor, base_dir, image_dir):  
        component_num = latent_tensor.shape[0]
        decomp_list = [image.clone() for _ in range(component_num)]
        decomp_image = torch.stack(decomp_list, dim=0)
        decomp_image = decomp_image.reshape(component_num, image.shape[1], image.shape[2], image.shape[3])
        decomp_image = decomp_image.clone().detach().requires_grad_(True)

        if latent_tensor.device != self.device:
            latent_tensor = latent_tensor.to(self.device)
        
        optimizer = optim.Adam([decomp_image], lr=0.01, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.995)
        
        decomploss_list = []
        sumloss_list = []
        normloss_list = []
        
        ## decide lower and upper
        # lower = torch.zeros(image.shape[1],image.shape[2],image.shape[3]).to(self.device)
        # zero_upper = torch.ones(image.shape[1],image.shape[2],image.shape[3]).to(self.device)
        # decomp_upper = image.clone().reshape(image.shape[1],image.shape[2],image.shape[3]).mul_(self.var).add_(self.mean)
        lower = torch.zeros(decomp_image.shape).to(self.device)
        zero_upper = torch.ones(decomp_image.shape).to(self.device)
        decomp_upper = torch.stack([image.clone()]*component_num, dim=0).reshape(decomp_image.shape).mul_(self.var).add_(self.mean)
        print(f'lower: {lower.shape}')
        print(f'zero_upper: {zero_upper.shape}')
        print(f'decomp_upper: {decomp_upper.shape}')

        for i in range(self.epoch):
            print(f'now {i} epoch')
            
            optimizer.zero_grad()
            
            ## batch processing
            x = (decomp_image.clone()-self.mean) / self.var
            output = self.model(x)
            each_component_loss = torch.tensor([]).to(self.device)
            each_component_norm = torch.tensor([]).to(self.device)
            for k in range(output.shape[0]):
                component_loss = torch.norm(output[k] - latent_tensor[k], p=2)
                each_component_loss = torch.cat((each_component_loss, component_loss.unsqueeze(0)), dim=0)
                if k < output.shape[0]-1:
                    component_norm = torch.norm(decomp_image[k], p=2)
                    each_component_norm = torch.cat((each_component_norm, component_norm.unsqueeze(0)), dim=0)
            decomp_loss = torch.sum(each_component_loss)
            norm_loss = torch.sum(each_component_norm)
            sum_image = torch.sum(decomp_image, dim=0)
            reverse_image = (image.clone()*self.var + self.mean).reshape(sum_image.shape)
            sum_loss = torch.norm(sum_image - reverse_image, p=2)
            
            integration_loss = decomp_loss + 4*sum_loss + 1e-2*norm_loss
            integration_loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            
            if self.clamp==True:
                decomp_image.data = torch.clamp(decomp_image, min=lower, max=decomp_upper)
                # for j in range(decomp_image.shape[0]-1):
                #     decomp_image[j].data = torch.clamp(decomp_image[j], min=lower, max=decomp_upper)
                #     print('clamped')
                # decomp_image[-1].data = torch.clamp(decomp_image[-1], min=lower, max=zero_upper)
            
            decomploss_list.append(each_component_loss.to('cpu').detach())
            sumloss_list.append(sum_loss.to('cpu').detach())
            normloss_list.append(norm_loss.to('cpu').detach())
            del output, decomp_loss, sum_loss, norm_loss, integration_loss, each_component_loss, each_component_norm
        
        # save loss curve
        decomploss_tensor = torch.stack(decomploss_list)
        loss_list = [sumloss_list, normloss_list]
        loss_labels = ["Summation Loss", "Norm Loss"]
        colors = plt.cm.viridis(np.linspace(0, 1, decomploss_tensor.shape[1]+2))
        for loss, label in zip(loss_list, loss_labels):
            plt.plot(np.arange(len(loss)), loss, label=label, color=colors[0])
            colors = np.delete(colors, 0, 0)
        for i in range(decomploss_tensor.shape[1]):
            plt.plot(np.arange(len(decomploss_tensor[:,i])), decomploss_tensor[:,i], label=f'Component{i}', color=colors[0])
            colors = np.delete(colors, 0, 0)
        plt.yscale("log")
        plt.xlabel("Epoch") 
        plt.ylabel("Loss Value") 
        plt.title("Loss Curve")
        plt.legend() 
        plt.grid(True, linestyle="--", alpha=0.6) 
        plt.savefig(f'{base_dir}/loss.png')
        plt.show()
        # return decomp_image, decomploss_list, sumloss_list
        if self.clamp==True:
            self.visualize(decomp_image, image_dir)
        else:
            self.non_clop_visualize(decomp_image, image_dir)
