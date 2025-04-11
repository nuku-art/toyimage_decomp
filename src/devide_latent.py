import torch
import math

class process_latent():
    def __init__(self, device):
        self.device = device
    
    def devide(self, latent, attribution, num=2, operators=None, high_priority=True):
        latent = latent.to('cpu').detach()
        if not isinstance(attribution, torch.Tensor):
            attribution = torch.tensor(attribution)
        
        if operators != None:
            if not isinstance(operators, torch.Tensor):
                operators = torch.tensor(operators).to(self.device)
            squeeze_latent = latent.clone().squeeze()
            if squeeze_latent.dim() != 2:
                squeeze_latent = squeeze_latent.reshape(1, -1)
            transform_latent = operators @ squeeze_latent
            latent = transform_latent.reshape(latent.shape[0], operators.shape[0], latent.shape[2], latent.shape[3])
            attribution = operators @ attribution

        latent = latent.squeeze()

        descend_indices = torch.argsort(attribution, descending=True)
        ascend_indices = torch.argsort(attribution, descending=False)
        
        large_num = math.ceil(num/2)
        small_num = int(num -  large_num)
        if high_priority==True:
            large_latent_tensor = torch.zeros((large_num, *latent.shape))
            for i in range(large_num):
                large_latent_tensor[i,descend_indices[i],:,:]  = latent[descend_indices[i],:,:]
            if small_num > 0:
                small_latent_tensor = torch.zeros((small_num, *latent.shape))
                for j in range(small_num):
                    small_latent_tensor[j,ascend_indices[j],:,:]  = latent[ascend_indices[j],:,:]
            used_indices = torch.cat([descend_indices[:large_num], ascend_indices[:small_num]])
            else_indices = ~torch.isin(torch.arange(latent.shape[0]), used_indices)
            else_latent_tensor = torch.zeros((1,*latent.shape))
            else_latent_tensor[0,else_indices,:,:] = latent[else_indices,:,:]
        else:
            small_latent_tensor = torch.zeros((large_num, *latent.shape))
            for j in range(large_num):
                small_latent_tensor[j,ascend_indices[j],:,:]  = latent[ascend_indices[j],:,:]
            if small_num > 0:
                large_latent_tensor = torch.zeros((small_num, *latent.shape))
                for i in range(small_num):
                   large_latent_tensor[i,descend_indices[i],:,:]  = latent[descend_indices[i],:,:]
            used_indices = torch.cat([descend_indices[:small_num], ascend_indices[:large_num]])
            else_indices = ~torch.isin(torch.arange(latent.shape[0]), used_indices)
            else_latent_tensor = torch.zeros((1,*latent.shape))
            else_latent_tensor[0,else_indices,:,:] = latent[else_indices,:,:]
        zero_latent_tensor = torch.zeros((1,*latent.shape))
        
        if high_priority==True and small_num<=0:
            latent_tensor = torch.cat([large_latent_tensor, else_latent_tensor, zero_latent_tensor], dim=0)
        elif high_priority==False and small_num<=0:
            latent_tensor = torch.cat([small_latent_tensor, else_latent_tensor, zero_latent_tensor], dim=0)
        else:
            latent_tensor = torch.cat([large_latent_tensor, small_latent_tensor, else_latent_tensor, zero_latent_tensor], dim=0)
        
        return latent_tensor
        
        
        