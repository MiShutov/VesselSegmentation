import torch
import torch.nn as nn
from itertools import combinations_with_replacement

from transformers_models.modules import UpSampleModule, DownSampleModule, ConvModule


class GaussianBlur3D(nn.Module):
    def __init__(self, in_channels, sigma, device, kernel_size=7):
        super(GaussianBlur3D, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv3d(in_channels, in_channels, self.kernel_size, stride=1,
                              padding=self.kernel_size//2, dilation=1, bias=False, 
                              padding_mode='replicate').to(self.device)
        self.set_weights(sigma)
    
    def set_weights(self, sigma):
        if not (isinstance(sigma, torch.FloatTensor) or
                isinstance(sigma, torch.cuda.FloatTensor)):
            sigma = torch.tensor(sigma, dtype=torch.float32)
        if sigma.shape in (torch.Size([]), torch.Size([1])):
            sigma = sigma.expand(3)
        assert sigma.shape == torch.Size([3])
        sigma = sigma.to(self.device)
        kernel1d = torch.arange(self.kernel_size).type(torch.float32) - self.kernel_size//2
        kernel1d = kernel1d.to(self.device)
        kernel1 = (kernel1d)**2 / 2 / sigma[0]**2
        kernel2 = (kernel1d)**2 / 2 / sigma[1]**2
        kernel3 = (kernel1d)**2 / 2 / sigma[2]**2
        kernel1 = torch.exp(-kernel1)
        kernel2 = torch.exp(-kernel2)
        kernel3 = torch.exp(-kernel3)
        kernel3d = torch.einsum('i,j,k->ijk', kernel1, kernel2, kernel3)
        
        kernel3d /= kernel3d.sum()   
        kernel3d = kernel3d.expand(self.in_channels, self.in_channels, *kernel3d.shape)
        self.conv.weight = torch.nn.Parameter(kernel3d.to(self.device), requires_grad=False)
        
    def forward(self, x, sigma=None):
        if sigma is not None:
            self.set_weights(sigma)    
            
        return(self.conv(x))


class HessianTorch(nn.Module):
    def __init__(self, in_channels, sigma, device, with_blur=True):
        super(HessianTorch, self).__init__()
        self.with_blur = with_blur
        if self.with_blur:
            self.blur = GaussianBlur3D(in_channels=in_channels,
                                        sigma=sigma,
                                        device=device)
        
    
    def forward(self, x, sigma=None):
        assert len(x.shape)==5
        axes = [2, 3, 4]
        if self.with_blur:
            x = self.blur(x, sigma)
        gradient = torch.gradient(x, dim=axes)
        H_elems = [torch.gradient(gradient[ax0-2], axis=ax1)[0]
              for ax0, ax1 in combinations_with_replacement(axes, 2)]

        out = torch.stack(H_elems)
        return out


class HessBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 start_scale,
                 device,
                 fc_channels=10,
                 act=nn.Sigmoid(),
                 with_blur=True,
                 learnable_scale=True):
        super(HessBlock, self).__init__()
        self.device = device
        self.learnable_scale = learnable_scale
        self.scale = start_scale
        if not (isinstance(self.scale, torch.FloatTensor) or
                isinstance(self.scale, torch.cuda.FloatTensor)):
            self.scale = torch.tensor(self.scale, dtype=torch.float32)
        if self.scale.shape == torch.Size([]):
            self.scale = self.scale.unsqueeze(0)
        
        if self.learnable_scale:
            self.scale = nn.parameter.Parameter(data=self.scale).to(device)
        else:
            self.scale = self.scale.to(device)
        
        self.linear = nn.Sequential(
            nn.Linear(6+self.scale.shape[0], fc_channels, bias=True),
            nn.ReLU(),
            #nn.Linear(fc_channels, fc_channels, bias=True),
            #nn.ReLU(),
            nn.Linear(fc_channels, 1, bias=True),
            act
        ).to(device)
        self.hess = HessianTorch(in_channels=in_channels,
                                 sigma=self.scale.detach().cpu(),
                                 device=device,
                                 with_blur=with_blur)
        self.flat = nn.Flatten(start_dim=2, end_dim=4)
        
    def forward(self, x):
        input_sizes = x.shape
        if self.learnable_scale:
            x = self.hess(x, self.scale).permute(1,2,3,4,5,0)
        else:
            x = self.hess(x).permute(1,2,3,4,5,0)
        x = self.flat(x)
        
        scale_attention = self.scale.expand(*x.shape[:3], self.scale.shape[0])
        x = torch.cat([x, scale_attention], axis=3)
        
        x = self.linear(x)
        x = torch.unflatten(x, 2, input_sizes[2:])
        x = x.squeeze(-1)
        return x


class HessFeatures(nn.Module):
    def __init__(self,
                 in_channels,
                 n_hess_blocks,
                 start_scale=1,
                 device='cpu',
                 hess_with_blur=True,
                 hess_learnable_scale=True,
                 out_act=nn.ReLU()):
        super(HessFeatures, self).__init__()    
        self.device = device
        
        self.scale = torch.tensor(start_scale, dtype=torch.float32)
        
        self.HessBlocks = nn.ModuleList(
            [HessBlock(in_channels=in_channels,
                       start_scale=self.scale,
                       device=device,
                       act=out_act,
                       with_blur=hess_with_blur,
                       learnable_scale=hess_learnable_scale) for _ in range(n_hess_blocks)])
        
    def forward(self, x):
        h = []
        for HessBlock in self.HessBlocks:
            h.append(HessBlock(x))     
        out = torch.cat(h, 1)
        return(out)
    
    def to_device(self, device):
        self.to(device)
        for HessBlock in self.HessBlocks:
            HessBlock.device = device
        
        for HessBlock in self.HessBlocks:
            HessBlock.hess.gauss.device = device


class HessNet(nn.Module):
    def __init__(self,
                 device,
                 in_channels=1,
                 out_channels=1,
                 start_scale=1,
                 n_hess_blocks=4):
        super(HessNet, self).__init__()
        self.device = device

        self.hess = HessFeatures(in_channels=1,
                                  n_hess_blocks=2,
                                  start_scale=start_scale,
                                  device=device,
                                  hess_with_blur=True,
                                  hess_learnable_scale=True)
            
        self.conv = ConvModule(in_channels=3, out_channels=1,
                                kernel_size=5, norm=None, act=None, bias=True, padding='auto')

        
        self.act = nn.Sigmoid()

        self.to(device)

    def to(self, device):
        return super().to(torch.device(device))
    
    def forward(self, x):
        out = self.conv(torch.cat([x, self.hess(x)], axis=1))
        return self.act(out)



