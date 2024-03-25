import torch
import torch.nn as nn
from modules import ConvModule

class OutputResizeHead(nn.Module):
    def __init__(self, out_size, in_channels, out_channels, act=nn.Sigmoid(), mode='upsample'):
        super(OutputResizeHead, self).__init__()
        self.out_size = out_size
        if mode=='upsample':
            self.up = nn.Sequential(
                    nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                    nn.Upsample(size=self.out_size, mode='trilinear')
                    )
        if mode=='up_conv':
            self.up = nn.Sequential(
                    nn.Upsample(size=self.out_size, mode='trilinear'),
                    ConvModule(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=3, stride=1, dilation=1, padding=1, dim=3, act=None)
            )
        self.act = act  
    
    def forward(self, x):
        x = self.up(x)
        x = self.act(x)
        return(x)