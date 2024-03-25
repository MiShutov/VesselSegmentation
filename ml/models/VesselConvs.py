import torch
import torch.nn as nn
from transformers_models.modules import UpSampleModule, DownSampleModule, ConvModule

class JustConv(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=5):
        super(JustConv, self).__init__()
            
        self.conv = ConvModule(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, norm=None,
                               act='sigmoid', bias=True, padding='auto')
    
    def forward(self, x):
        return self.conv(x)


class TwoConv(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 hidden_channels=5,
                 kernel_size=5):
        super(TwoConv, self).__init__()
            
        self.conv1 = ConvModule(in_channels=in_channels, out_channels=hidden_channels,
                               kernel_size=kernel_size, norm='batch_norm',
                               act='relu', bias=True, padding='auto')

        self.conv2 = ConvModule(in_channels=hidden_channels, out_channels=out_channels,
                               kernel_size=kernel_size, norm='batch_norm',
                               act='sigmoid', bias=True, padding='auto')

    
    def forward(self, x):
        return self.conv2(self.conv1(x))