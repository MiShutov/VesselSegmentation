import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname('../ml/.'))
from activates import get_activate


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 dilation=1,
                 padding=0,
                 padding_mode='auto',
                 dim=3,
                 norm='batch_norm',
                 act='relu',
                 layer_norm_shape=None,
                 order=('conv', 'norm', 'act'),
                 bias=True):
        super().__init__()
        self.order = order

        assert dim in [1, 2, 3]
        assert norm in ['batch_norm', 'instance_norm', 'layer_norm', None]
        assert padding_mode in ['auto', 'manual']

        self.act = get_activate(act)()
        
        if norm=='batch_norm':
            if dim==1: self.norm = nn.BatchNorm1d(num_features=out_channels)
            if dim==2: self.norm = nn.BatchNorm2d(num_features=out_channels)
            if dim==3: self.norm = nn.BatchNorm3d(num_features=out_channels)
        elif norm=='instance_norm':
            if dim==1: self.norm = nn.InstanceNorm1d(num_features=out_channels)
            if dim==2: self.norm = nn.InstanceNorm2d(num_features=out_channels)
            if dim==3: self.norm = nn.InstanceNorm3d(num_features=out_channels)
        elif norm=='layer_norm':
            if layer_norm_shape is None:
                raise RuntimeError(f"DownSampleModule: 'layer_norm' requires layer_norm_shape arg!")
            self.norm = nn.LayerNorm(layer_norm_shape)
        else:
            self.norm = nn.Identity()    

        if dim==1: conv_fn = nn.Conv1d
        if dim==2: conv_fn = nn.Conv2d
        if dim==3: conv_fn = nn.Conv3d
        
        if padding_mode=='auto':
            pad = kernel_size//2 * (1 + dilation-1)
        else:
            pad = padding
        self.conv = conv_fn(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, dilation=dilation,
                            padding=pad, padding_mode="reflect", bias=bias)
        
        self.actions = {
            'act' : self.act,
            'norm' : self.norm,
            'conv' : self.conv
        }
        
    def forward(self, x):
        for action in self.order:
            x = self.actions[action](x)
        return(x)  


class UpSampleModule(nn.Module):
    def __init__(self,
                 dim=3,
                 mode='upsample',
                 up_coef=2,
                 conv_kernel_size=3,
                 in_channels=None,
                 out_channels=None,
                 norm=None,
                 layer_norm_shape=None,
                 act=None,
                 ):
                 
        super().__init__()
        assert dim in [1, 2, 3]
        assert mode in ['upsample', 'conv']
        assert norm in ['batch_norm', 'instance_norm', 'layer_norm', None]

        if dim==1: interpolation = 'linear'
        if dim==2: interpolation = 'bilinear'
        if dim==3: interpolation = 'trilinear'
        
        if mode=='upsample':
            self.up = nn.Upsample(scale_factor=up_coef, mode=interpolation)
        elif mode=='conv':
            assert in_channels is not None
            if out_channels is None:
                out_channels = in_channels
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=up_coef, mode=interpolation),
                ConvModule(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=conv_kernel_size, stride=1, dilation=1, dim=dim,
                           norm=norm, act=act, layer_norm_shape=layer_norm_shape)        
        )
            
    def forward(self, x):
        x = self.up(x)
        return x


class DownSampleModule(nn.Module):
    def __init__(self,
                 dim=3,
                 mode='avg_pool',
                 down_coef=2,
                 conv_kernel_size=3,
                 in_channels=None,
                 out_channels=None,
                 stride=1,
                 dilation=1,
                 act=None,
                 norm=None,
                 layer_norm_shape=None
                 ):
                 
        super().__init__()
        assert dim in [1, 2, 3]
        assert mode in ['max_pool', 'avg_pool', 'conv']
        assert norm in ['batch_norm', 'instance_norm', 'layer_norm', None]

        if mode=='max_pool':
            if dim==1: self.down = nn.MaxPool1d(kernel_size=down_coef)
            if dim==2: self.down = nn.MaxPool2d(kernel_size=down_coef)
            if dim==3: self.down = nn.MaxPool3d(kernel_size=down_coef)
        elif mode=='avg_pool':
            if dim==1: self.down = nn.AvgPool1d(kernel_size=down_coef)
            if dim==2: self.down = nn.AvgPool2d(kernel_size=down_coef)
            if dim==3: self.down = nn.AvgPool3d(kernel_size=down_coef)
        elif mode=='conv':
            assert in_channels is not None
            if out_channels is None:
                out_channels = in_channels
            self.down = ConvModule(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=conv_kernel_size, stride=down_coef, dilation=dilation,
                                   norm=norm, act=act, layer_norm_shape=layer_norm_shape, dim=dim)
        
    def forward(self, x):
        x = self.down(x)
        return x


     