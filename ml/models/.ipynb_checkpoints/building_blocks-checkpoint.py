import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from ml.activates import swish, norm_act, GLU
from ml.utils import check_None

check_None_now = check_None

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1,
                 padding_mode='reflect', act = nn.ReLU()):
        super().__init__()
        kernel_size_coef = (kernel_size-1)//2-1
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=dilation+kernel_size_coef, dilation=dilation,
                                 padding_mode=padding_mode)
        
        self.norm = torch.nn.InstanceNorm3d(out_channels, eps=1e-05,
                                            momentum=0.1, affine=False,
                                            track_running_stats=False)
        self.act = act
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))



class MultuScaleDilationConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, stride=1, padding_mode='reflect'):
        super().__init__()
        
        self.Block1 = ConvBlock(in_channels, out_channels//4, kernel_size=1,
                                stride=stride, dilation=1, padding_mode=padding_mode)
        self.Block2 = ConvBlock(in_channels, out_channels//4, kernel_size=3,
                                stride=stride, dilation=1, padding_mode=padding_mode)
        self.Block3 = ConvBlock(in_channels, out_channels//4, kernel_size=3,
                                stride=stride, dilation=2, padding_mode=padding_mode)
        self.Block4 = ConvBlock(in_channels, out_channels//4, kernel_size=3,
                                stride=stride, dilation=4, padding_mode=padding_mode)


    def forward(self, x):
        x1 = self.Block1(x)
        x2 = self.Block2(x)
        x3 = self.Block3(x)
        x4 = self.Block4(x)
        return torch.cat((x1, x2, x3, x4), 1)


def NormLayer3d(c, mode='batch'):
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm3d(c)


class NoiseInjection3d(nn.Module):    
    def __init__(self, alpha=0.1, requires_grad=True):
        super(NoiseInjection3d, self).__init__()
        self.weight = nn.Parameter(alpha*torch.ones(1), requires_grad=requires_grad)
        
    def forward(self, x):
        batch_size, _, h, w, d = x.shape
        noise = torch.randn(batch_size, 1, h, w, d, requires_grad=True).to(x.device)
        return x + self.weight * noise        


class Noising3d(nn.Module):
    def __init__(self, ampl=0.3):
        super().__init__()
        self.ampl = ampl

    def forward(self, x, noise=None):
        noise = torch.randn(*(x.shape)).to(x.device)
        return x + self.ampl * noise      
    

def UpBlockSmall3d(in_channels, out_channels):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv3d(in_channels, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
        NormLayer3d(out_channels*2), GLU())
    return block    


def UpBlockBig3d(in_channels, out_channels):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv3d(in_channels, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
        NoiseInjection3d(),
        NormLayer3d(out_channels*2), GLU(),
        nn.Conv3d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
        NoiseInjection3d(),
        NormLayer3d(out_channels*2), GLU()
        )
    return block


class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super(conv_block, self).__init__()
        self.act = norm_act(in_channels)
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              padding_mode="reflect", bias=bias)

    def forward(self, x):
        check_None_now(x)
        x = self.act(x)
        check_None_now(x)
        x = self.conv(x)
        check_None_now(x)
        return x

    
class stem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super(stem, self).__init__()
        
        self.main_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  padding_mode="reflect", bias=bias),
            conv_block(in_channels=out_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        )
        
        self.shortcut_conv =  nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(1,1,1), stride=1, padding=0, bias=bias),
            norm_act(out_channels, nn.Identity())
        )
        
        
    def forward(self, x):
        main = self.main_conv(x)
        shortcut = self.shortcut_conv(x)
        return main + shortcut


class residual_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 drop=0,
                 bias=False):
        super(residual_block, self).__init__()
        
        self.main_conv = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            conv_block(in_channels=out_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        )
        self.shortcut_conv =  nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(1,1,1), stride=stride, padding=0,
                                  padding_mode="replicate", bias=bias),
            norm_act(out_channels, nn.Identity())
        )
        self.dropout = nn.Dropout3d(p=drop)

    def forward(self, x):
        main = self.main_conv(x)
        shortcut = self.shortcut_conv(x)
        out = main + shortcut
        return self.dropout(out)


class upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 stride=2, padding=1,
                 output_padding=1, drop=0,  
                 bias=False,
                 input_noise=False):
        super(upsample_block, self).__init__()
        
        if input_noise:
            self.noise = NoiseInjection3d()
        else:
            self.noise = nn.Identity()
        
        
        self.unconv = torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                               padding=padding, output_padding=output_padding, bias=bias)
        
    def forward(self, x):
        x = self.noise(x)
        out = self.unconv(x)
        return out


class attention_gate(nn.Module):
    def __init__(self, in_channels1, in_channels2, intermediate_channels, act=nn.PReLU()):
        super(attention_gate, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=in_channels1, out_channels=intermediate_channels,
                              kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=in_channels2, out_channels=intermediate_channels,
                              kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv3d(in_channels=intermediate_channels, out_channels=1,
                              kernel_size=1, stride=1, padding=0)
        self.act = act
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        x1_conv = self.conv1(x1)
        x2_conv = self.conv2(x2)
        inter = self.act(x1_conv + x2_conv)
        inter = self.sigmoid(self.conv(inter))
        return x1*inter


class attention_concat(nn.Module):
    def __init__(self, main_channels, skip_channels, with_attention=True):
        super(attention_concat, self).__init__()
        self.with_attention = with_attention
        if with_attention:
            self.att_gate = attention_gate(skip_channels, main_channels, main_channels)
    
    def forward(self, main, skip):
        if self.with_attention:
            attention_across = self.att_gate(skip, main)
            return torch.cat([main, attention_across], dim=1)
        else:
            return torch.cat([main, skip], dim=1)


class downsample(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1,
                 bias=False, drop=0.0, input_noise=False,
                 use_spec_norm=False, act=nn.ReLU()):
        super(downsample, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              padding_mode="reflect", bias=bias)
        self.act = act
        
        if input_noise:
            self.noise = Noising3d()
        else:
            self.noise = nn.Identity()
        
        if use_spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
            self.norm = nn.Identity()
        else:
            self.norm = nn.Identity()#nn.InstanceNorm3d(out_channels, affine=True)
        self.dropout = nn.Dropout3d(p=drop)

    def forward(self, x):
        x = self.noise(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)


class VG_discriminator(nn.Module):
    def __init__(self, in_channels=1, channels_coef=64,
                 use_input_noise=False, use_layer_noise=False,
                 use_spec_norm=False, act=nn.ReLU()):
        super(VG_discriminator, self).__init__()
        
        if use_input_noise:
            self.input_noise = Noising3d()
        else:
            self.input_noise = nn.Identity()
        
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=channels_coef,
                              kernel_size=4, stride=2, padding=1,
                              padding_mode="reflect")
        
        if use_spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
            self.norm = nn.Identity()
        else:
            self.norm = nn.InstanceNorm3d(channels_coef, affine=True)
        self.act = act
        
        self.downsample = nn.Sequential(
            downsample(channels_coef, 2*channels_coef, kernel_size=4, input_noise=use_layer_noise,
                       stride=1, padding='same', act=nn.PReLU()),
            downsample(2*channels_coef, 4*channels_coef, kernel_size=4, input_noise=use_layer_noise,
                       stride=2, padding=1, act=nn.PReLU()),
            downsample(4*channels_coef, 8*channels_coef, kernel_size=4, input_noise=use_layer_noise,
                       stride=2, padding=1, act=nn.PReLU()),
            downsample(8*channels_coef, 16*channels_coef, kernel_size=4, input_noise=use_layer_noise,
                       stride=2, padding=1, act=nn.PReLU())
        )
        self.conv2 = nn.Conv3d(in_channels=16*channels_coef, out_channels=1,
                               kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.input_noise(x)
        
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)        
        x = self.downsample(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x


class VG_discriminator2d(nn.Module):
    def __init__(self, in_channels=1, channels_coef=64,
                 use_input_noise=False, use_layer_noise=False,
                 use_spec_norm=False, act=nn.ReLU()):
        super(VG_discriminator2d, self).__init__()
        
        if use_input_noise:
            self.input_noise = NoiseInjection3d()
        else:
            self.input_noise = nn.Identity()
        
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=channels_coef,
                              kernel_size=(4, 4, 1), stride=2, padding=(1,1,0),
                              padding_mode="reflect")
        
        if use_spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
            self.norm = nn.Identity()
        else:
            self.norm = nn.InstanceNorm3d(channels_coef, affine=True)
        self.act = act
        
        self.downsample = nn.Sequential(
            downsample(channels_coef, 2*channels_coef, kernel_size=(4, 4, 1), input_noise=use_layer_noise
                       , stride=1, padding='same', act=nn.PReLU()),
            downsample(2*channels_coef, 4*channels_coef, kernel_size=(4, 4, 1), input_noise=use_layer_noise
                       , stride=2, padding=(1, 1, 0), act=nn.PReLU()),
            downsample(4*channels_coef, 8*channels_coef, kernel_size=(4, 4, 1), input_noise=use_layer_noise
                       , stride=2, padding=(1, 1, 0), act=nn.PReLU())
        )
        self.conv2 = nn.Conv3d(in_channels=8*channels_coef, out_channels=1,
                               kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.input_noise(x)
        
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        
        x = self.downsample(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1,
                 padding_mode='reflect', act = nn.ReLU()):
        super().__init__()
        kernel_size_coef = (kernel_size-1)//2-1
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=dilation+kernel_size_coef, dilation=dilation,
                                 padding_mode=padding_mode)
        
        self.norm = torch.nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.act = act
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MultuScaleDilationConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, stride=1, padding_mode='reflect'):
        super().__init__()
        
        self.Block1 = ConvBlock(in_channels, out_channels//4, kernel_size=1,
                                stride=stride, dilation=1, padding_mode=padding_mode)
        self.Block2 = ConvBlock(in_channels, out_channels//4, kernel_size=3,
                                stride=stride, dilation=1, padding_mode=padding_mode)
        self.Block3 = ConvBlock(in_channels, out_channels//4, kernel_size=3,
                                stride=stride, dilation=2, padding_mode=padding_mode)
        self.Block4 = ConvBlock(in_channels, out_channels//4, kernel_size=3,
                                stride=stride, dilation=4, padding_mode=padding_mode)


    def forward(self, x):
        x1 = self.Block1(x)
        x2 = self.Block2(x)
        x3 = self.Block3(x)
        x4 = self.Block4(x)
        return torch.cat((x1, x2, x3, x4), 1)