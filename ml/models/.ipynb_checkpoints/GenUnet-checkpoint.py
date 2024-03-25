import torch
import torch.nn as nn
from ml.transformers_models.modules import UpSampleModule, DownSampleModule, ConvModule


class GenConvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, dim, in_channels, out_channels,
                 kernel_size=3, bias=True, act='relu'):
        super(GenConvBlock, self).__init__()
        
        self.conv1 = ConvModule(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding_mode='auto',
                                dim=dim,
                                norm='batch_norm',
                                act=act,
                                order=('conv', 'norm', 'act'),
                                bias=bias)
        
        self.conv2 = ConvModule(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding_mode='auto',
                                dim=dim,
                                norm='batch_norm',
                                act=act,
                                order=('conv', 'norm', 'act'),
                                bias=bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# class up_conv(nn.Module):
#     """
#     Up Convolution Block
#     """

#     # def __init__(self, in_ch, out_ch):
#     def __init__(self, in_channels, out_channels, k_size=3, stride=1,
#                  padding=1, bias=True, act_fn=nn.ReLU(inplace=True)):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
#                       stride=stride, padding=padding, bias=bias),
#             nn.BatchNorm3d(num_features=out_channels),
#             act_fn)

#     def forward(self, x):
#         x = self.up(x)
#         return x



class GenUnet(nn.Module):
    def __init__(self, dim, in_channels=1, out_channels=1, channels=16,
                 act=nn.Sigmoid(), depth=4):
        super(GenUnet, self).__init__()

        self.dim = dim
        self.depth = depth
        
        filters = [in_channels,]
        for i in range(depth+1):
            filters.append(channels * (2**i))
        
        self.Pools = nn.ModuleList([])
        self.Convs = nn.ModuleList([])
        self.UpConvs = nn.ModuleList([])
        self.Ups = nn.ModuleList([])

        for i in range(1, depth+1):
            #self.Pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.Pools.append(DownSampleModule(dim=dim, mode='max_pool', down_coef=2))
            self.Convs.append(GenConvBlock(dim=dim, in_channels=filters[i], out_channels=filters[i+1]))
            self.UpConvs.append(GenConvBlock(dim=dim, in_channels=filters[i+1], out_channels=filters[i]))
            self.Ups.append(UpSampleModule(in_channels=filters[i+1], out_channels=filters[i], dim=dim,
                                           mode='conv', up_coef=2, conv_kernel_size=3, norm='batch_norm',
                                           layer_norm_shape=None, act='relu'))
            #self.Ups.append(up_conv(filters[i+1], filters[i]))

        self.inConv = GenConvBlock(dim=dim, in_channels=in_channels, out_channels=channels)
        self.outConv = nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = act
        
    def forward(self, x):
        down_features = []    
        down_features.append(self.inConv(x))
        
        for i in range(self.depth):
            down_features.append(self.Convs[i](self.Pools[i](down_features[i])))


        x = down_features[i+1]
        for i in reversed(range(self.depth)):
            x = self.Ups[i](x)
            x = torch.cat((down_features[i], x), dim=1)
            x = self.UpConvs[i](x)

        out = self.outConv(x) 
        out = self.act(out)
        return out
