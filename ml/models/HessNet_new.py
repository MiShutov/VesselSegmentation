import torch
import torch.nn as nn
from itertools import combinations_with_replacement
from ml.transformers_models.modules import UpSampleModule, DownSampleModule, ConvModule


class HessianTorch():
    def __call__(self, x, sigma=None):
        assert len(x.shape)==5
        axes = [2, 3, 4]
        gradient = torch.gradient(x, dim=axes)
        H_elems = [torch.gradient(gradient[ax0-2], axis=ax1)[0]
              for ax0, ax1 in combinations_with_replacement(axes, 2)]

        out = torch.stack(H_elems)
        return out


class HessBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 projection_channels=1,
                 fc_channels=10,
                 act=nn.Sigmoid()):
        
        super(HessBlock, self).__init__()
        self.proj_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels*projection_channels,
                                    kernel_size=5, stride=1, padding=5//2, dilation=1,
                                    bias=False, padding_mode='replicate')

        # self.proj_conv = nn.Sequential(
        #     nn.Conv3d(in_channels=in_channels,
        #               out_channels=(in_channels*projection_channels)//2,
        #               kernel_size=3, stride=1, padding=3//2, dilation=1,
        #               bias=False, padding_mode='replicate'),
        #     nn.Conv3d(in_channels=(in_channels*projection_channels)//2,
        #               out_channels=in_channels*projection_channels,
        #               kernel_size=3, stride=1, padding=3//2, dilation=1,
        #               bias=False, padding_mode='replicate'),
        # )

        self.learnable_frangi = nn.Sequential(
            nn.Linear(6 * in_channels * projection_channels, fc_channels, bias=True),
            nn.ReLU(),
            nn.Linear(fc_channels, in_channels, bias=True),
            act
        )
        
        self.hess = HessianTorch()
    
    
    def forward(self, x):
        x = self.proj_conv(x)
        x = self.hess(x).permute(1,3,4,5,0,2)
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = self.learnable_frangi(x)
        x = x.permute(0,4,1,2,3)
        return x


class HessFeatures(nn.Module):
    def __init__(self,
                 in_channels,
                 n_hess_blocks,
                 projection_channels=8,
                 out_act=nn.ReLU()):
        super(HessFeatures, self).__init__()    
        self.HessBlocks = nn.ModuleList(
            [HessBlock(in_channels,
                      projection_channels=projection_channels,
                      fc_channels=10,
                      act=nn.Sigmoid()) for _ in range(n_hess_blocks)])


    
    def forward(self, x):
        h = []
        for HessBlock in self.HessBlocks:
            h.append(HessBlock(x))     
        out = torch.cat(h, 1)
        return(out)


class HessNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 projection_channels=8,
                 n_hess_blocks=4):
        super(HessNet, self).__init__()
        self.hess = HessFeatures(in_channels=in_channels,
                                 n_hess_blocks=n_hess_blocks,
                                 projection_channels=projection_channels,
                                 out_act=nn.ReLU())
        
        self.out_conv = ConvModule(in_channels=in_channels*(1+n_hess_blocks),
                               out_channels=out_channels,
                               kernel_size=5,
                               norm=None,
                               act=None,
                               bias=True,
                               padding='auto')

        # self.out_conv = nn.Sequential(
        #     nn.Conv3d(in_channels=in_channels*(1+n_hess_blocks),
        #               out_channels=in_channels*(1+n_hess_blocks)//2,
        #               kernel_size=3, stride=1, padding=3//2, dilation=1,
        #               bias=False, padding_mode='replicate'),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=in_channels*(1+n_hess_blocks)//2,
        #               out_channels=in_channels*projection_channels,
        #               kernel_size=3, stride=1, padding=3//2, dilation=1,
        #               bias=False, padding_mode='replicate'),
        # )

        self.act = nn.Sigmoid()

    
    def forward(self, x):
        x = torch.cat([x, self.hess(x)], axis=1)
        out = self.out_conv(x)
        return self.act(out)


class HessUnet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 n_hess_blocks=4):
        super(HessUnet, self).__init__()
        self.hess_1 = GenHessBlock(in_channels,
                                   multiply_channels=n_hess_blocks,
                                   projection_kernel_size=5,
                                   projection_channels=8,
                                   act=nn.ReLU())

        
        self.hess_2 = GenHessBlock(in_channels,
                                   multiply_channels=n_hess_blocks,
                                   projection_kernel_size=3,
                                   projection_channels=4,
                                   act=nn.ReLU())

        self.down = DownSampleModule(dim=3,
                                     mode='avg_pool', #conv
                                     down_coef=2,
                                     conv_kernel_size=3,
                                     in_channels=None,
                                     stride=1,
                                     dilation=1,
                                     act=None,)

        self.up = UpSampleModule(dim=3,
                                 mode='upsample',
                                 up_coef=2,
                                 conv_kernel_size=3,
                                 in_channels=None,
                                 norm=None,
                                 layer_norm_shape=None,
                                 act=None,)
        
        self.out_conv = ConvModule(in_channels=in_channels*(2*in_channels*n_hess_blocks),
                                   out_channels=out_channels,
                                   kernel_size=5,
                                   norm=None,
                                   act=None,
                                   bias=True,
                                   padding='auto')

        self.act = nn.Sigmoid()

    
    def forward(self, x):
        h1 = self.hess_1(x)
        
        h2 = self.down(x)
        h2 = self.hess_2(h2)
        h2 = self.up(h2)
        
        out = self.out_conv(torch.cat([h1, h2], axis=1))
        return self.act(out)

