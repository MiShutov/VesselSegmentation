import torch
import torch.nn as nn
import torch.utils.data

from ml.models.unet3d import conv_block, up_conv

class Unet_MSS(nn.Module):

    """
    UNet - Basic Implementation
    Input _ [batch * channel(# of channels of each image) * depth(# of frames) * height * width].
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, channels_coef=64, in_ch=1, out_ch=1,
                 act_fn=nn.ReLU(inplace=True), minimax_out=False):
        super(Unet_MSS, self).__init__()

        self.minimax_out = minimax_out
        filters = [channels_coef, channels_coef * 2, channels_coef * 4,
                   channels_coef * 8, channels_coef * 16] 

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0], act_fn=act_fn)
        self.Conv2 = conv_block(filters[0], filters[1], act_fn=act_fn)
        self.Conv3 = conv_block(filters[1], filters[2], act_fn=act_fn)
        self.Conv4 = conv_block(filters[2], filters[3], act_fn=act_fn)
        self.Conv5 = conv_block(filters[3], filters[4], act_fn=act_fn)


        self.Up4 = up_conv(filters[4], filters[3])
        self.Up_conv4 = conv_block(filters[4], filters[3], act_fn=act_fn)

        self.Up3 = up_conv(filters[3], filters[2])
        self.Up_conv3 = conv_block(filters[3], filters[2], act_fn=act_fn)

        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1], act_fn=act_fn)

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0], act_fn=act_fn)

        #1x1x1 Convolution for Deep Supervision
        self.Conv_d1 = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_d2 = conv_block(filters[1], 1)
        self.Conv_d3 = conv_block(filters[2], 1)

        
        for submodule in self.modules():
            submodule.register_forward_hook(self.nan_hook)

        self.act = torch.nn.Sigmoid()

    def nan_hook(self, module, inp, output):
        for i, out in enumerate(output):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                print(module)
                raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up4(e5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d3_out  = self.Conv_d3(d3)
        d3_out = self.act(d3_out)
                
        d2 = self.Up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv2(d2) 

        d2_out  = self.Conv_d2(d2)
        d2_out = self.act(d2_out)

        d1 = self.Up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        d1_out = self.Conv_d1(d1)
        if self.minimax_out:
            d1_out = (d1_out - d1_out.min())/(d1_out.max() - d1_out.min() + 1e-8)
        else:
            d1_out = self.act(d1_out)
        
        #return [d1_out, d2_out, d3_out]
        #return [d1_out, d2_out]
        
        return d1_out
    
    