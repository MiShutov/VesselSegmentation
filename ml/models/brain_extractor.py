import torch
import torch.nn as nn


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, act_fn=nn.ReLU(inplace=True)):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            act_fn,
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            act_fn
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    # def __init__(self, in_ch, out_ch):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, act_fn=nn.ReLU(inplace=True), scale_factor=(2, 2, 2)):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear'),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            act_fn)

    def forward(self, x):
        x = self.up(x)
        return x


class SpecialUNet(nn.Module):
    """
    UNet - Basic Implementation
    Input _ [batch * channel(# of channels of each image) * depth(# of frames) * height * width].
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=1, channels=16, act=nn.Sigmoid()):
        super(SpecialUNet, self).__init__()
        
        n1 = channels #
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 64,128,256,512,1024

        self.Maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,2,2))
        self.Maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,2,2))
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0], kernel_size=3)
        self.Conv2 = conv_block(filters[0], filters[1], kernel_size=3)
        self.Conv3 = conv_block(filters[1], filters[2], kernel_size=3)
        self.Conv4 = conv_block(filters[2], filters[3], kernel_size=3)
        self.Conv5 = conv_block(filters[3], filters[4], kernel_size=3)

        self.Up5 = up_conv(filters[4], filters[3], scale_factor=(2, 2, 2))
        self.Up_conv5 = conv_block(filters[4], filters[3], kernel_size=3)

        self.Up4 = up_conv(filters[3], filters[2], scale_factor=(2, 2, 2))
        self.Up_conv4 = conv_block(filters[3], filters[2], kernel_size=3)

        self.Up3 = up_conv(filters[2], filters[1], scale_factor=(2, 2, 2))
        self.Up_conv3 = conv_block(filters[2], filters[1], kernel_size=3)

        self.Up2 = up_conv(filters[1], filters[0], scale_factor=(2, 2, 2))
        self.Up_conv2 = conv_block(filters[1], filters[0], kernel_size=3)

        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.act = act

    def forward(self, x):
        e1 = self.Conv1(x)
        #print("e1:", e1.shape)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        #print("e2:", e2.shape)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        #print("e3:", e3.shape)
        
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        #print("e4:", e4.shape)
        
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        #print("e5:", e5.shape)
        
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        #print("d5:", d5.shape)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        #print("d4:", d4.shape)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        #print("d3:", d3.shape)
        
        d2 = self.Up2(d3)
        #print("d2:", d2.shape, "e1:", e1.shape)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        #print("d2:", d2.shape)
        
        out = self.Conv(d2)
        out = self.act(out)
        return out
    
