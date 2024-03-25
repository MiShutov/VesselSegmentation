import torch
import torch.nn as nn
from ml.models.building_blocks import (norm_act, conv_block, stem, residual_block,
                            upsample_block, attention_gate, attention_concat, NoiseInjection3d)
from utils import check_None

class ResUNet(nn.Module):
    def __init__(self, in_channels=1, channels_coef=16,
                 out_act=nn.Sigmoid(),
                 with_attention=True,
                 use_encoder_noise=False,
                 use_decoder_noise=False,
                 use_input_noise=False,
                 drop=0.0,
                 dropout_change_per_layer=0.0,
                 minimax_out=False):
        super(ResUNet, self).__init__()
        lc = [channels_coef, 2*channels_coef, 4*channels_coef, 8*channels_coef, 16*channels_coef]
        
        self.minimax_out = minimax_out
        self.check_None = check_None

        if use_input_noise:
            self.input_noise = NoiseInjection3d()
        else:
            self.input_noise = nn.Identity()    
            
        
        self.stem = stem(in_channels=in_channels, out_channels=lc[0])
        
        self.encoder = nn.ModuleList(
           [residual_block(lc[0], lc[1], stride=2, drop=drop),
            residual_block(lc[1], lc[2], stride=2, drop=drop + 1*dropout_change_per_layer),
            residual_block(lc[2], lc[3], stride=2, drop=drop + 2*dropout_change_per_layer),
            residual_block(lc[3], lc[4], stride=2, drop=drop + 3*dropout_change_per_layer)]
        )
        self.bridge = nn.Sequential(
            conv_block(lc[4], lc[4]),
            conv_block(lc[4], lc[4])
        )    
        self.decoder = nn.ModuleList([
           nn.ModuleList([upsample_block(lc[4], lc[3], input_noise=use_decoder_noise),
                          attention_concat(lc[3], lc[3], with_attention),
                          residual_block(2*lc[3], lc[3], stride=1)]),
            
           nn.ModuleList([upsample_block(lc[3], lc[2], input_noise=use_decoder_noise),
                          attention_concat(lc[2], lc[2], with_attention),
                          residual_block(2*lc[2], lc[2], stride=1)]),
            
           nn.ModuleList([upsample_block(lc[2], lc[1], input_noise=use_decoder_noise),
                          attention_concat(lc[1], lc[1], with_attention),
                          residual_block(2*lc[1], lc[1], stride=1)]),
            
           nn.ModuleList([upsample_block(lc[1], lc[0]),
                          attention_concat(lc[0], lc[0], with_attention),
                          residual_block(2*lc[0], lc[0], stride=1)]),
        ])
        
        self.output_block = nn.Sequential(
            nn.Conv3d(in_channels=lc[0], out_channels=1, kernel_size=1, stride=1, padding=0),
            out_act
        )
        
    def forward(self, x):
        x = self.input_noise(x)
        
        skip_layers = []
        x = self.stem(x)
        skip_layers.append(x)
        #encode
        for enc_blok in self.encoder:
            x = enc_blok(x)
            skip_layers.append(x)
        #bridge
        x = self.bridge(x)
        
        #decode
        for idx, dec_blok in enumerate(self.decoder):
            x = dec_blok[0](x)
            x = dec_blok[1](x, skip_layers[3-idx])
            x = dec_blok[2](x)   
        out = self.output_block(x)
        
        if self.minimax_out:
            out = (out - out.min())/(out.max() - out.min() + 1e-8)
        
        return out
    

class ResUNet2d(nn.Module):
    def __init__(self, in_channels=1, channels_coef=16,
                 out_act=nn.Sigmoid(),
                 with_attention=True,
                 use_encoder_noise=False,
                 use_decoder_noise=False,
                 use_input_noise=False,
                 drop=0.0,
                 dropout_change_per_layer=0.0):
        super(ResUNet2d, self).__init__()
        
        self.check_None = check_None
        
        lc = [channels_coef, 2*channels_coef, 4*channels_coef, 8*channels_coef, 16*channels_coef]
        
        if use_input_noise:
            self.input_noise = NoiseInjection3d()
        else:
            self.input_noise = nn.Identity()    
        
        self.stem = stem(in_channels=in_channels, out_channels=lc[0], kernel_size=(3, 3, 1),
                 stride=1, padding=(1, 1, 0))
        
        self.encoder = nn.ModuleList(
           [residual_block(lc[0], lc[1], kernel_size=(3,3,1), stride=2, padding=(1,1,0), drop=drop),
            residual_block(lc[1], lc[2], kernel_size=(3,3,1), stride=2, padding=(1,1,0),
                           drop=drop + 1*dropout_change_per_layer),
            residual_block(lc[2], lc[3], kernel_size=(3,3,1), stride=2, padding=(1,1,0),
                           drop=drop + 2*dropout_change_per_layer),
            residual_block(lc[3], lc[4], kernel_size=(3,3,1), stride=2, padding=(1,1,0),
                           drop=drop + 3*dropout_change_per_layer)]
        )
        self.bridge = nn.Sequential(
            conv_block(lc[4], lc[4], kernel_size=(3,3,1), padding=(1,1,0)),
            conv_block(lc[4], lc[4], kernel_size=(3,3,1), padding=(1,1,0))
        )    
        self.decoder = nn.ModuleList([
           nn.ModuleList([upsample_block(lc[4], lc[3], kernel_size=(3,3,1), padding=(1,1,0),
                                         output_padding=(1,1,0), input_noise=use_decoder_noise),
                          attention_concat(lc[3], lc[3], with_attention),
                          residual_block(2*lc[3], lc[3], kernel_size=(3,3,1), padding=(1,1,0), stride=1)]),
            
           nn.ModuleList([upsample_block(lc[3], lc[2], kernel_size=(3,3,1), padding=(1,1,0),
                                         output_padding=(1,1,0), input_noise=use_decoder_noise),
                          attention_concat(lc[2], lc[2], with_attention),
                          residual_block(2*lc[2], lc[2], kernel_size=(3,3,1), padding=(1,1,0), stride=1)]),
            
           nn.ModuleList([upsample_block(lc[2], lc[1], kernel_size=(3,3,1), padding=(1,1,0),
                                         output_padding=(1,1,0), input_noise=use_decoder_noise),
                          attention_concat(lc[1], lc[1], with_attention),
                          residual_block(2*lc[1], lc[1], kernel_size=(3,3,1), padding=(1,1,0), stride=1)]),
            
           nn.ModuleList([upsample_block(lc[1], lc[0], kernel_size=(3,3,1), padding=(1,1,0), output_padding=(1,1,0)),
                          attention_concat(lc[0], lc[0], with_attention),
                          residual_block(2*lc[0], lc[0], kernel_size=(3,3,1), padding=(1,1,0), stride=1)]),
        ])
        
        self.output_block = nn.Sequential(
            nn.Conv3d(in_channels=lc[0], out_channels=1, kernel_size=1, stride=1, padding=0),
            out_act
        )
        
    def forward(self, x):
        x = self.input_noise(x)
        
        skip_layers = []
        x = self.stem(x)
        check_None(x)
        skip_layers.append(x)
        
        check_None(x)
        
        ### encode
        for enc_blok in self.encoder:
            x = enc_blok(x)
            skip_layers.append(x)
        check_None(x)

        
        ### bridge
        x = self.bridge(x)
        check_None(x)
        
        ### decode
        for idx, dec_blok in enumerate(self.decoder):
            x = dec_blok[0](x)
            x = dec_blok[1](x, skip_layers[3-idx])
            x = dec_blok[2](x)    
        out = self.output_block(x)
        
        return out