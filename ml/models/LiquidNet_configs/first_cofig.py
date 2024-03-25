import torch
import torch.nn as nn
from LiquidNet import LiquidNetBlock, conv_block, bottle_neck_connection

def get_config(channel_coef = 8, act_fn = torch.nn.PReLU()):
    block_11_settings = {
    "in_blocks" : {
        "IN" : nn.Identity(),
        },    
    "backbone" : conv_block(1, channel_coef, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_12_settings = {
        "in_blocks" : {
            "b21" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b11" : nn.Identity(),
            },
        "backbone" : conv_block(3*channel_coef, 4*channel_coef, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_13_settings = {
        "in_blocks" : {
            "b22" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b12" : bottle_neck_connection(4*channel_coef, 4*channel_coef, 8*channel_coef, act_fn=act_fn),
            },
        "backbone" : conv_block(12*channel_coef, 4*channel_coef, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_14_settings = {
        "in_blocks" : {
            "b23" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b13" : nn.Identity(),
            },
        "backbone" : conv_block(12*channel_coef, 4*channel_coef, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_21_settings = {
        "in_blocks" : {
            "b11" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(channel_coef, 2*channel_coef, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_22_settings = {
        "in_blocks" : {
            "b31" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b21" : nn.Identity(),
            "b12" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(10*channel_coef, 8*channel_coef, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_23_settings = {
        "in_blocks" : {
            "b32" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b22" : nn.Identity(),
            "b13" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(16*channel_coef, 8*channel_coef, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_31_settings = {
        "in_blocks" : {
            "b21" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(2*channel_coef, 4*channel_coef, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_32_settings = {
        "in_blocks" : {
            "b31" : bottle_neck_connection(4*channel_coef, 4*channel_coef, 8*channel_coef, act_fn=act_fn),
            "b22" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(12*channel_coef, 4*channel_coef, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_out1_settings = {
         "in_blocks" : {
            "b14" : nn.Identity(),
            },
        "backbone" : conv_block(4*channel_coef, 1, kernel_size=3, stride=1, padding=1, act_fn=nn.Sigmoid()),
    }

    block_out2_settings = {
         "in_blocks" : {
            "b23" : nn.Identity(),
            },
        "backbone" : conv_block(8*channel_coef, 1, kernel_size=3, stride=1, padding=1, act_fn=nn.Sigmoid()),
    }


    net_blocks = { 
        "b11" : LiquidNetBlock(block_11_settings),
        "b12" : LiquidNetBlock(block_12_settings),
        "b13" : LiquidNetBlock(block_13_settings),
        "b14" : LiquidNetBlock(block_14_settings),
        "b21" : LiquidNetBlock(block_21_settings),
        "b22" : LiquidNetBlock(block_22_settings),
        "b23" : LiquidNetBlock(block_23_settings),
        "b31" : LiquidNetBlock(block_31_settings),
        "b32" : LiquidNetBlock(block_32_settings),
        "out1" : LiquidNetBlock(block_out1_settings),
        "out2" : LiquidNetBlock(block_out2_settings),
    }
    return(net_blocks, ["out1", "out2"])

if __name__ == "__main__":
    get_config()
    