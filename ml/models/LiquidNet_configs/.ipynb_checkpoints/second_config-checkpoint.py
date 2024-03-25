import torch
import torch.nn as nn
#from ..LiquidNet import LiquidNetBlock, conv_block, bottle_neck_connection
from LiquidNet import LiquidNetBlock, conv_block, bottle_neck_connection

def get_config(channel_coef=8, act_fn=torch.nn.PReLU()):
    A = channel_coef
    
    block_11_settings = {
    "in_blocks" : {
        "IN" : nn.Identity(),
        },    
    "backbone" : conv_block(1, A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_12_settings = {
        "in_blocks" : {
            "b21" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b11" : nn.Identity(),
            },
        "backbone" : conv_block(3*A, A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_13_settings = {
        "in_blocks" : {
            "b22" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b12" : nn.Identity(),
            },
        "backbone" : conv_block(3*A, A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_14_settings = {
        "in_blocks" : {
            "b23" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b13" : nn.Identity(),
            },
        "backbone" : conv_block(3*A, A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }
    
    block_15_settings = {
        "in_blocks" : {
            "b24" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b14" : nn.Identity(),
            },
        "backbone" : conv_block(3*A, A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_21_settings = {
        "in_blocks" : {
            "b11" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(A, 2*A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_22_settings = {
        "in_blocks" : {
            "b31" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b21" : nn.Identity(),
            "b12" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(7*A, 2*A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_23_settings = {
        "in_blocks" : {
            "b32" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b22" : nn.Identity(),
            "b13" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(7*A, 2*A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }
    
    block_24_settings = {
        "in_blocks" : {
            "b33" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            "b23" : nn.Identity(),
            "b14" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(7*A, 2*A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_31_settings = {
        "in_blocks" : {
            "b21" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(2*A, 4*A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_32_settings = {
        "in_blocks" : {
            "b31" : nn.Identity(),
            "b22" : nn.MaxPool3d(2, 2),
            "b41" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            },
        "backbone" : conv_block(14*A, 4*A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }
    
    block_33_settings = {
        "in_blocks" : {
            "b32" : nn.Identity(),
            "b23" : nn.MaxPool3d(2, 2),
            "b42" : nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            },
        "backbone" : conv_block(14*A, 4*A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }
    
    block_41_settings = {
        "in_blocks" : {
            "b31" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(4*A, 8*A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }
    
    block_42_settings = {
        "in_blocks" : {
            "b41" : bottle_neck_connection(8*A, 8*A, 16*A, act_fn=act_fn),
            "b32" : nn.MaxPool3d(2, 2),
            },
        "backbone" : conv_block(12*A, 8*A, kernel_size=3, stride=1, padding=1, act_fn=act_fn), 
    }

    block_out1_settings = {
         "in_blocks" : {
            "b14" : nn.Identity(),
            },
        "backbone" : conv_block(A, 1, kernel_size=3, stride=1, padding=1, act_fn=nn.Sigmoid()),
    }

    block_out2_settings = {
         "in_blocks" : {
            "b24" : nn.Identity(),
            },
        "backbone" : conv_block(2*A, 1, kernel_size=3, stride=1, padding=1, act_fn=nn.Sigmoid()),
    }


    net_blocks = { 
        "b11" : LiquidNetBlock(block_11_settings),
        "b12" : LiquidNetBlock(block_12_settings),
        "b13" : LiquidNetBlock(block_13_settings),
        "b14" : LiquidNetBlock(block_14_settings),
        "b15" : LiquidNetBlock(block_15_settings),
        "b21" : LiquidNetBlock(block_21_settings),
        "b22" : LiquidNetBlock(block_22_settings),
        "b23" : LiquidNetBlock(block_23_settings),
        "b24" : LiquidNetBlock(block_24_settings),
        "b31" : LiquidNetBlock(block_31_settings),
        "b32" : LiquidNetBlock(block_32_settings),
        "b33" : LiquidNetBlock(block_33_settings),
        "b41" : LiquidNetBlock(block_41_settings),
        "b42" : LiquidNetBlock(block_42_settings),
        "out1" : LiquidNetBlock(block_out1_settings),
        "out2" : LiquidNetBlock(block_out2_settings),
    }
    return(net_blocks, ["out1", "out2"])

if __name__ == "__main__":
    get_config()