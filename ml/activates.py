import torch
import torch.nn as nn


class swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)
    

class GLU(nn.Module):
    """Gated Linear Unit"""
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class norm_act(nn.Module):
    def __init__(self, channels, act=swish()):
        super(norm_act, self).__init__()
        self.act = act
        #self.norm = #nn.InstanceNorm3d(channels, affine=False)
        #self.norm = torch.nn.LazyInstanceNorm3d()
        #self.norm = torch.nn.BatchNorm3d(channels)
        self.norm = nn.Identity(channels)
        
    def forward(self, x):
        normed_x = self.norm(x)
        out = self.act(normed_x)
        return out

activates_dict = {
    'relu' : nn.ReLU,
    'elu' : nn.ELU,
    'leaky_relu' : nn.LeakyReLU,
    'prelu' : nn.PReLU,
    'sigmoid' : nn.Sigmoid,
    'tanh' : nn.Tanh,
    'gelu' : nn.GELU,
    'swish' : swish,
    'glu' : GLU,
}

def get_activate(act):
    if act in list(activates_dict.keys()) + [None,]:
        if activates_dict.get(act):
            return(activates_dict.get(act))
        else:
            return(nn.Identity)
    else:
        raise RuntimeError(f'activates.py: get_activate: can\'t upload <{act}> activation')


class GetActivates(dict):
    relu = nn.ReLU
    elu = nn.ELU
    leaky_relu = nn.LeakyReLU
    prelu = nn.PReLU
    sigmoid = nn.Sigmoid
    tanh = nn.Tanh
    gelu = nn.GELU
    swish = swish
    glu = GLU

