import torch
import torch.nn as nn

class VanGan(nn.Module):
    def __init__(self, modules):
        super(VanGan, self).__init__()
        self.gen_IS = modules['gen_IS']
        self.gen_SI = modules['gen_SI']
        self.disc_I = modules['disc_I']
        self.disc_S = modules['disc_S']