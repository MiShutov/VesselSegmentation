import torch
import torch.nn as nn
from BaseDecodeHead import BaseDecodeHead
from transformer_blocks import ConvModule

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048,
                 embedding_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, embedding_dim, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)

        self.net_depth = len(self.in_channels)
        self.MLPs = nn.ModuleList(
            [MLP(input_dim=self.in_channels[i], embedding_dim=embedding_dim) for i in range(self.net_depth)]
        )

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*self.net_depth,
            out_channels=embedding_dim,
            kernel_size=1, dim=3
        )

        self.linear_pred = nn.Conv3d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        assert len(inputs) == self.net_depth
        x = self._transform_inputs(inputs)

        N = x[0].shape[0]
        first_lvl_shape = x[0].shape[2:]
        
        projections = []
        for i in range(self.net_depth):
            patch_grid = x[i].size()[2:]
            tmp = self.MLPs[i](x[i]).permute(0,2,1).reshape(N, -1, *patch_grid)
            tmp = nn.Upsample(size=first_lvl_shape, mode='trilinear', align_corners=False)(tmp)
            projections.append(tmp)
            
        x = torch.cat(projections, dim=1)
        x = self.linear_fuse(torch.cat(projections, dim=1))
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x