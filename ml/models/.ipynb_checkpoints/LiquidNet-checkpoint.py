import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=True, act_fn=nn.ReLU(inplace=True)):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            #nn.BatchNorm3d(num_features=out_channels),
            nn.InstanceNorm3d(out_channels, affine=True),
            act_fn,
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            #nn.BatchNorm3d(num_features=out_channels),
            nn.InstanceNorm3d(out_channels, affine=True),
            act_fn
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class bottle_neck_connection(nn.Module):
    def __init__(self, in_channels, out_channels, bottle_channels,
                 bias=True, act_fn=nn.ReLU(inplace=True)):
        super(bottle_neck_connection, self).__init__()
        self.bottleneck1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=bottle_channels, kernel_size=3,
                      stride=2, padding=1, bias=bias),
            #nn.BatchNorm3d(num_features=bottle_channels),
            nn.InstanceNorm3d(bottle_channels, affine=True),
            act_fn,
        )
        
        self.bottleneck2 = nn.Sequential(
            nn.Conv3d(in_channels=bottle_channels, out_channels=bottle_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            act_fn,
        )
        
        self.bottleneck3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=bottle_channels, out_channels=out_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=bias),
            #nn.BatchNorm3d(num_features=out_channels),
            nn.InstanceNorm3d(out_channels, affine=True),
            act_fn,
        )

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        return x
    

class LiquidNetBlock(nn.Module):
    def __init__(self, settings):
        super(LiquidNetBlock, self).__init__()
        self.backbone = settings["backbone"]
        self.in_blocks = torch.nn.ModuleDict(settings["in_blocks"])
        
    def forward(self, in_list):
        x = torch.cat(in_list, dim=1)
        x = self.backbone(x)
        return(x)
    
    
class LiquidNet(nn.Module):
    def __init__(self, net_blocks, out_blocks, debug=False):
        super(LiquidNet, self).__init__()
        self.net_blocks = torch.nn.ModuleDict(net_blocks)
        self.out_blocks = out_blocks
        
        self.net_graph = self.make_Net_graph()
        self.debug = debug
        
        
    def make_Net_graph(self):
        graph = {}
        first_vertex = "IN"
        verified_verteсes = set( (first_vertex,))

        block_queue = [*self.net_blocks.keys()]
        queued_blocks_count = 0
        run = True

        while run:
            if len(block_queue)==0:
                break

            if queued_blocks_count>len(block_queue):
                print(graph)
                raise RuntimeError('Net::make_Net_graph::Error: Can\'t build graph, please check Net.net_blocks')

            block_name = block_queue.pop(0)
            in_blocks = self.net_blocks[block_name].in_blocks.keys()

            if sorted(list(in_blocks))== sorted(list(verified_verteсes.intersection(in_blocks))):
                verified_verteсes.add(block_name)
                for start in in_blocks:
                    if graph.get(start) is None:
                        graph.update({start : [block_name,]})
                    else: 
                        graph[start].append(block_name)
                queued_blocks_count = 0
            else:
                block_queue.append(block_name)
                queued_blocks_count+=1
        
        assert sorted(list(self.out_blocks))== sorted(list(verified_verteсes.intersection(self.out_blocks)))
        return graph
    

    def forward(self, IN):
        block_outs = {"IN" : IN} #we will iteratively calculate all outputs
        
        for vertes in self.net_graph.keys():
            for block_name in self.net_graph[vertes]:
                if (block_name not in block_outs.keys()):
                    block = self.net_blocks[block_name]
                    if self.debug:
                        print("block_name:", block_name)
                        print("block.in_blocks:", block.in_blocks)

        
                    in_list = [] #upload all inputs to block
                    for in_block in block.in_blocks: 
                        resampled_out = block.in_blocks[in_block](block_outs[in_block])
                        in_list.append(resampled_out)
                    block_outs.update({block_name : (block(in_list))})
        return [block_outs[out_name] for out_name in self.out_blocks]
    
    
#     def forward_old(self, IN):
#         block_outs = {"IN" : IN} #we will iteratively calculate all outputs
        
#         for vertes in self.net_graph.keys():
#             queue = [*self.net_graph[vertes]] 
#             while len(queue) > 0:
#                 block_name = queue.pop(0)
#                 if (block_name not in block_outs.keys()):
#                     block = self.net_blocks[block_name]
                    
                    
#                     put_to_queue = False #If true, we can't calculate output now, set this block to the end of queue
#                     for in_block in block.in_blocks:
#                         if block_outs.get(in_block) is None:
#                             queue.append(block_name)
#                             put_to_queue = True
#                             break
#                     if put_to_queue:
#                         continue
                

#                     in_list = [] #upload all inputs to block
#                     for in_block in block.in_blocks: 
#                         resampled_out = block.in_blocks[in_block](block_outs[in_block])
#                         in_list.append(resampled_out)
#                     block_outs.update({block_name : (block(in_list))})
#         return(block_outs["out"])