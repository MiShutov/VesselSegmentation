from torch import nn


class TransformerParams(dict):
    def __init__(self, cfg={}, *args, **kwargs):
        super(TransformerParams, self).__init__(*args, **kwargs)
        self.__dict__.update({
            'img_size' : cfg.get('img_size', None),
            
            'in_channels' : cfg.get('in_channels', None),
            'num_classes' : cfg.get('num_classes', None),
            
            'embedding_dims' : cfg.get('embedding_dims', None),
            'num_heads' : cfg.get('num_heads', None),

            'patch_sizes' : cfg.get('patch_sizes', None),
            'patch_strides' : cfg.get('patch_strides', None),
            'sr_ratios' : cfg.get('sr_ratios', None),
            'mlp_ratios' : cfg.get('mlp_ratios', None),
            'sr_ratios' : cfg.get('sr_ratios', [1,]),
            'depths' : cfg.get('depths', [3,]),
            
            'qk_scale' : cfg.get('qk_scale', None),
            'qkv_bias' : cfg.get('qkv_bias', True),
            
            'drop_rate' : cfg.get('drop_rate', 0),
            'attn_drop_rate' : cfg.get('attn_drop_rate', 0),
            'drop_path_rate' : cfg.get('drop_path_rate', 0.0),

            'norm_layer' : cfg.get('norm_layer', nn.LayerNorm),
        })


class TransformerModule(nn.Module, dict):
    def __init__(self, cfg={}, *args, **kwargs):
        super(TransformerModule, self).__init__(*args, **kwargs)
        self.__dict__.update({
            'img_size' : cfg.get('img_size', None),
            
            'in_channels' : cfg.get('in_channels', None),
            'num_classes' : cfg.get('num_classes', None),
            
            'embedding_dims' : cfg.get('embedding_dims', None),
            'num_heads' : cfg.get('num_heads', None),

            'patch_sizes' : cfg.get('patch_sizes', None),
            'patch_strides' : cfg.get('patch_strides', None),
            'sr_ratios' : cfg.get('sr_ratios', None),
            'mlp_ratios' : cfg.get('mlp_ratios', None),
            'sr_ratios' : cfg.get('sr_ratios', [1,]),
            'depths' : cfg.get('depths', [3,]),
            
            'qk_scale' : cfg.get('qk_scale', None),
            'qkv_bias' : cfg.get('qkv_bias', True),
            
            'drop_rate' : cfg.get('drop_rate', 0),
            'attn_drop_rate' : cfg.get('attn_drop_rate', 0),
            'drop_path_rate' : cfg.get('drop_path_rate', 0.0),

            'norm_layer' : cfg.get('norm_layer', nn.LayerNorm),
        })


