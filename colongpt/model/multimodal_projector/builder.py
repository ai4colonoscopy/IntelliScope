import re
import math
import torch
from torch import nn
from functools import partial
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SELayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
import einops


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class PPC(nn.Module):
    def __init__(self, config, pyramid_shapes):
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.token_trans_layer = nn.Sequential(nn.Linear(inc, ouc), nn.GELU())

        self.pyramid_pool_layers = []
        for shape in pyramid_shapes:
            self.pyramid_pool_layers.append(nn.AdaptiveAvgPool2d(shape))

        self.conv = nn.Conv2d(ouc, ouc, 3, 1, 1, bias=True)
        self.linear_layer = nn.Linear(ouc, ouc)

    def forward(self, tokens):
        tokens = self.token_trans_layer(tokens)  # [16, 729, 1152]

        bs, num_tokens, c = tokens.shape
        patch_size = int(math.sqrt(num_tokens))
        spatial_tokens = tokens.permute(0, 2, 1).reshape(bs, -1, patch_size, patch_size)  # [16, 2048, 27, 27]
        spatial_tokens_list = []
        for i, pool_layer in enumerate(self.pyramid_pool_layers):  
            pooled = pool_layer(spatial_tokens)  # [16, 2048, 14, 14] [16, 2048, 7, 7] [16, 2048, 1, 1]
            if i < len(self.pyramid_pool_layers) - 1:
                pooled = self.conv(pooled)
            pooled = pooled.flatten(2).transpose(1, 2)  # [16, 196, 2048] [16, 49, 2048] [16, 1, 2048]
            spatial_tokens_list.append(pooled)
        concat_tokens = torch.cat(spatial_tokens_list, dim=1)  # [16, 246, 2048]

        return self.linear_layer(concat_tokens)  # [16, 246, 2048]


class MultigranularityAdapter(nn.Module):
    def __init__(self, config=None, projector_type='ppc_14_7_1'):
        super().__init__()

        shapes = projector_type.split('_')[1:]
        pyramid_shapes = [(int(shape), int(shape)) for shape in shapes]
        self.model = PPC(config, pyramid_shapes=pyramid_shapes)
    def forward(self, x):
        return self.model(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    elif projector_type.startswith('ppc'):
        print('Using pyramid pooling connector...')
        return MultigranularityAdapter(config, projector_type)


    raise ValueError(f'Unknown multimodal adapter type: {projector_type}')
