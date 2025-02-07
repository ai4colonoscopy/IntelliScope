import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# reference:https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L174
class EmbeddingInterpolator:
    def __init__(self, pos_embed, patch_embed):
        self.pos_embed = pos_embed
        self.patch_embed = patch_embed

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

#References:
# DINO: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L174
def interpolate_pos_encoding(x, w, h, patcah_size):
    npatch = x.shape[1] - 1
    N = x.shape[1] - 1
    class_pos_embed = x[:, 0]
    patch_pos_embed = x[:, 1:]
    dim = x.shape[-1]
    w0 = w // patcah_size
    # print('[DEBUG-w0]', w0)
    h0 = h // patcah_size
    # print('[DEBUG-h0]', h0)
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

#References:
# DINO: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L174
def interpolate_pos_encoding_clip(x, w, h, patcah_size):
    npatch = x.shape[0] - 1
    N = x.shape[0] - 1
    class_pos_embed = x[0]
    patch_pos_embed = x[1:]
    dim = x.shape[-1]
    w0 = w // patcah_size
    # print('[DEBUG-w0]', w0)
    h0 = h // patcah_size
    # print('[DEBUG-h0]', h0)
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.unsqueeze(0).reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
    pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=0)
    return pos_embed.squeeze(0)

#References:
# pytorch: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
def interpolate_eva(weight_key, new_length):
    original_length, num_features = weight_key.shape
    weight_key = weight_key.unsqueeze(0).unsqueeze(0)
    scale_factor = new_length / original_length
    interpolated_weight_key = F.interpolate(weight_key, scale_factor=(scale_factor, 1), mode='bicubic', align_corners=False)
    interpolated_weight_key = interpolated_weight_key.squeeze(0).squeeze(0)

    return interpolated_weight_key