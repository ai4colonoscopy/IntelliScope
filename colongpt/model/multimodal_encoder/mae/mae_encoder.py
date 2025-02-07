import torch
import torch.nn as nn
from transformers import ViTForImageClassification, AutoImageProcessor


class MAEVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, img_size=384):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.from_local_modified_mae ='cache/downloaded-weights/facebook_vit_mae_large_224to384'
        self.select_layer = -2
        self.img_size = img_size

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = ViTForImageClassification.from_pretrained(self.vision_tower_name).config

    def load_model(self):
        '''some modifications
        1. upsample input size from 224*224px to 384*384px
        2. interpolate positional embedding from 14*14 to 24*24
        '''
        # print(self.from_local_modified_mae)
        self.image_processor = AutoImageProcessor.from_pretrained(self.from_local_modified_mae)

        # NOTE
        # a bug encounter here when using `self.vision_tower = ViTMAEModel.from_pretrained`
        # we refer to https://discuss.huggingface.co/t/how-to-use-vit-mae-for-image-classification/35421/2
        # so here is a pretty ugly implementation, which just use `ViTForImageClassification` class to load MAE weights
        self.vision_tower = ViTForImageClassification.from_pretrained(self.from_local_modified_mae)
        # self.vision_tower.config.mask_ratio = 0
        
        self.vision_tower.requires_grad_(False) # freeze model
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]  # remove [CLS] token
        # print('[DEBUG-image_features]',image_features.shape)

        return image_features   # 1, 576, 1024

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
