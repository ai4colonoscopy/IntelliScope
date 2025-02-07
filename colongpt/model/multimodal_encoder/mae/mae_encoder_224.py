import torch
import torch.nn as nn
from transformers import ViTForImageClassification, AutoImageProcessor, ViTMAEModel


class MAEVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, img_size=224):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.img_size = img_size

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = ViTForImageClassification.from_pretrained(self.vision_tower_name).config

    def load_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = ViTForImageClassification.from_pretrained(self.vision_tower_name)
        
        self.vision_tower.requires_grad_(False) # freeze model
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]  # remove [CLS] token

        return image_features

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
