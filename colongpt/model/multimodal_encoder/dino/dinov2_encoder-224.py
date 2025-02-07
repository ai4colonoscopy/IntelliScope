import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

#DINOV2
class Dinov2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, img_size=224):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.img_size = img_size

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoModel.from_pretrained(self.vision_tower_name).config

    def load_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.crop_size = {'height': self.img_size, 'width': self.img_size}
        self.image_processor.size = {'shortest_edge': self.img_size}

        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False) # freeze model
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]  # remove [CLS] token
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
        if isinstance(images, list):
            image_features = []
            for image in images:
                image = image.float()  # convert to float32
                # print('[DEBUG-Img-shape]',image.shape)
                image = (image - image.min()) / (image.max() - image.min())  # normalize to [0, 1]
                processed_image = self.image_processor(images=image, return_tensors="pt").pixel_values
                image_forward_out = self.vision_tower(processed_image.to(device=self.device, dtype=torch.float32),  # use float32 for DINOv2
                                                      output_hidden_states=True, return_dict=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype).bfloat16()  # back to BFloat16
                image_features.append(image_feature)
                # print('[DEBUG-img-fearure]',image_feature.shape) # debug
        else:
            images = images.float()  # convert to float32
            # print('[DEBUG-Img-shape]',images.shape)
            images = (images - images.min()) / (images.max() - images.min())  # normalize to [0, 1]
            processed_image = self.image_processor(images=images, return_tensors="pt").pixel_values
            image_forward_out = self.vision_tower(processed_image.to(device=self.device, dtype=torch.float32),  # use float32 for DINOv2
                                                   output_hidden_states=True, return_dict=True)
            image_features = self.feature_select(image_forward_out).to(images.dtype).bfloat16()  # back to BFloat16
            # print('[DEBUG-img-fearure]',image_features.shape) # debug [1, 256, 1024]
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



# [DEBUG-Img-shape] torch.Size([1, 3, 224, 224])
# [DEBUG-img-fearure] torch.Size([1, 256, 1024])