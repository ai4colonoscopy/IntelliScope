# import torch
# import torch.nn as nn
# import math
# from transformers import AutoImageProcessor, AutoModel

# # Helper function for position encoding interpolation
# def interpolate_pos_encoding(pos_embed, w, h, patch_size):
#     try:
#         # Debug: Print input shapes
#         print(f"[DEBUG] pos_embed shape: {pos_embed.shape}, target width: {w}, height: {h}, patch_size: {patch_size}")

#         N = pos_embed.shape[1] - 1  # Assume first token is class token
#         class_pos_embed = pos_embed[:, 0]
#         patch_pos_embed = pos_embed[:, 1:]
#         dim = pos_embed.shape[-1]

#         # Assuming square input for patches
#         w0 = int(math.sqrt(N))
#         h0 = int(math.sqrt(N))
#         # print(f"[DEBUG-w0]: {w0}, [DEBUG-h0]: {h0}")
        
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#             scale_factor=(w / w0, h / h0),
#             mode='bicubic',
#         )
#         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
#         # print(f"[DEBUG-pos_embed] {patch_pos_embed.shape}")
        
#         return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
#     except Exception as e:
#         raise ValueError(f"Error during interpolate_pos_encoding: {e}")


# # DINOV2 with Interpolation
# class Dinov2VisionTower(nn.Module):
#     def __init__(self, vision_tower, args, target_size=384):
#         super().__init__()
#         self.is_loaded = False
#         self.vision_tower_name = vision_tower
#         self.select_layer = -2
#         self.target_size = target_size

#         # Hardcoded parameters for testing
#         self.hardcoded_hidden_size = 1024
#         self.hardcoded_num_patches = (self.target_size // 14) ** 2  # Assuming patch size = 14

#         # Debugging: load immediately
#         self.load_model()

#     def load_model(self):
#         try:
#             self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
#             self.image_processor.crop_size = {'height': self.target_size, 'width': self.target_size}
#             self.image_processor.size = {'shortest_edge': self.target_size}

#             self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name)
#             self.vision_tower.requires_grad_(False)  # Freeze model

#             # Simulate `position_embeddings` initialization (hardcoded for testing)
#             pos_embed = torch.zeros(
#                 (1, self.hardcoded_num_patches + 1, self.hardcoded_hidden_size)
#             )
#             patch_size = 14  # Hardcoded patch size
#             pos_embed = interpolate_pos_encoding(pos_embed, self.target_size, self.target_size, patch_size)

#             # Assign back to the model
#             self.vision_tower.embeddings.position_embeddings = nn.Parameter(pos_embed)
#             self.is_loaded = True
#         except Exception as e:
#             raise ValueError(f"Error in load_model: {e}")

#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
#         image_features = image_features[:, 1:]  # Remove [CLS] token
#         return image_features

#     @torch.no_grad()
#     def forward(self, images):
#         if not self.is_loaded:
#             self.load_model()

#         images = images.float()  # Convert to float32
#         images = (images - images.min()) / (images.max() - images.min())  # Normalize to [0, 1]

#         processed_image = self.image_processor(images=images, return_tensors="pt").pixel_values
#         image_forward_out = self.vision_tower(
#             processed_image.to(device=self.device, dtype=torch.float32),
#             output_hidden_states=True, return_dict=True,
#         )
#         image_features = self.feature_select(image_forward_out).to(images.dtype).bfloat16()  # Back to BFloat16
#         return image_features

#     @property
#     def hidden_size(self):
#         return self.hardcoded_hidden_size

#     @property
#     def num_patches(self):
#         return self.hardcoded_num_patches

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only


import torch
import torch.nn as nn
import math
from transformers import AutoImageProcessor, AutoModel

# Helper function for position encoding interpolation
def interpolate_pos_encoding(pos_embed, w, h, patch_size):
    try:
        # Debug: Print input shapes
        print(f"[DEBUG] pos_embed shape: {pos_embed.shape}, target width: {w}, height: {h}, patch_size: {patch_size}")

        N = pos_embed.shape[1] - 1  # Assume first token is class token
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = pos_embed.shape[-1]

        # Assuming square input for patches
        w0 = int(math.sqrt(N))
        h0 = int(math.sqrt(N))
        # print(f"[DEBUG-w0]: {w0}, [DEBUG-h0]: {h0}")
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w / w0, h / h0),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        # print(f"[DEBUG-pos_embed] {patch_pos_embed.shape}")
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    except Exception as e:
        raise ValueError(f"Error during interpolate_pos_encoding: {e}")


# DINOV2 with Interpolation
class Dinov2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, target_size=384):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.target_size = target_size

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoModel.from_pretrained(self.vision_tower_name).config
            
        # Hardcoded parameters for testing
        self.hardcoded_hidden_size = 1024
        self.hardcoded_num_patches = (self.target_size // 14) ** 2  # Assuming patch size = 14

        # Debugging: load immediately
        self.load_model()

    def load_model(self):
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
            self.image_processor.crop_size = {'height': self.target_size, 'width': self.target_size}
            self.image_processor.size = {'shortest_edge': self.target_size}

            self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)  # Freeze model

            # Simulate `position_embeddings` initialization (hardcoded for testing)
            pos_embed = torch.zeros(
                (1, self.hardcoded_num_patches + 1, self.hardcoded_hidden_size)
            )
            patch_size = 14  # Hardcoded patch size
            pos_embed = interpolate_pos_encoding(pos_embed, self.target_size, self.target_size, patch_size)

            # Assign back to the model
            self.vision_tower.embeddings.position_embeddings = nn.Parameter(pos_embed)
            self.is_loaded = True
        except Exception as e:
            raise ValueError(f"Error in load_model: {e}")

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]  # Remove [CLS] token
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()

        images = images.float()  # Convert to float32
        images = (images - images.min()) / (images.max() - images.min())  # Normalize to [0, 1]

        processed_image = self.image_processor(images=images, return_tensors="pt").pixel_values
        image_forward_out = self.vision_tower(
            processed_image.to(device=self.device, dtype=torch.float32),
            output_hidden_states=True, return_dict=True,
        )
        image_features = self.feature_select(image_forward_out).to(images.dtype).bfloat16()  # Back to BFloat16
        return image_features

    @property
    def hidden_size(self):
        return self.hardcoded_hidden_size

    @property
    def num_patches(self):
        return self.hardcoded_num_patches

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only
