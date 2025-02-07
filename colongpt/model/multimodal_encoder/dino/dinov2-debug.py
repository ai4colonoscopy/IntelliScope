import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModelForPreTraining, AutoModel
from PIL import Image
import math
import requests

def interpolate_pos_encoding(x, w, h, patcah_size):
    npatch = x.shape[1] - 1
    N = pos_embed.shape[1] - 1
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // patcah_size
    print('[DEBUG-w0]', w0)
    h0 = h // patcah_size
    print('[DEBUG-h0]', h0)
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


image_processor = AutoImageProcessor.from_pretrained('cache/downloaded-weights/dinov2-large')
image_processor.crop_size = {'height': 384, 'width': 384}
image_processor.size = {'shortest_edge': 384}
model = AutoModel.from_pretrained('cache/downloaded-weights/dinov2-large')

pos_embed = model.state_dict()['embeddings.position_embeddings'] # ([1, 1370, 1024]
print('[DEBUG]', model.embeddings.position_embeddings.shape)

patcah_size = model.config.patch_size
pos_embed = interpolate_pos_encoding(pos_embed, 384, 384, patcah_size)
print('[DEBUG-pos_embed]', pos_embed.shape)

model.embeddings.position_embeddings = nn.Parameter(pos_embed)
print('[DEBUG]', model.embeddings.position_embeddings.shape)


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state[:, 1:]
print('last-hidden-state:', last_hidden_states.shape) 



# from transformers import AutoImageProcessor, AutoModel
# from PIL import Image
# import requests


# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
# processor.crop_size = {'height': 224, 'width': 224}
# processor.size = {'shortest_edge': 224}

# model = AutoModel.from_pretrained('facebook/dinov2-large')
# print('patch-embed:', model.embeddings.patch_embeddings) 

# inputs = processor(images=image, return_tensors="pt")
# print('input-image:', inputs.data['pixel_values'].shape) 

# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state[:, 1:]
# print('last-hidden-state:', last_hidden_states.shape) 

# python bunny/model/multimodal_encoder/dino/dino_debug.py


