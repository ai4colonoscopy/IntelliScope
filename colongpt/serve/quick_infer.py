import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

from colongpt.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from colongpt.conversation import conv_templates, SeparatorStyle
from colongpt.model.builder import load_pretrained_model
from colongpt.util.utils import disable_torch_init
from colongpt.util.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


model_path = "ai4colonoscopy/ColonGPT-v1"
model_base = "microsoft/phi-1_5"
model_type = "phi-1.5"
image_file = "cache/examples/example2.png"
device = "cuda"  # or "cpu"
conv_mode = "colongpt"


disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, model_type, False, False, device=device)

conv = conv_templates[conv_mode].copy()
roles = conv.roles

image = Image.open(image_file).convert('RGB')
image_tensor = process_images([image], image_processor, model.config)
if type(image_tensor) is list:
    image_tensor = [image.to(model.device, dtype=model.dtype) for image in image_tensor]
else:
    image_tensor = image_tensor.to(model.device, dtype=model.dtype)

inp = DEFAULT_IMAGE_TOKEN + '\n' + "Categorize the object."
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=512,
        use_cache=True,
        stopping_criteria=[stopping_criteria]
    )

outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace("<|endoftext|>", "").strip()

print(f"{outputs}")
