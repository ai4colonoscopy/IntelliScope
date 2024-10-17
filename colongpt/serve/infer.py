import json
import os
import argparse
import torch
from PIL import Image
from transformers import TextStreamer

from colongpt.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from colongpt.conversation import conv_templates, SeparatorStyle
from colongpt.model.builder import load_pretrained_model
from colongpt.util.utils import disable_torch_init
from colongpt.util.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


# for colon300k image files
def load_image_from_json(json_file, image_dir, i):
    with open(json_file, 'r') as f:
        data_list = json.load(f)
    image_id = data_list[i]['id']
    image_file = os.path.join(image_dir, image_id)
    image = Image.open(image_file).convert('RGB')
    return image

# inference: cli.py
def main(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.model_type, args.load_8bit,
                                                                           args.load_4bit, device=args.device)

    conv_mode = "colongpt"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                              args.conv_mode,
                                                                                                              args.conv_mode))
    else:
        args.conv_mode = conv_mode

    with open(args.json_file, 'r') as f:
        data_list = json.load(f)

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            predicted_data = json.load(f)
        predicted_ids = [item['id'] for item in predicted_data]
    else:
        predicted_ids = []

    with open(args.output_path, 'a') as f:
        f.write('[')

    for i in range(len(data_list)):
        item = data_list[i]
        image_id = item['id']

        if image_id in predicted_ids:
            continue

        conversations = item['conversations']

        image = load_image_from_json(args.json_file, args.image_dir, i)

        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=model.dtype) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=model.dtype)

        # 开启新对话
        conv = conv_templates[args.conv_mode].copy()
        roles = conv.roles

        # 使用human 的 value 作为 prompt
        conv.append_message(conv.roles[0], conversations[0]['value'])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace("<|endoftext|>", "").strip()

        conversations.append({
            "from": "prediction",
            "value": outputs
        })

        with open(args.output_path, 'a') as f:
            json.dump({
                "id": image_id,
                "image": item['image'],
                "conversations": conversations
            }, f, indent=4)
            f.write(',')

    with open(args.output_path, 'rb+') as f:
        f.seek(-1, os.SEEK_END)
        f.truncate()
    with open(args.output_path, 'a') as f:
        f.write(']')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--conv_mode", type=str, default=None)
    args = parser.parse_args()
    main(args)
