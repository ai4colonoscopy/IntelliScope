import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
from PIL import Image
import warnings

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

device = 'cuda'  # or cpu
torch.set_default_device(device)

model_name = "ai4colonoscopy/ColonGPT"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # or float32 for cpu
    device_map='auto',
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keyword, tokenizer, input_ids):
        self.keyword_id = tokenizer(keyword).input_ids
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for keyword_id in self.keyword_id:
            if keyword_id in input_ids[0, -len(self.keyword_id):]:
                return True
        return False

prompt = "Categorize the object." 
text = f"USER: <image>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

image = Image.open('cache/examples/example2.png')
image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

stop_str = "<|endoftext|>"
stopping_criteria = KeywordsStoppingCriteria(stop_str, tokenizer, input_ids)

output_ids = model.generate(
    input_ids,
    images=image_tensor,
    do_sample=False,
    temperature=0,
    max_new_tokens=512,
    use_cache=True,
    stopping_criteria=[stopping_criteria]
)

outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace("<|endoftext|>", "").strip()
print(outputs)