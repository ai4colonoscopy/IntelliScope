import argparse
import datetime
import json
import os
import time
import gradio as gr
import requests
import hashlib
import pypandoc
import base64

from io import BytesIO

from colongpt.conversation import (default_conversation, conv_templates, SeparatorStyle)
from colongpt.constants import LOGDIR
from colongpt.util.utils import (build_logger, server_error_msg, violates_moderation, moderation_msg)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "ColonGPT Client"}

no_change_btn = gr.update()
enable_btn = gr.update(interactive=True)
disable_btn = gr.update(interactive=False)

priority = {
    "ColonGPT": "aaaaaaa",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.update(
                value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.update(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def save_conversation(conversation):
    print("save_conversation_wrapper is called")
    html_content = "<html><body>"

    for role, message in conversation.messages:
        if isinstance(message, str):  # only text
            html_content += f"<p><b>{role}</b>: {message}</p>"
        elif isinstance(message, tuple):  # text+image
            text, image_obj, _ = message

            # add text
            if text:
                html_content += f"<p><b>{role}</b>: {text}</p>"

            # add image
            buffered = BytesIO()
            image_obj.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode()
            html_content += f'<img src="data:image/png;base64,{encoded_image}" /><br>'

    html_content += "</body></html>"

    doc_path = "./conversation.docx"
    pypandoc.convert_text(html_content, 'docx', format='html', outputfile=doc_path,
                          extra_args=["-M2GB", "+RTS", "-K64m", "-RTS"])
    return doc_path


def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    logger.info(f"Input Text: {text}")
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        conv_mode = "colongpt"

        new_state = conv_templates[conv_mode].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    logger.info(f"Processed Input Text: {state.messages[-2][1]}")
    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
                        json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), enable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.PLAIN, ] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()
    print('=========> get_images')
    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
    print('=========> state', state.messages[-1][-1])

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=1000)
        print("====> response ok")
        print("====> response dir", dir(response))
        print("====> response", response)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (enable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (enable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
""")

block_css = """
.centered {
    text-align: center;
}
#buttons button {
    min-width: min(120px,100%);
}
#file-downloader {
    min-width: min(120px,100%);
    height: 50px;
}
#imagebox {
    height: 280px;
    object-fit: contain;
}
"""

def trigger_download(doc_path):
    return doc_path


def build_demo(embed_mode):         
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="ColonGPT",
                   theme=gr.themes.Default(primary_hue="blue", secondary_hue="lime"),
                   css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            
            gr.Markdown("""
            # 🤖 ColonGPT

            [🏠[Home](https://github.com/ai4colonoscopy)] | [📜[Paper](https://arxiv.org)] | [💻[Code](https://github.com/ai4colonoscopy/ColonGPT-tmp)]
            """)

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                        allow_custom_value=True
                    )

                imagebox = gr.Image(type="pil", elem_id="imagebox")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = 'cache/examples'
                gr.Examples(examples=[
                    [f"{cur_dir}/web_example_1.jpeg", "Categorize the object."],
                    # [f"{cur_dir}/web_example_2.jpeg", "Offer a thorough explanation of the image."],
                    [f"{cur_dir}/web_example_3.png", "Could you give the position of polyp?"],
                    [f"{cur_dir}/web_example_4.jpg", "Could you provide the category for {<644><498><1039><1016>}?"],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                            label="Temperature", )
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", )
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True,
                                                  label="Max output tokens", )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="ColonGPT Chatbot",
                                     avatar_images=[f"{cur_dir}/web_doctor.png", f"{cur_dir}/web_robot.png"],
                                     height=675)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")


        if not embed_mode:
            gr.Markdown(tos_markdown)
        url_params = gr.JSON(visible=False)


        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot]
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params,
                queue=False
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
                        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()
    logger.info(args)
    demo = build_demo(args.embed)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=True,
        max_threads=10
    )
