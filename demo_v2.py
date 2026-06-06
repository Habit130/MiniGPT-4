import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "eval_configs" / "minigptv2_checkpoint_0_eval.yaml"
DEFAULT_LLAMA_MODEL_PATH = PROJECT_ROOT / "models" / "vicuna-7b"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoint_0.pth"
DEFAULT_EVA_VIT_PATH = PROJECT_ROOT / "models" / "eva_vit_g.pth"
DEFAULT_TRANSLATION_MODEL_PATH = PROJECT_ROOT / "models" / "opus-mt-en-zh"
EXAMPLE_DIR = PROJECT_ROOT / "assets" / "examples"


def parse_args():
    parser = argparse.ArgumentParser(description="智农卫士作物病害诊断网页")
    parser.add_argument("--cfg-path", default=str(DEFAULT_CONFIG_PATH),
                        help="path to configuration file.")
    parser.add_argument("--llama-model", default=str(DEFAULT_LLAMA_MODEL_PATH),
                        help="path to the local Vicuna 7B model directory.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH),
                        help="path to the fine-tuned MiniGPT checkpoint.")
    parser.add_argument("--eva-vit", default=str(DEFAULT_EVA_VIT_PATH),
                        help="path to the EVA ViT-G checkpoint.")
    parser.add_argument("--translation-model", default=str(DEFAULT_TRANSLATION_MODEL_PATH),
                        help="path to the local English-to-Chinese translation model.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--server-name", default="127.0.0.1",
                        help="network interface used by the Gradio server.")
    parser.add_argument("--server-port", type=int, default=7860,
                        help="port used by the Gradio server.")
    parser.add_argument("--share", action="store_true",
                        help="create a public Gradio share link.")
    parser.add_argument("--inbrowser", action="store_true",
                        help="open the web interface in the default browser.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
os.chdir(PROJECT_ROOT)

required_paths = {
    "Vicuna 模型配置": Path(args.llama_model) / "config.json",
    "微调权重": Path(args.checkpoint),
    "EVA ViT-G 权重": Path(args.eva_vit),
    "中英翻译模型配置": Path(args.translation_model) / "config.json",
}
missing_paths = [
    f"- {label}: {path}"
    for label, path in required_paths.items()
    if not path.exists()
]
if missing_paths:
    raise FileNotFoundError(
        "缺少网页推理所需的模型文件，请按照 README.md 放置权重：\n"
        + "\n".join(missing_paths)
    )

os.environ["MINIGPT_EVA_VIT_PATH"] = str(Path(args.eva_vit).resolve())
cfg = Config(args)

device = 'cuda:{}'.format(args.gpu_id)

model_config = cfg.model_cfg
model_config.llama_model = str(Path(args.llama_model).resolve())
model_config.ckpt = str(Path(args.checkpoint).resolve())
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
bounding_box_size = 100

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

model = model.eval()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)


def extract_substrings(string):
    # first check if there is no-finished bracket
    index = string.rfind('}')
    if index != -1:
        string = string[:index + 1]

    pattern = r'<p>(.*?)\}(?!<)'
    matches = re.findall(pattern, string)
    substrings = [match for match in matches]

    return substrings


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def save_tmp_img(visual_img):
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = "/tmp/gradio" + file_name
    visual_img.save(file_path)
    return file_path


def mask2bbox(mask):
    if mask is None:
        return ''
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum():
        # Get the top, bottom, left, and right boundaries
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
    else:
        bbox = ''

    return bbox


def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']

    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)

    return text


def reverse_escape(text):
    md_chars = ['\\<', '\\>']

    for char in md_chars:
        text = text.replace(char, char[1:])

    return text


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (210, 210, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for
    color_id, color in enumerate(colors)
}

used_colors = colors


def visualize_all_bbox_together(image, generation):
    if image is None:
        return None, ''

    generation = html.unescape(generation)

    image_width, image_height = image.size
    image = image.resize([500, int(500 / image_width * image_height)])
    image_width, image_height = image.size

    string_list = extract_substrings(generation)
    if string_list:  # it is grounding or detection
        mode = 'all'
        entities = defaultdict(list)
        i = 0
        j = 0
        for string in string_list:
            try:
                obj, string = string.split('</p>')
            except ValueError:
                print('wrong string: ', string)
                continue
            bbox_list = string.split('<delim>')
            flag = False
            for bbox_string in bbox_list:
                integers = re.findall(r'-?\d+', bbox_string)
                if len(integers) == 4:
                    x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                    left = x0 / bounding_box_size * image_width
                    bottom = y0 / bounding_box_size * image_height
                    right = x1 / bounding_box_size * image_width
                    top = y1 / bounding_box_size * image_height

                    entities[obj].append([left, bottom, right, top])

                    j += 1
                    flag = True
            if flag:
                i += 1
    else:
        integers = re.findall(r'-?\d+', generation)

        if len(integers) == 4:  # it is refer
            mode = 'single'

            entities = list()
            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
            left = x0 / bounding_box_size * image_width
            bottom = y0 / bounding_box_size * image_height
            right = x1 / bounding_box_size * image_width
            top = y1 / bounding_box_size * image_height
            entities.append([left, bottom, right, top])
        else:
            # don't detect any valid bbox to visualize
            return None, ''

    if len(entities) == 0:
        return None, ''

    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)

    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):

        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    indices = list(range(len(entities)))

    new_image = image.copy()

    previous_bboxes = []
    # size of text
    text_size = 0.5
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 2
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 2

    # num_bboxes = sum(len(x[-1]) for x in entities)
    used_colors = colors  # random.sample(colors, k=num_bboxes)

    color_id = -1
    for entity_idx, entity_name in enumerate(entities):
        if mode == 'single' or mode == 'identify':
            bboxes = entity_name
            bboxes = [bboxes]
        else:
            bboxes = entities[entity_name]
        color_id += 1
        for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
            skip_flag = False
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm), int(y1_norm), int(x2_norm), int(y2_norm)

            color = used_colors[entity_idx % len(used_colors)]  # tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            if mode == 'all':
                l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o

                if y1 < text_height + text_offset_original + 2 * text_spaces:
                    y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                    x1 = orig_x1 + r_o

                # add text background
                (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size,
                                                               text_line)
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (
                            text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

                for prev_bbox in previous_bboxes:
                    if computeIoU((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']) > 0.95 and \
                            prev_bbox['phrase'] == entity_name:
                        skip_flag = True
                        break
                    while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']):
                        text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                        text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                        y1 += (text_height + text_offset_original + 2 * text_spaces)

                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                            text_bg_y2 = image_h
                            y1 = image_h
                            break
                if not skip_flag:
                    alpha = 0.5
                    for i in range(text_bg_y1, text_bg_y2):
                        for j in range(text_bg_x1, text_bg_x2):
                            if i < image_h and j < image_w:
                                if j < text_bg_x1 + 1.35 * c_width:
                                    # original color
                                    bg_color = color
                                else:
                                    # white
                                    bg_color = [255, 255, 255]
                                new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(
                                    np.uint8)

                    cv2.putText(
                        new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces),
                        cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
                    )

                    previous_bboxes.append(
                        {'bbox': (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), 'phrase': entity_name})

    if mode == 'all':
        def color_iterator(colors):
            while True:
                for color in colors:
                    yield color

        color_gen = color_iterator(colors)

        # Add colors to phrases and remove <p></p>
        def colored_phrases(match):
            phrase = match.group(1)
            color = next(color_gen)
            return f'<span style="color:rgb{color}">{phrase}</span>'

        generation = re.sub(r'{<\d+><\d+><\d+><\d+>}|<delim>', '', generation)
        generation_colored = re.sub(r'<p>(.*?)</p>', colored_phrases, generation)
    else:
        generation_colored = ''

    pil_image = Image.fromarray(new_image)
    return pil_image, generation_colored


def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return (
        None,
        gr.update(value=None, interactive=True),
        gr.update(
            placeholder='请先上传作物图片，再描述你观察到的问题',
            interactive=True,
        ),
        chat_state,
        img_list,
    )


def image_upload_trigger(upload_flag, replace_flag, img_list):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    upload_flag = 1
    if img_list:
        replace_flag = 1
    return upload_flag, replace_flag


def example_trigger(text_input, image, upload_flag, replace_flag, img_list):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    upload_flag = 1
    if img_list or replace_flag == 1:
        replace_flag = 1

    return upload_flag, replace_flag


def gradio_ask(user_message, chatbot, chat_state, gr_img, img_list, upload_flag, replace_flag):
    if len(user_message) == 0:
        text_box_show = '请输入需要咨询的问题'
    else:
        text_box_show = ''

    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']
    else:
        mask = None

    if '[identify]' in user_message:
        # check if user provide bbox in the text input
        integers = re.findall(r'-?\d+', user_message)
        if len(integers) != 4:  # no bbox in text
            bbox = mask2bbox(mask)
            user_message = user_message + bbox

    if chat_state is None:
        chat_state = CONV_VISION.copy()

    if upload_flag:
        if replace_flag:
            chat_state = CONV_VISION.copy()  # new image, reset everything
            replace_flag = 0
            chatbot = []
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        upload_flag = 0

    known_task_tags = (
        "[grounding]",
        "[refer]",
        "[detection]",
        "[identify]",
        "[vqa]",
    )
    model_prompt = user_message
    if not any(tag in user_message for tag in known_task_tags):
        model_prompt = "[vqa] " + user_message

    model_message = (
        model_prompt
        + "\nIMPORTANT OUTPUT LANGUAGE: Respond only in natural Simplified Chinese."
        + " Do not output English sentences."
        + " Base the answer on the image. If disease is involved, state the likely"
        + " disease, visible evidence, and practical management suggestions."
        + " If uncertain, say so clearly and do not invent a diagnosis."
    )
    chat.ask(model_message, chat_state)

    chatbot = chatbot + [[user_message, None]]

    if '[identify]' in user_message:
        visual_img, _ = visualize_all_bbox_together(gr_img, user_message)
        if visual_img is not None:
            file_path = save_tmp_img(visual_img)
            chatbot = chatbot + [[(file_path,), None]]

    return text_box_show, chatbot, chat_state, img_list, upload_flag, replace_flag


def needs_chinese_translation(text):
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    latin_count = len(re.findall(r'[A-Za-z]', text))
    return latin_count > max(12, chinese_count * 2)


def translate_answer_to_chinese(answer):
    batch = translation_tokenizer(
        [answer],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        output_tokens = translation_model.generate(
            **batch,
            max_new_tokens=256,
            num_beams=4,
        )
    return translation_tokenizer.batch_decode(
        output_tokens,
        skip_special_tokens=True,
    )[0].strip()


def gradio_answer(chatbot, chat_state, img_list, temperature):
    if len(img_list) > 0 and not isinstance(img_list[0], torch.Tensor):
        chat.encode_img(img_list)

    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              temperature=temperature,
                              max_new_tokens=180,
                              max_length=1200)[0]
    if needs_chinese_translation(llm_message):
        translated_message = translate_answer_to_chinese(llm_message)
        if not needs_chinese_translation(translated_message):
            llm_message = translated_message
            chat_state.messages[-1][1] = translated_message

    chatbot[-1][1] = llm_message
    return chatbot, chat_state


def gradio_stream_answer(chatbot, chat_state, img_list, temperature):
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)
    streamer = chat.stream_answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=temperature,
                                  max_new_tokens=500,
                                  max_length=2000)
    output = ''
    for new_output in streamer:
        escapped = escape_markdown(new_output)
        output += escapped
        chatbot[-1][1] = output
        yield chatbot, chat_state
    chat_state.messages[-1][1] = '</s>'
    return chatbot, chat_state


def gradio_visualize(chatbot, gr_img):
    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']

    unescaped = reverse_escape(chatbot[-1][1])
    visual_img, generation_color = visualize_all_bbox_together(gr_img, unescaped)
    if visual_img is not None:
        if len(generation_color):
            chatbot[-1][1] = generation_color
        file_path = save_tmp_img(visual_img)
        chatbot = chatbot + [[None, (file_path,)]]

    return chatbot


def gradio_taskselect(idx):
    prompt_list = [
        '',
        '[vqa] 请判断作物是否健康；如有异常，指出最可能的病害。',
        '[vqa] 请描述叶片、茎秆或果实上可见的异常症状。',
        '[vqa] 请根据图像分析可能的发病原因。',
        '[vqa] 请给出针对该病害的田间管理与防治建议。',
        '[vqa] 请判断这是什么作物，并说明当前生长状态。'
    ]
    instruct_list = [
        '自由咨询：可询问病害、症状、诱因或管理建议。',
        '健康诊断：快速判断健康状态和最可能的病害类型。',
        '症状分析：重点描述斑点、变色、霉层、卷曲等视觉特征。',
        '病因研判：结合可见症状分析可能的病原或环境诱因。',
        '防治建议：获取清园、栽培管理和用药方向等建议。',
        '作物识别：识别作物种类并概括当前生长状态。',
    ]
    return prompt_list[idx], instruct_list[idx]




chat = Chat(model, vis_processor, device=device)
translation_model_path = str(Path(args.translation_model).resolve())
translation_tokenizer = AutoTokenizer.from_pretrained(
    translation_model_path,
    local_files_only=True,
)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    translation_model_path,
    local_files_only=True,
).eval()

page_header = """
<section class="hero">
  <div class="hero-copy">
    <div class="eyebrow">智慧植保 · 智能视觉诊断</div>
    <h1>智农卫士</h1>
    <p class="hero-subtitle">面向田间场景的作物病害图像问答与辅助诊断平台</p>
    <div class="hero-badges">
      <span>55+ 健康与病害类别</span>
      <span>4,213 个验证样本</span>
      <span>本地私有化推理</span>
    </div>
  </div>
  <div class="hero-mark" aria-hidden="true">
    <div class="leaf leaf-a"></div>
    <div class="leaf leaf-b"></div>
    <div class="pulse"></div>
  </div>
</section>
"""

workflow = """
<div class="workflow">
  <div class="workflow-item"><b>01</b><span>上传清晰的作物图片</span></div>
  <div class="workflow-arrow">→</div>
  <div class="workflow-item"><b>02</b><span>选择诊断模式或直接提问</span></div>
  <div class="workflow-arrow">→</div>
  <div class="workflow-item"><b>03</b><span>获取症状研判与管理建议</span></div>
</div>
"""

notice = """
<div class="notice-card">
  <div class="notice-title">使用提示</div>
  <ul>
    <li>建议上传光线均匀、主体清晰的叶片、茎秆或果实近照。</li>
    <li>可继续追问症状依据、可能诱因和田间管理建议。</li>
    <li>结果仅用于辅助判断，重大疫情与用药方案请咨询当地植保人员。</li>
  </ul>
</div>
"""

footer = """
<div class="site-footer">
  <span>智农卫士 · 作物病害视觉诊断平台</span>
  <span>AI 辅助研判，不替代专业植保诊断</span>
</div>
"""

custom_css = """
:root {
  --brand-950: #12372a;
  --brand-800: #1c563d;
  --brand-650: #2f7654;
  --brand-500: #4f9b70;
  --brand-100: #e9f4ec;
  --cream: #f7f5ee;
  --ink: #173128;
  --muted: #61746b;
  --line: #dce7df;
}

body, .gradio-container {
  background:
    radial-gradient(circle at 8% 0%, rgba(118, 166, 126, .18), transparent 28rem),
    radial-gradient(circle at 96% 8%, rgba(221, 190, 105, .14), transparent 25rem),
    #f5f7f3 !important;
  color: var(--ink);
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif !important;
}

.gradio-container {
  max-width: 1380px !important;
  padding: 28px 30px 18px !important;
}

.hero {
  position: relative;
  overflow: hidden;
  display: flex;
  min-height: 250px;
  align-items: center;
  justify-content: space-between;
  padding: 44px 54px;
  border: 1px solid rgba(255,255,255,.16);
  border-radius: 28px;
  background:
    linear-gradient(118deg, rgba(13,49,36,.98), rgba(29,93,62,.94)),
    url("");
  box-shadow: 0 24px 60px rgba(25, 66, 48, .18);
  color: white;
}

.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
  background-size: 28px 28px;
  mask-image: linear-gradient(90deg, black, transparent 80%);
}

.hero-copy { position: relative; z-index: 2; max-width: 760px; }
.eyebrow {
  margin-bottom: 14px;
  color: #c9ead3 !important;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: .2em;
}
.hero h1 {
  margin: 0 !important;
  color: white !important;
  font-size: clamp(42px, 5vw, 68px) !important;
  font-weight: 800 !important;
  letter-spacing: .08em;
  line-height: 1.05 !important;
}
.hero-subtitle {
  margin: 18px 0 24px !important;
  color: rgba(255,255,255,.88) !important;
  font-size: 18px;
}
.hero-badges { display: flex; flex-wrap: wrap; gap: 10px; }
.hero-badges span {
  padding: 8px 13px;
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 999px;
  background: rgba(255,255,255,.08);
  color: #f0faf3 !important;
  font-size: 13px;
  backdrop-filter: blur(8px);
}

.hero-mark { position: relative; width: 210px; height: 170px; margin-right: 24px; }
.leaf {
  position: absolute;
  width: 86px;
  height: 135px;
  border-radius: 90% 8% 90% 8%;
  background: linear-gradient(145deg, #a8d5a2, #4c9a70);
  box-shadow: inset -12px -10px 30px rgba(13,49,36,.26);
}
.leaf::after {
  content: "";
  position: absolute;
  left: 50%;
  top: 13px;
  width: 2px;
  height: 110px;
  transform: rotate(36deg);
  transform-origin: top;
  background: rgba(255,255,255,.45);
}
.leaf-a { left: 32px; top: 18px; transform: rotate(-38deg); }
.leaf-b { right: 26px; bottom: 6px; transform: rotate(45deg) scale(.78); opacity: .78; }
.pulse {
  position: absolute;
  inset: 33px;
  border: 1px solid rgba(206,237,211,.28);
  border-radius: 50%;
  box-shadow: 0 0 0 24px rgba(206,237,211,.06), 0 0 0 48px rgba(206,237,211,.035);
}

.workflow {
  display: grid;
  grid-template-columns: 1fr auto 1fr auto 1fr;
  align-items: center;
  gap: 16px;
  margin: 22px 0;
  padding: 18px 24px;
  border: 1px solid var(--line);
  border-radius: 20px;
  background: rgba(255,255,255,.82);
  box-shadow: 0 8px 28px rgba(28,71,51,.06);
}
.workflow-item { display: flex; align-items: center; gap: 12px; color: var(--muted); }
.workflow-item b {
  display: grid; width: 36px; height: 36px; place-items: center;
  border-radius: 12px; background: var(--brand-100); color: var(--brand-800);
}
.workflow-item span { font-weight: 600; }
.workflow-arrow { color: #a2b5aa; font-size: 20px; }

#diagnosis-panel, #assistant-panel {
  padding: 20px !important;
  border: 1px solid var(--line) !important;
  border-radius: 24px !important;
  background: rgba(255,255,255,.94) !important;
  box-shadow: 0 16px 44px rgba(30,72,53,.08) !important;
}

#crop-image {
  overflow: hidden;
  border: 1px dashed #a8c4b2 !important;
  border-radius: 18px !important;
  background: #f8fbf8 !important;
}
#crop-image .wrap { min-height: 430px; }

#assistant-chat {
  min-height: 420px;
  border: 1px solid #e0e9e3 !important;
  border-radius: 18px !important;
  background: linear-gradient(180deg, #fbfdfb, #f4f8f5) !important;
}

.panel-heading h3 { margin: 0 0 4px !important; color: var(--brand-950) !important; }
.panel-heading p { margin: 0 0 14px !important; color: var(--muted); font-size: 14px; }

#mode-grid { margin-top: 10px; }
#mode-grid .label-wrap { color: var(--brand-950); font-weight: 700; }
#mode-grid table { border-collapse: separate !important; border-spacing: 8px !important; }
#mode-grid td {
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  background: #f7faf7 !important;
  color: var(--brand-800) !important;
  font-weight: 600;
}
#mode-grid td:hover { border-color: var(--brand-500) !important; background: var(--brand-100) !important; }

#prompt-box textarea {
  min-height: 78px !important;
  border: 1px solid #cfded4 !important;
  border-radius: 15px !important;
  background: white !important;
  font-size: 15px !important;
}
#send-button {
  min-width: 108px !important;
  border: none !important;
  border-radius: 14px !important;
  background: linear-gradient(135deg, var(--brand-800), var(--brand-500)) !important;
  box-shadow: 0 8px 20px rgba(35,105,72,.22) !important;
  color: white !important;
  font-weight: 700 !important;
}
#restart-button {
  border: 1px solid #cfe0d4 !important;
  border-radius: 13px !important;
  background: #f3f8f4 !important;
  color: var(--brand-800) !important;
}

.notice-card {
  margin-top: 14px;
  padding: 17px 19px;
  border-radius: 16px;
  background: linear-gradient(135deg, #f2f7ef, #fffaf0);
  color: var(--muted);
  font-size: 13px;
}
.notice-title { margin-bottom: 8px; color: var(--brand-800); font-weight: 800; }
.notice-card ul { margin: 0; padding-left: 18px; }
.notice-card li { margin: 5px 0; }

.examples-title { margin: 30px 0 10px; }
.examples-title h2 { margin-bottom: 5px !important; color: var(--brand-950) !important; }
.examples-title p { color: var(--muted); }
.gradio-examples { padding: 16px !important; border: 1px solid var(--line) !important; border-radius: 20px !important; background: white !important; }

.site-footer {
  display: flex; justify-content: space-between; gap: 16px;
  margin-top: 24px; padding: 20px 4px 4px;
  border-top: 1px solid var(--line); color: #76897f; font-size: 12px;
}

footer { display: none !important; }

@media (max-width: 900px) {
  .gradio-container { padding: 14px !important; }
  .hero { min-height: auto; padding: 32px 26px; }
  .hero-mark { display: none; }
  .workflow { grid-template-columns: 1fr; }
  .workflow-arrow { display: none; }
  .site-footer { flex-direction: column; }
}
"""

text_input = gr.Textbox(
    placeholder='请先上传作物图片，再描述你观察到的问题',
    interactive=True,
    show_label=False,
    container=False,
    scale=8,
    elem_id="prompt-box",
)
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
).set(
    button_primary_background_fill="#256b49",
    button_primary_background_fill_hover="#1c563d",
    block_title_text_color="#173128",
)

with gr.Blocks(
    theme=theme,
    css=custom_css,
    title="智农卫士 · 作物病害视觉诊断",
    analytics_enabled=False,
) as demo:
    gr.HTML(page_header)
    gr.HTML(workflow)

    with gr.Row():
        with gr.Column(scale=5, elem_id="diagnosis-panel"):
            gr.Markdown(
                "### 上传作物图像\n建议使用清晰近照，确保病斑、叶脉或霉层等特征可见。",
                elem_classes="panel-heading",
            )
            image = gr.Image(
                type="pil",
                tool='sketch',
                brush_radius=20,
                label="作物图像",
                elem_id="crop-image",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.6,
                step=0.1,
                interactive=True,
                label="回答灵活度",
                info="数值越低，回答越稳定；数值越高，表达越丰富。",
            )

            clear = gr.Button("清空并重新诊断", elem_id="restart-button")

            gr.HTML(notice)

        with gr.Column(scale=7, elem_id="assistant-panel"):
            gr.Markdown(
                "### 智能诊断助手\n选择分析模式，或直接输入你最关心的问题。",
                elem_classes="panel-heading",
            )
            chat_state = gr.State(value=None)
            img_list = gr.State(value=[])
            chatbot = gr.Chatbot(
                label='诊断对话',
                elem_id="assistant-chat",
                bubble_full_width=False,
            )

            dataset = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=[
                    ['自由咨询'],
                    ['健康诊断'],
                    ['症状分析'],
                    ['病因研判'],
                    ['防治建议'],
                    ['作物识别'],
                ],
                type="index",
                label='诊断模式',
                elem_id="mode-grid",
            )
            task_inst = gr.Markdown('自由咨询：可询问病害、症状、诱因或管理建议。')
            with gr.Row():
                text_input.render()
                send = gr.Button(
                    "开始诊断",
                    variant='primary',
                    size='sm',
                    scale=1,
                    elem_id="send-button",
                )

    upload_flag = gr.State(value=0)
    replace_flag = gr.State(value=0)
    image.upload(image_upload_trigger, [upload_flag, replace_flag, img_list], [upload_flag, replace_flag])

    gr.HTML(
        """
        <div class="examples-title">
          <h2>典型病害示例</h2>
          <p>点击任一示例即可载入真实作物图像与诊断问题。</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column():
            gr.Examples(examples=[
                [str(EXAMPLE_DIR / "tomato_late_blight.jpg"),
                 "这株番茄可能患有什么病害？请说明判断依据。", upload_flag, replace_flag, img_list],
                [str(EXAMPLE_DIR / "apple_cedar_rust.jpg"),
                 "请诊断这片苹果叶的异常，并给出管理建议。", upload_flag, replace_flag, img_list],
                [str(EXAMPLE_DIR / "corn_northern_leaf_blight.jpg"),
                 "这片玉米叶出现了什么问题？", upload_flag, replace_flag, img_list],
                [str(EXAMPLE_DIR / "rice_blast.jpg"),
                 "请判断水稻叶片的病害类型和典型症状。", upload_flag, replace_flag, img_list],
            ], inputs=[image, text_input, upload_flag, replace_flag, img_list], fn=example_trigger,
                outputs=[upload_flag, replace_flag], label="粮食与果树病害")
        with gr.Column():
            gr.Examples(examples=[
                [str(EXAMPLE_DIR / "tomato_yellow_leaf_curl.jpg"),
                 "番茄叶片卷曲发黄，最可能是什么原因？", upload_flag, replace_flag, img_list],
                [str(EXAMPLE_DIR / "potato_early_blight.jpg"),
                 "请分析这片马铃薯叶的病斑特征。", upload_flag, replace_flag, img_list],
                [str(EXAMPLE_DIR / "strawberry_leaf_scorch.jpg"),
                 "草莓叶片出现异常，应该如何进行田间管理？", upload_flag, replace_flag, img_list],
                [str(EXAMPLE_DIR / "bell_pepper_healthy.jpg"),
                 "请判断这片甜椒叶是否健康，并说明依据。", upload_flag, replace_flag, img_list],
            ], inputs=[image, text_input, upload_flag, replace_flag, img_list], fn=example_trigger,
                outputs=[upload_flag, replace_flag], label="蔬菜与健康对照")

    dataset.click(
        gradio_taskselect,
        inputs=[dataset],
        outputs=[text_input, task_inst],
        show_progress="hidden",
        postprocess=False,
        queue=False,
    )

    text_input.submit(
        gradio_ask,
        [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
        [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag], queue=False
    ).success(
        gradio_answer,
        [chatbot, chat_state, img_list, temperature],
        [chatbot, chat_state]
    ).success(
        gradio_visualize,
        [chatbot, image],
        [chatbot],
        queue=False,
    )

    send.click(
        gradio_ask,
        [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
        [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag], queue=False
    ).success(
        gradio_answer,
        [chatbot, chat_state, img_list, temperature],
        [chatbot, chat_state]
    ).success(
        gradio_visualize,
        [chatbot, image],
        [chatbot],
        queue=False,
    )

    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, chat_state, img_list], queue=False)
    gr.HTML(footer)

demo.launch(
    share=args.share,
    server_name=args.server_name,
    server_port=args.server_port,
    enable_queue=True,
    allowed_paths=[str(EXAMPLE_DIR)],
    inbrowser=args.inbrowser,
)
