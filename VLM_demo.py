from openai import OpenAI
import base64
from PIL import Image
import json
import math
import numpy as np
import requests
import json_numpy
import matplotlib.pyplot as plt
import time
import torch
from utils import load_prompt
import sys
sys.path.append("EdgeSAM/")
from EdgeSAM.edge_sam import sam_model_registry, SamPredictor

json_numpy.patch()
sam_checkpoint = "EdgeSAM/weights/edge_sam.pth"
model_type = "edge_sam"

device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def use_sam(frame,bbox):
    frame = np.array(frame)
    predictor.set_image(frame)
    input_boxes = torch.tensor(bbox, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        num_multimask_outputs=1,
    )
    return masks.cpu().numpy()

def write_state(output_json_path,state,lock):
    while True:
        with lock:
            with open(output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(state, json_file)
                break
        time.sleep(0.1)
        

def read_state(state_json_path,lock):
    while True:
        with lock:
            with open(state_json_path, 'r', encoding='utf-8') as json_file:
                loaded_state = json.load(json_file)
                break
        time.sleep(0.1)

    return loaded_state

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def resize_bbox_to_original(bbox_list, original_size, resized_size):
    # 获取原图和调整后图像的尺寸
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    
    # 计算缩放比例
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    
    # 将边界框放大回原图像尺寸
    resized_bbox_list = []
    for detection in bbox_list:
        bbox = detection['bbox']
        
        # 放大边界框坐标
        x1, y1, x2, y2 = bbox
        new_x1 = int(x1 * scale_x)
        new_y1 = int(y1 * scale_y)
        new_x2 = int(x2 * scale_x)
        new_y2 = int(y2 * scale_y)
        
        # 更新到放大后的边界框列表
        resized_bbox_list.append({
            'bbox': [new_x1, new_y1, new_x2, new_y2],
            'label': detection['label']
        })
    
    return resized_bbox_list


def smart_resize(image_path, factor = 28, vl_high_resolution_images = False):
    # 打开指定的PNG图片文件
    image = Image.open(image_path)

    # 获取图片的原始尺寸
    height = image.height
    width = image.width
    # 将高度调整为28的整数倍
    h_bar = round(height / factor) * factor
    # 将宽度调整为28的整数倍
    w_bar = round(width / factor) * factor
    
    # 图像的Token下限：4个Token
    min_pixels = 28 * 28 * 4
    
    # 根据vl_high_resolution_images参数确定图像的Token上限
    if not vl_high_resolution_images:
        max_pixels = 1280 * 28 * 28
    else:
        max_pixels = 16384 * 28 * 28
        
    # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return w_bar , h_bar


def get_world_bboxs_list(image_path,objects):

    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene, you need to detect {objects}. Detect all objects in the image and return their locations in the form of coordinates, don't give up any information about the details. The format of output should be like" +"{“bbox”: [x1, y1, x2, y2], “label”: the name of this object in English.} not {“bbox_2d”: [x1, y1, x2, y2], “label”: the name of this object in Chinese}"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}]
        )

    bbox_list_str = completion.choices[0].message.content[7:-3]

    bbox_list = json.loads(bbox_list_str)

    # 打开图片
    image = Image.open(image_path)
    w , h = image.size

    w_bar,h_bar = smart_resize(image_path)

    bbox_list_orignal = resize_bbox_to_original(bbox_list, (w, h), (w_bar, h_bar))

    return bbox_list_orignal



def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2)) 

def show_mask(mask ,ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def get_response(url,query):
    response = requests.post(url, json=query)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)



def VLMs_state(image_path, query, planner ,action, objects):

    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    prompt = load_prompt(f"vlm_rlbench/state.txt")

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene." + f"The format of output should be like {prompt}.\n Objects : {objects}\nQuery : {query}\nPlanner : {planner}\nAction : {action}\nPlease just give me the corresponding json, no explanation and no text required"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}]
        )

    resstr = completion.choices[0].message.content.replace("```","").replace("json","")

    state = json.loads(resstr)

    return state

    

