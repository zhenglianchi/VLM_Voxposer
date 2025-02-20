from openai import OpenAI
import base64
from PIL import Image, ImageDraw
import json
import math
import numpy as np
import requests
import json_numpy
import matplotlib.pyplot as plt
import dashscope
json_numpy.patch()

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

def get_obj_bboxs_list(image_path,obj):

    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[{"role": "user","content": [
                #{"type": "text","text": "Detect all objects in the image and return their locations in the form of coordinates. The format of output should be like {“bbox”: [x1, y1, x2, y2], “label”: the name of this object in Chinese}"},
                {"type": "text","text": f"This is a robotic arm operation scene. Detect {obj} in the image and return their locations in the form of coordinates. The format of output should be like" 
                + "{“bbox”: [x1, y1, x2, y2], “label”: the name of this object in English}  not {“bbox_2d”: [x1, y1, x2, y2], “label”: the name of this object in Chinese}"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}]
        )

    bbox_list_str = completion.choices[0].message.content[7:-3]

    #print(bbox_list_str)

    if bbox_list_str.strip()[0] != "[":
        bbox_list = [json.loads(bbox_list_str)]
    else:
        bbox_list = json.loads(bbox_list_str)


    # 打开图片
    image = Image.open(image_path)
    w , h = image.size

    w_bar,h_bar = smart_resize(image_path)

    bbox_list_orignal = resize_bbox_to_original(bbox_list, (w, h), (w_bar, h_bar))

    '''draw = ImageDraw.Draw(image)

    # 遍历结果，绘制边界框
    for detection in bbox_list_orignal:
        bbox = detection['bbox']
        label = detection['label']
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((bbox[0], bbox[1]), label, fill="red")

    # 显示图片
    image.show()'''

    return bbox_list_orignal

def get_world_bboxs_list(image_path):

    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[{"role": "user","content": [
                {"type": "text","text": "This is a robotic arm operation scene. Detect all objects in the image and return their locations in the form of coordinates, don't give up any information about the details. The format of output should be like {“bbox”: [x1, y1, x2, y2], “label”: the name of this object in English} not {“bbox_2d”: [x1, y1, x2, y2], “label”: the name of this object in Chinese}"},
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

    draw = ImageDraw.Draw(image)

    # 遍历结果，绘制边界框
    '''for detection in bbox_list_orignal:
        bbox = detection['bbox']
        label = detection['label']
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((bbox[0], bbox[1]), label, fill="red")

    # 显示图片
    image.show()'''

    return bbox_list_orignal

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2)) 

def show_mask(mask, ax, random_color=False):
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

def get_entitites(image_path, bboxes):
    url_sam2 = "http://10.129.38.192:8006/act"
    image = np.array(Image.open(image_path).convert("RGB"))

    ent = bboxes.copy()
    boxes = [item["bbox"] for item in bboxes]

    query_sam2 = {"image": image, "input_box": np.array(boxes)}
    result = get_response(url_sam2,query_sam2)
        
    masks = result["masks"]
    scores = result["scores"]
    for i in range(len(ent)):
        ent[i]["mask"] = masks[i]
        ent[i]["score"] = scores[i]

    '''plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        if len(masks) != 1:
            show_mask(mask.squeeze(0),plt.gca())
        else:
            show_mask(mask,plt.gca())

    for item in ent:
        box = item["bbox"]
        show_box(box, plt.gca())

    plt.axis('off')
    plt.show()'''
    return ent

def get_action(image_path, instruction):

    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)
    prompt = '''
    {
        "objects" : ['blue block', 'yellow block', 'mug'],
        "Query" : "place the blue block on the yellow block, and avoid the mug at all time.",
        "action" : ["grasp the blue block while keeping at least 15cm away from the mug","back to default pose","move to 5cm on top of the yellow block while keeping at least 15cm away from the mug","open gripper"]
    }
    '''

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene. I want {instruction}, Please arrange the robot arm movements according to the given picture." + f"The format of output should be like {prompt}"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}]
        )

    action = json.loads(completion.choices[0].message.content[7:-3])

    return action

def get_state(image_path, instruction):

    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    prompt = '''
    {
        "Query" : "grasp the blue block while keeping at least 15cm away from the mug",
        "affordable" : ["blue block"],
        "avoid" : ["mug","yellow block"]
    }
    '''

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene. I want {instruction}, Please tell me what objects I should approach and catch, and what objects I should avoid. " + f"The format of output should be like {prompt}"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}]
        )

    state = json.loads(completion.choices[0].message.content[7:-3])

    return state



if __name__ == '__main__':
    image_path = "R.jpeg"
    obj = "dog"
    bbox = get_obj_bboxs_list(image_path,obj)
    entities = get_entitites(image_path, bbox)
    print(entities)
    



