from PIL import Image, ImageDraw
import numpy as np
import requests
import cv2
from VLM_demo import get_obj_bboxs_list,show_mask,show_box
import json_numpy
json_numpy.patch()
from matplotlib import pyplot as plt


def get_response(url,query):
    response = requests.post(url, json=query)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)

def get_masks(image_path, bbox,if_init):
    url_sam2 = "http://10.129.149.177:8006/act"

    image = np.array(Image.open(image_path).convert("RGB"))

    bbox = np.array(bbox[0]['bbox'],dtype=np.float32)

    query_sam2 = {"image": image, "input_box": bbox, "if_init": if_init}

    result = get_response(url_sam2,query_sam2)

    return result


if __name__ == "__main__":
    cap = cv2.VideoCapture("case.mp4")
    if_init = True
    plt.figure(figsize=(10, 10))
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        image_path = "case.jpeg"
        cv2.imwrite(image_path, frame)

        if if_init:
            text = "水杯"
            bbox = get_obj_bboxs_list(image_path,text)
    
            result = get_masks(image_path, bbox, if_init)
            obj_ids = result["obj_ids"]
            masks = result["masks"]
            #print(masks)

            if_init = False

        else:
            result = get_masks(image_path, bbox, if_init)
            obj_ids = result["obj_ids"]
            masks = result["masks"]
            #print(masks)

        plt.clf()
        plt.imshow(frame)
        masks = masks.squeeze(0)

        for mask in masks:
            mask = mask > 0.0
            show_mask(mask,plt.gca())

        bbox_ = bbox[0]["bbox"]
        bbox_ = [bbox_[0][0],bbox_[0][1],bbox_[1][0],bbox_[1][1]]
        show_box(bbox_, plt.gca())

        plt.axis('off')
        plt.draw()
        plt.pause(0.01)
            
    



