
import numpy as np
from PIL import Image
from VLM_demo import get_entitites,get_world_bboxs_list,get_action,get_state,get_multi_image_world_bboxs_list
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def get_world_mask_list(image_path,instruction):

    bbox = get_world_bboxs_list(image_path, instruction)
    entities = get_entitites(image_path, bbox)

    original_image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    for item in entities:
        mask = item['mask']
        label = item['label']
        bbox = item['bbox']

        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2


        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        plt.text(x_center, y_center, label, color="black", fontsize=16, ha='center', va='center')

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)

    plt.axis('off')
    # 保存图像到本地
    output_image_path = "tmp/mask_image.png"
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

    #plt.show()

    return output_image_path, entities

def write_state(output_json_path,state):
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(state, json_file)

def read_state(state_json_path):
    with open(state_json_path, 'r', encoding='utf-8') as json_file:
        loaded_state = json.load(json_file)
    return loaded_state

def get_multi_image_world_mask_list(image_path1,image_path2,instruction):

    bbox = get_multi_image_world_bboxs_list(image_path1, image_path2, instruction)
    print(bbox)
    
    entities1 = get_entitites(image_path1, bbox[0])
    entities2 = get_entitites(image_path2, bbox[1])

    original_image = Image.open(image_path1)
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    for item in entities1:
        mask = item['mask']
        label = item['label']
        bbox = item['bbox']

        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2


        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        plt.text(x_center, y_center, label, color="black", fontsize=16, ha='center', va='center')

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)

    plt.axis('off')
    # 保存图像到本地
    output_image_path1 = "tmp/mask_image1.png"
    plt.savefig(output_image_path1, bbox_inches='tight', pad_inches=0)
    plt.show()


    original_image = Image.open(image_path2)
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    for item in entities2:
        mask = item['mask']
        label = item['label']
        bbox = item['bbox']

        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2


        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        plt.text(x_center, y_center, label, color="black", fontsize=16, ha='center', va='center')

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)

    plt.axis('off')
    # 保存图像到本地
    output_image_path2 = "tmp/mask_image2.png"
    plt.savefig(output_image_path2, bbox_inches='tight', pad_inches=0)

    plt.show()

    return output_image_path1, output_image_path2, entities1, entities2

