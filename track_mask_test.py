from PIL import Image
import numpy as np
from VLM_demo import show_mask,add_points,track_mask
import json_numpy
json_numpy.patch()
from matplotlib import pyplot as plt
import cv2
from world_state import get_world_bboxs_list
import time


cap = cv2.VideoCapture("test.mp4")
ret, frame = cap.read()

image = Image.fromarray(np.array(frame))
image_path = "test.jpeg"
image.save(image_path)

objects = "[tiger1,tiger2]"
print("正在获取目标框")
bbox = get_world_bboxs_list(image_path, objects)
print("获取目标框成功")

bbox_entities = add_points(frame, bbox_entities=bbox ,if_init=True)

plt.figure(figsize=(20, 20))
while True:
    start_time = time.time()
    ret, frame = cap.read()

    plt.clf()
    plt.imshow(frame)

    result = track_mask(frame, if_init=False)
    obj_ids = result['obj_ids']
    masks_ = result['masks']
    print(masks_.shape)  #[4,1,768,1024]

    for item in bbox_entities:
        id = item['id']
        #print(id)
        mask = masks_[id]
        label = item['label']
        h, w = mask.shape[-2:]
        #print(h,w)
        mask = (mask>0.0).astype(np.uint8)

        show_mask(mask,plt.gca())

    end_time = time.time()  # 记录结束时间
    elapsed_time_ms = (end_time - start_time) * 1000  # 计算并转换为毫秒
    print("update state success!"+f"Consumed time: {elapsed_time_ms:.2f} ms")
    plt.axis('off')
    plt.draw()
    plt.pause(0.1)

    