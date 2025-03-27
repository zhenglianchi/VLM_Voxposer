from PIL import Image
import numpy as np
import json_numpy
json_numpy.patch()
from matplotlib import pyplot as plt
import cv2
from VLM_demo import get_world_bboxs_list
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("EdgeSAM/")
from EdgeSAM.edge_sam import sam_model_registry, SamPredictor
from VLM_demo import show_box,show_mask  

sam_checkpoint = "EdgeSAM/weights/edge_sam.pth"
model_type = "edge_sam"

device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

cap = cv2.VideoCapture("video.mp4")
plt.figure(figsize=(10, 10))

image_path = "video.jpeg"

objects = "[浣熊,猫，冰箱]"
print("正在获取目标框")
bbox_entities = get_world_bboxs_list(image_path, objects)
print("获取目标框成功")

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)
    plt.axis('off')
    image = Image.fromarray(np.array(frame))

    bbox = [item["bbox"] for item in bbox_entities]
    print(bbox)
    predictor.set_image(frame)
    input_boxes = torch.tensor(bbox, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        num_multimask_outputs=1,
    )
    print(masks.shape)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.pause(0.01)
    plt.clf()

plt.close()
cap.release()
    