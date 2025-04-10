from ultralytics import YOLOE
import numpy as np
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from PIL import Image
import matplotlib.pyplot as plt

model = YOLOE("yoloe-11s-seg.pt")

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

# Handcrafted shape can also be passed, please refer to app.py
# Multiple boxes or handcrafted shapes can also be passed as visual prompt in an image
visuals = dict(
    bboxes=[
        np.array(
            [
                [142, 267, 230, 373],  # For bin
                [335, 338, 367, 362],  # For rubbish
                [280, 331, 307, 358],  # For tomato
                [401, 334, 429, 364],  # For tomato
            ],
        ), 
    ]
    ,
    cls=[
        np.array(
            [
                0,  # For bin
                1,  # For rubbish
                2,  # For tomato
                2,  # For tomato
            ]
        ), 
    ]
)

source_image = Image.open('robot1.jpeg')
target_image = Image.open('robot2.jpeg')
width,height = source_image.size
imgsz = (height,width)

result = model.predict(source_image, prompts=visuals, predictor=YOLOEVPSegPredictor,
              return_vpe=True, save=False, verbose=False, imgsz=imgsz)

model.set_classes(["bin", "rubbish", "tomato"], model.predictor.vpe)
model.predictor = None  # remove VPPredictor
result = model.predict(target_image, save=False, conf=0.5, iou=0.5, verbose=False, imgsz=imgsz)
print(result[0].boxes.data)
print(result[0].masks.data.shape)
print(type(result[0].boxes.data.detach().cpu().numpy()[0][-1]))