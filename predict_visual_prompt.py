from ultralytics import YOLOE
import numpy as np
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from PIL import Image

model = YOLOE("yoloe-11s-seg.pt")

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

result = model.predict(source_image, prompts=visuals, predictor=YOLOEVPSegPredictor,
              return_vpe=True, save=True)

model.set_classes(["bin", "rubbish", "tomato"], model.predictor.vpe)
model.predictor = None  # remove VPPredictor
result = model.predict(target_image, save=True)
print(result[0].boxes.data)
print(result[0].masks.data.shape)
