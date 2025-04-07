import cv2
import numpy as np
import torch
import sys
from matplotlib import pyplot as plt
from PIL import Image

# EdgeSAM 和 VLM
sys.path.append("EdgeSAM/")
from EdgeSAM.edge_sam import sam_model_registry, SamPredictor
from VLM_demo import get_world_bboxs_list, show_box, show_mask

# ByteTrack
sys.path.append("ByteTrack/")
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.tracking_utils.visualize import plot_tracking

# 初始化 EdgeSAM
sam_checkpoint = "EdgeSAM/weights/edge_sam.pth"
model_type = "edge_sam"
device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 视频和初始框
video_path = "video.mp4"
image_path = "video.jpeg"
objects = "[浣熊,猫，冰箱]"

print("正在获取目标框")
bbox_entities = get_world_bboxs_list(image_path, objects)
print("获取目标框成功")

# 初始检测框（x1,y1,x2,y2）转为 [x, y, w, h, score, class_id]，用于 ByteTrack 初始化
init_dets = []
for item in bbox_entities:
    x1, y1, x2, y2 = item["bbox"]
    w, h = x2 - x1, y2 - y1
    init_dets.append([x1, y1, w, h, 0.99, 0])  # score=0.99, class_id=0
init_dets = np.array(init_dets)

# 初始化 ByteTrack
tracker_args = {
    "track_thresh": 0.5,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "min_box_area": 10,
    "frame_rate": 30,
}
tracker = BYTETracker(tracker_args, frame_rate=30)

# 打开视频
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
frame_id = 0
if not ret:
    print("无法读取视频")
    exit()

# 手动为第一帧更新追踪器（初始化）
tracker.update(init_dets, frame.shape[:2], frame_id)

plt.figure(figsize=(10, 10))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    timer = Timer()
    timer.tic()

    # 模拟检测：在真实使用中应从检测器获取 bbox。这里直接使用已有 track 的框。
    # 我们只使用跟踪功能，因此检测部分可以为空，ByteTrack 会使用上帧信息继续追踪
    online_targets = tracker.update(np.empty((0, 6)), frame.shape[:2], frame_id)

    # 构建 bbox 输入 SAM
    boxes_xyxy = []
    for t in online_targets:
        tlwh = t.tlwh
        x1, y1, x2, y2 = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
        boxes_xyxy.append([x1, y1, x2, y2])

    if len(boxes_xyxy) == 0:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)

    input_boxes = torch.tensor(boxes_xyxy, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        num_multimask_outputs=1,
    )

    # 可视化
    plt.imshow(frame_rgb)
    plt.axis('off')
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
    for box in boxes_xyxy:
        show_box(np.array(box), plt.gca())

    plt.pause(0.01)
    plt.clf()

plt.close()
cap.release()
