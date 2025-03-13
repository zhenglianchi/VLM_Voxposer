from PIL import Image
import numpy as np
from VLM_demo import show_mask
import json_numpy
json_numpy.patch()
from matplotlib import pyplot as plt



rgb, _, _ = self.get_rgb_depth(self.cam, get_rgb=True, get_depth=False, get_pcd=False)

image = Image.fromarray(np.array(rgb))
image_path = "tmp/rgb.jpeg"
image.save(image_path)

objects = env.get_object_names()
print("正在获取目标框")
bbox = get_world_bboxs_list(image_path, objects)
print("获取目标框成功")

frame = capture_rgb()

bbox_entities = add_points(frame, bbox_entities=bbox ,if_init=True)

state = {}

plt.figure(figsize=(20, 20))
while True:
    start_time = time.time()
    frame = capture_rgb()
    _, _, pcd_ = self.get_rgb_depth(self.cam, get_rgb=False, get_depth=True, get_pcd=True)

    plt.clf()
    plt.imshow(frame)
    try:
        result = track_mask(frame, if_init=False)
        obj_ids = result['obj_ids']
        masks_ = result['masks']
        #print(masks_.shape)  #[4,1,768,1024]
        for item in bbox_entities:
            points, masks, normals = [], [], []
            points.append(pcd_.reshape(-1, 3))
            id = item['id']
            #print(id)
            mask = masks_[id]
            label = item['label']
            h, w = mask.shape[-2:]
            #print(h,w)
            mask = (mask>0.0).astype(np.uint8)

            show_mask(mask,plt.gca())

            mask =  mask.reshape(h, w).reshape(-1)
            masks.append(mask)

            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, env.lookat_vectors[self.cam_name]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)

            points = np.array(points)
            masks = np.array(masks)
            normals = np.array(normals)

            obj_points = points[np.isin(masks, 1)]
            if len(obj_points) == 0:
                print(f"Scene not object {label}!")
                continue
            obj_normals = normals[np.isin(masks, 1)]
            # voxel downsample using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            pcd.normals = o3d.utility.Vector3dVector(obj_normals)
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
            obj_points = np.asarray(pcd_downsampled.points)
            obj_normals = np.asarray(pcd_downsampled.normals)

            state[label]= self.get_obs(obj_points, obj_normals, label)

        state['gripper'] = self.get_ee_obs()
        state['workspace'] = self.get_table_obs()

        # 将state保存为JSON文件
        state_json_path = f"tmp/state_{self.cam_name}.json"
        write_state(state_json_path, state,lock)
        end_time = time.time()  # 记录结束时间
        elapsed_time_ms = (end_time - start_time) * 1000  # 计算并转换为毫秒
        print("update state success!"+f"Consumed time: {elapsed_time_ms:.2f} ms")
        plt.axis('off')
        plt.draw()
        plt.savefig(f"tmp/state_{self.cam_name}.png", bbox_inches='tight', pad_inches=0)
        #plt.pause(0.01)