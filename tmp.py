'''
for cam_name in env.camera_names:
    cam = env.name2cam[cam_name]
    rgb, depth, pcd_ = get_rgb_depth(cam, get_rgb=True, get_depth=True, get_pcd=True)

    image = Image.fromarray(np.array(rgb))
    image_path = "tmp/rgb.jpeg"
    image.save(image_path)

    objects = env.get_object_names()
    output_image_path, entities = get_world_mask_list(image_path,objects)

    state = get_state(output_image_path, instruction, objects)
    print(state)


    for item in entities:
        points, masks, normals = [], [], []
        points.append(pcd_.reshape(-1, 3))
        # 这里得到的是该相机下每个物体的mask、normal
        mask = item['mask']
        label = item['label']
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask =  mask.reshape(h, w).reshape(-1)
        masks.append(mask)

        # estimate normals using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[-1])
        pcd.estimate_normals()
        cam_normals = np.asarray(pcd.normals)
        # use lookat vector to adjust normal vectors
        flip_indices = np.dot(cam_normals, env.lookat_vectors[cam_name]) > 0
        cam_normals[flip_indices] *= -1
        normals.append(cam_normals)

        points = np.array(points)
        masks = np.array(masks)
        normals = np.array(normals)

        obj_points = points[np.isin(masks, 1)]
        if len(obj_points) == 0:
            raise ValueError(f"Scene not any object!")
        obj_normals = normals[np.isin(masks, 1)]
        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)

        state[label]["points"] = obj_points
        state[label]["normals"] = obj_normals


    # 将state保存为JSON文件
    state_json_path = f"tmp/state_{cam_name}.json"
    write_state(state_json_path,state)


# 读取两个相机下的state并进行合并
state1 = read_state(f"tmp/state_front.json")
state2 = read_state(f"tmp/state_wrist.json")
print(state1)
print(state2)
'''