from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks
from world_state import get_world_mask_list, write_state, read_state,get_multi_image_world_mask_list
from PIL import Image
from VLM_demo import get_state
import open3d as o3d
from utils import Observation, normalize_vector
import os

def get_rgb_depth(sensor=None, get_rgb=True, get_depth=True,get_pcd=True):
    rgb = depth = pcd = None
    if sensor is not None and (get_rgb or get_depth):
        sensor.handle_explicitly()
        if get_rgb:
            rgb = sensor.capture_rgb()
            rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
        if get_depth or get_pcd:
            depth = sensor.capture_depth(False)
        if get_pcd:
            depth_m = depth
            near = sensor.get_near_clipping_plane()
            far = sensor.get_far_clipping_plane()
            depth_m = near + depth * (far - near)
            pcd = sensor.pointcloud_from_depth(depth_m)
            if not get_depth:
                depth = None
    return rgb, depth, pcd

#load config
config = get_config('rlbench')

#Initializes Vox map visualizer
visualizer = ValueMapVisualizer(config['visualizer'])

#Initializes the VoxPoserRLBench environment.
#launch rlbench environment
env = VoxPoserRLBench(visualizer=visualizer)

#Initializes LMPs
lmps, lmp_env = setup_LMP(env, config, debug=False)

#high level plan
voxposer_ui = lmps['plan_ui']

# below are the tasks that have object names added to the "task_object_names.json" file
# uncomment one to use
#env.load_task(tasks.PutRubbishInBin)
env.load_task(tasks.LampOff)
#env.load_task(tasks.OpenWineBottle)
#env.load_task(tasks.PushButton)
#env.load_task(tasks.TakeOffWeighingScales)
#env.load_task(tasks.MeatOffGrill)
#env.load_task(tasks.SlideBlockToTarget)
#env.load_task(tasks.TakeLidOffSaucepan)
#env.load_task(tasks.TakeUmbrellaOutOfUmbrellaStand)
descriptions, obs = env.reset()

# set the object names to be prompt text
set_lmp_objects(lmps, env.get_object_names()) 

# set the random instruction prompt text
instruction = np.random.choice(descriptions)

# run the high level plan with the LMP 
#voxposer_ui(instruction)

def get_obs(lmp_env, obj_pc,obj_normal, label):
    obs_dict = dict()
    voxel_map = lmp_env._points_to_voxel_map(obj_pc)
    aabb_min = lmp_env._world_to_voxel(np.min(obj_pc, axis=0))
    aabb_max = lmp_env._world_to_voxel(np.max(obj_pc, axis=0))
    obs_dict['occupancy_map'] = voxel_map  # in voxel frame
    obs_dict['name'] = label
    obs_dict['position'] = lmp_env._world_to_voxel(np.mean(obj_pc, axis=0))  # in voxel frame
    obs_dict['aabb'] = np.array([aabb_min, aabb_max])  # in voxel frame
    obs_dict['_position_world'] = np.mean(obj_pc, axis=0)  # in world frame
    obs_dict['_point_cloud_world'] = obj_pc  # in world frame
    obs_dict['normal'] = normalize_vector(obj_normal.mean(axis=0))

    object_obs = Observation(obs_dict)
    return object_obs

def get_ee_obs(lmp_env):
    # 获取末端姿态
    obs_dict = dict()
    obs_dict['name'] = "gripper"
    obs_dict['position'] = lmp_env.get_ee_pos()
    obs_dict['aabb'] = np.array([lmp_env.get_ee_pos(), lmp_env.get_ee_pos()])
    obs_dict['_position_world'] = lmp_env._env.get_ee_pos()
    object_obs = Observation(obs_dict)
    return object_obs

def get_table_obs(lmp_env):
    offset_percentage = 0.1
    x_min = lmp_env._env.workspace_bounds_min[0] + offset_percentage * (lmp_env._env.workspace_bounds_max[0] - lmp_env._env.workspace_bounds_min[0])
    x_max = lmp_env._env.workspace_bounds_max[0] - offset_percentage * (lmp_env._env.workspace_bounds_max[0] - lmp_env._env.workspace_bounds_min[0])
    y_min = lmp_env._env.workspace_bounds_min[1] + offset_percentage * (lmp_env._env.workspace_bounds_max[1] - lmp_env._env.workspace_bounds_min[1])
    y_max = lmp_env._env.workspace_bounds_max[1] - offset_percentage * (lmp_env._env.workspace_bounds_max[1] - lmp_env._env.workspace_bounds_min[1])
    table_max_world = np.array([x_max, y_max, 0])
    table_min_world = np.array([x_min, y_min, 0])
    table_center = (table_max_world + table_min_world) / 2
    obs_dict = dict()
    obs_dict['name'] = "workspace"
    obs_dict['position'] = lmp_env._world_to_voxel(table_center)
    obs_dict['_position_world'] = table_center
    obs_dict['normal'] = np.array([0, 0, 1])
    obs_dict['aabb'] = np.array([lmp_env._world_to_voxel(table_min_world), lmp_env._world_to_voxel(table_max_world)])

    object_obs = Observation(obs_dict)
    return object_obs

def update_state():
    cam_wrist = env.name2cam["wrist"]
    cam_front = env.name2cam["front"]

    rgb_wrist, depth_wrist, pcd_wrist = get_rgb_depth(cam_wrist, get_rgb=True, get_depth=True, get_pcd=True)
    rgb_front, depth_front, pcd_front = get_rgb_depth(cam_front, get_rgb=True, get_depth=True, get_pcd=True)

    image_wrist = Image.fromarray(np.array(rgb_wrist))
    image_path_wrist = "tmp/rgb_wrist.jpeg"
    image_wrist.save(image_path_wrist)
    image_front = Image.fromarray(np.array(rgb_front))
    image_path_front = "tmp/rgb_front.jpeg"
    image_front.save(image_path_front)

    objects = env.get_object_names()

    #objects = ["bin","rubbish", "tomato1", "tomato2"]
    #instruction = "Put the rubbish in the bin"

    output_image_path1, output_image_path2, entities1, entities2 = get_multi_image_world_mask_list(image_path_wrist,image_path_front,objects)

    state1 = get_state(output_image_path1, instruction, objects)
    print(state1)
    state2 = get_state(output_image_path2, instruction, objects)
    print(state2)

    '''for item in entities:
        points, masks, normals = [], [], []
        points.append(pcd_.reshape(-1, 3))
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
        flip_indices = np.dot(cam_normals, env.lookat_vectors["wrist"]) > 0
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

        state[label]["obs"] = get_obs(lmp_env, obj_points, obj_normals, label)

    state['gripper'] = get_ee_obs(lmp_env)
    state['workspace'] = get_table_obs(lmp_env)

    # 将state保存为JSON文件
    state_json_path = f"tmp/state_wrist.json"
    write_state(state_json_path,state)'''


update_state()
#state = read_state(f"tmp/state_wrist.json")

