from LMP import LMP
from utils import get_clock_time, normalize_vector, bcolors, Observation, VoxelIndexingWrapper
import numpy as np
from planners import PathPlanner
import time
from scipy.ndimage import distance_transform_edt
from controllers import Controller
import open3d as o3d
from VLM_demo import  write_state, get_world_bboxs_list,show_mask,process_visual_prompt,set_visual_prompt,predict_mask,encode_image,resize_bbox_to_original,smart_resize,get_response
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import json_numpy
import os
from scipy.spatial.transform import Rotation as R
from grasp_module import infer_grasps
matplotlib.use('Agg')

json_numpy.patch()

class LMP_interface():

  def __init__(self, env, lmp_config, controller_config, planner_config, env_name='rlbench'):
    self._env = env
    self._env_name = env_name
    self._cfg = lmp_config
    self._map_size = self._cfg['map_size']
    self._planner = PathPlanner(planner_config, map_size=self._map_size)
    self._controller = Controller(self._env, controller_config)
    self.cam_name = "front"
    self.cam = self._env.name2cam['front']

    # calculate size of each voxel (resolution)
    self._resolution = (self._env.workspace_bounds_max - self._env.workspace_bounds_min) / self._map_size
    print(f'Voxel resolution: {self._resolution}')
  # ======================================================
  # == functions exposed to LLMstate
  # ======================================================

  def get_rgb_depth(self, sensor=None, get_rgb=True, get_depth=True,get_pcd=True):
    rgb = depth = pcd = None
    if sensor is not None and (get_rgb or get_depth):
        sensor.handle_explicitly()
        if get_rgb:
            rgb = sensor.capture_rgb()
            rgb = Image.fromarray(np.clip((rgb * 255.).astype(np.uint8), 0, 255)).convert("RGB")
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


  def get_obs(self, obj_pc,obj_normal, label):
    obs_dict = dict()
    voxel_map = self._points_to_voxel_map(obj_pc)
    aabb_min = self._world_to_voxel(np.min(obj_pc, axis=0))
    aabb_max = self._world_to_voxel(np.max(obj_pc, axis=0))
    obs_dict['occupancy_map'] = voxel_map  # in voxel frame
    obs_dict['name'] = label
    obs_dict['position'] = self._world_to_voxel(np.mean(obj_pc, axis=0))  # in voxel frame
    obs_dict['aabb'] = np.array([aabb_min, aabb_max])  # in voxel frame
    obs_dict['_position_world'] = np.mean(obj_pc, axis=0)  # in world frame
    obs_dict['_point_cloud_world'] = obj_pc  # in world frame
    obs_dict['normal'] = normalize_vector(obj_normal.mean(axis=0))

    object_obs = {"obs":Observation(obs_dict)}
    return object_obs

  def get_ee_obs(self):
      # 获取末端姿态
      obs_dict = dict()
      obs_dict['name'] = "gripper"
      obs_dict['position'] = self.get_ee_pos()
      obs_dict['aabb'] = np.array([self.get_ee_pos(), self.get_ee_pos()])
      obs_dict['_position_world'] = self._env.get_ee_pos()
      object_obs = {"obs":Observation(obs_dict)}
      return object_obs

  def get_table_obs(self):
      offset_percentage = 0.1
      x_min = self._env.workspace_bounds_min[0] + offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      x_max = self._env.workspace_bounds_max[0] - offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      y_min = self._env.workspace_bounds_min[1] + offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      y_max = self._env.workspace_bounds_max[1] - offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      table_max_world = np.array([x_max, y_max, 0])
      table_min_world = np.array([x_min, y_min, 0])
      table_center = (table_max_world + table_min_world) / 2
      obs_dict = dict()
      obs_dict['name'] = "workspace"
      obs_dict['position'] = self._world_to_voxel(table_center)
      obs_dict['_position_world'] = table_center
      obs_dict['normal'] = np.array([0, 0, 1])
      obs_dict['aabb'] = np.array([self._world_to_voxel(table_min_world), self._world_to_voxel(table_max_world)])

      object_obs = {"obs":Observation(obs_dict)}
      return object_obs
  
  
  def update_box(self):
    rgb, _, _ = self.get_rgb_depth(self.cam, get_rgb=True, get_depth=False, get_pcd=False)
    image = Image.fromarray(np.array(rgb))
    image_path = f"tmp/images/rgb.jpeg"
    image.save(image_path)
    objects = self._env.get_object_names()
    bbox = get_world_bboxs_list(image_path, objects)
    return rgb, bbox


  def update_mask_entities(self,lock,q):
      if not os.path.exists("tmp/images"):
          os.makedirs("tmp/images")
      if not os.path.exists("tmp/masks"):
          os.makedirs("tmp/masks")
      state_json_path = f"tmp/state_{self.cam_name}.json"
      state = {}
      plt.figure(figsize=(20, 20))
      frame, bbox_entities = self.update_box()
      visuals,classes,label2id,id2label = process_visual_prompt(bbox_entities)
      set_visual_prompt(frame, visuals, classes)
      num = 0
      while q.empty():
        start_time = time.time()
        label_index = {}
        for item in classes:
          if item not in label_index.keys():
            label_index[item] = 1
        frame, depth, pcd_ = self.get_rgb_depth(self.cam, get_rgb=True, get_depth=True, get_pcd=True)
        plt.clf()
        plt.imshow(frame)
        boxes, masks = predict_mask(frame)
        for (box_ent, mask) in zip(boxes, masks):
            color = np.array(frame.copy(), dtype=np.float32) / 255.0
            depth = np.array(depth)
            workspace_mask = mask.astype(bool)
            intrinsic = self.cam.get_intrinsic_matrix()
            factor_depth = 0.1

            infer_grasps(color, depth, workspace_mask, intrinsic, factor_depth)

            points, masks, normals = [], [], []
            box = box_ent[:4]
            conf = box_ent[4]
            id = int(box_ent[5])
            label = id2label[id]
            points.append(pcd_.reshape(-1, 3))
            h, w = mask.shape[-2:]
            show_mask(mask,plt.gca())
            mask =  mask.reshape(h, w).reshape(-1)

            masks.append(mask)
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, self._env.lookat_vectors[self.cam_name]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
            points,masks,normals = np.array(points),np.array(masks),np.array(normals)
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

            obs = self.get_obs(obj_points, obj_normals, label)
            # 如果物体已经存在，则将新的相同的物体设定为object1，object2，以此类推
            if label in state.keys():
              old_label = label
              label = label + str(label_index[label])
              obs["label"] = label
              state[label] = obs
              label_index[old_label] += 1
            else:
               state[label] = obs

            x_min, y_min, x_max, y_max = box
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # 在中心位置显示label
            plt.text(center_x, center_y, label, color='white', ha='center', va='center', fontsize=12, weight='bold')

        state['gripper'] = self.get_ee_obs()
        state['workspace'] = self.get_table_obs()
        # 将state保存为JSON文件
        #print(state)
        write_state(state_json_path, state, lock)
        end_time = time.time()  # 记录结束时间
        print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] updated object state in {end_time - start_time:.3f}s{bcolors.ENDC}")
        plt.axis('off')
        plt.draw()
        plt.savefig(f"tmp/masks/mask_{num}.jpeg", bbox_inches='tight', pad_inches=0)
        num+=1
        #plt.pause(0.01)
  

  def get_ee_pos(self):
    return self._world_to_voxel(self._env.get_ee_pos())

  
  def reset_to_default_pose(self):
     self._env.reset_to_default_pose()
  
  # ======================================================
  # == helper functions
  # ======================================================
  def _world_to_voxel(self, world_xyz):
    _world_xyz = world_xyz.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    voxel_xyz = pc2voxel(_world_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return voxel_xyz

  def _voxel_to_world(self, voxel_xyz):
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    world_xyz = voxel2pc(voxel_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return world_xyz

  def _points_to_voxel_map(self, points):
    """convert points in world frame to voxel frame, voxelize, and return the voxelized points"""
    _points = points.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    return pc2voxel_map(_points, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)

  def _get_voxel_center(self, voxel_map):
    """calculte the center of the voxel map where value is 1"""
    voxel_center = np.array(np.where(voxel_map == 1)).mean(axis=1)
    return voxel_center

  def _get_scene_collision_voxel_map(self):
    collision_points_world, _ = self._env.get_scene_3d_obs(ignore_robot=True)
    collision_voxel = self._points_to_voxel_map(collision_points_world)
    return collision_voxel

  def _get_default_voxel_map(self, type='target'):
    """returns default voxel map (defaults to current state)"""
    def fn_wrapper():
      if type == 'target':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'obstacle':  # for LLM to do customization
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'velocity':
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size))
      elif type == 'gripper':
        # 这里gripper:1->0为张开;0->1为闭合
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size)) * self._env.get_last_gripper_action()
      elif type == 'rotation':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size, 4))
        voxel_map[:, :, :] = self._env.get_ee_quat()
      else:
        raise ValueError('Unknown voxel map type: {}'.format(type))
      voxel_map = VoxelIndexingWrapper(voxel_map)
      return voxel_map
    return fn_wrapper

  
  def _preprocess_avoidance_map(self, avoidance_map, affordance_map, movable_obs):
    # collision avoidance
    scene_collision_map = self._get_scene_collision_voxel_map()
    # anywhere within 15/100 indices of the target is ignored (to guarantee that we can reach the target)
    ignore_mask = distance_transform_edt(1 - affordance_map)
    scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    # anywhere within 15/100 indices of the start is ignored
    try:
      ignore_mask = distance_transform_edt(1 - movable_obs['occupancy_map'])
      scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    except KeyError:
      start_pos = movable_obs['position']
      ignore_mask = np.ones_like(avoidance_map)
      ignore_mask[start_pos[0] - int(0.1 * self._map_size):start_pos[0] + int(0.1 * self._map_size),
                  start_pos[1] - int(0.1 * self._map_size):start_pos[1] + int(0.1 * self._map_size),
                  start_pos[2] - int(0.1 * self._map_size):start_pos[2] + int(0.1 * self._map_size)] = 0
      scene_collision_map *= ignore_mask
    avoidance_map += scene_collision_map
    avoidance_map = np.clip(avoidance_map, 0, 1)
    return avoidance_map

def setup_LMP(env, general_config, debug=False):
  controller_config = general_config['controller']
  planner_config = general_config['planner']
  lmp_env_config = general_config['lmp_config']['env']

  #修改lmps_config
  lmps_config = general_config['lmp_config']['lmps']
  env_name = general_config['env_name']
  # LMP env wrapper
  lmp_env = LMP_interface(env, lmp_env_config, controller_config, planner_config, env_name=env_name)

  # creating the LMP that deals w/ high-level language commands
  task_planner = LMP(
      'planner', lmps_config, debug, env_name
  )

  lmps = {
      'plan_ui': task_planner
  }

  return lmps, lmp_env


# ======================================================
# jit-ready functions (for faster replanning time, need to install numba and add "@njit")
# ======================================================
def pc2voxel(pc, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """voxelize a point cloud"""
  pc = pc.astype(np.float32)
  # make sure the point is within the voxel bounds
  pc = np.clip(pc, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxels = (pc - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxels)
  voxels = np.round(voxels, 0, _out).astype(np.int32)
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  return voxels

def voxel2pc(voxels, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """de-voxelize a voxel"""
  # check voxel coordinates are non-negative
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  voxels = voxels.astype(np.float32)
  # de-voxelize
  pc = voxels / (map_size - 1) * (voxel_bounds_robot_max - voxel_bounds_robot_min) + voxel_bounds_robot_min
  return pc

def pc2voxel_map(points, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """given point cloud, create a fixed size voxel map, and fill in the voxels"""
  points = points.astype(np.float32)
  voxel_bounds_robot_min = voxel_bounds_robot_min.astype(np.float32)
  voxel_bounds_robot_max = voxel_bounds_robot_max.astype(np.float32)
  # make sure the point is within the voxel bounds
  points = np.clip(points, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxel_xyz = (points - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxel_xyz)
  points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32)
  voxel_map = np.zeros((map_size, map_size, map_size))
  for i in range(points_vox.shape[0]):
      voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
  return voxel_map