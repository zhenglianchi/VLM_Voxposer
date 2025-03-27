from LMP import LMP
from utils import get_clock_time, normalize_vector, pointat2quat, bcolors, Observation, VoxelIndexingWrapper
import numpy as np
from scipy.spatial.transform import Rotation as R
from planners import PathPlanner
import time
from scipy.ndimage import distance_transform_edt
import transforms3d
from controllers import Controller
import open3d as o3d
from VLM_demo import  write_state, get_world_bboxs_list,show_mask,use_sam
from PIL import Image
from transforms3d.quaternions import axangle2quat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# creating some aliases for end effector and table in case LLMs refer to them differently (but rarely this happens)
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']
TABLE_ALIAS = ['table', 'desk', 'workstation', 'work_station', 'work station', 'workspace', 'work_space', 'work space']

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
    print('#' * 50)
    print(f'## voxel resolution: {self._resolution}')
    print('#' * 50)
    print()
    print()
  
  # ======================================================
  # == functions exposed to LLMstate
  # ======================================================

  def get_rgb_depth(self, sensor=None, get_rgb=True, get_depth=True,get_pcd=True):
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
      image_path = "tmp/rgb.jpeg"
      image.save(image_path)

      objects = self._env.get_object_names()
      print("正在获取目标框")
      bbox = get_world_bboxs_list(image_path, objects)
      print("获取目标框成功")
      return bbox
  
  def _capture_rgb(self):
    self.cam.handle_explicitly()
    rgb = self.cam.capture_rgb()
    frame = Image.fromarray(np.clip((rgb * 255.).astype(np.uint8), 0, 255)).convert("RGB")
    return frame

  def update_mask_entities(self, bbox_entities,lock):
      #frame = self._capture_rgb()
      state = {}
      plt.figure(figsize=(20, 20))
      while True:
        start_time = time.time()
        frame = self._capture_rgb()
        _, _, pcd_ = self.get_rgb_depth(self.cam, get_rgb=False, get_depth=True, get_pcd=True)

        plt.clf()
        plt.imshow(frame)
        try:
          bbox = [item["bbox"] for item in bbox_entities]
          masks_ = use_sam(frame, bbox)
          print(masks_.shape)
          for (index,item) in enumerate(bbox_entities):
              points, masks, normals = [], [], []
              points.append(pcd_.reshape(-1, 3))
              mask = masks_[index]

              label = item['label']
              h, w = mask.shape[-2:]

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
              flip_indices = np.dot(cam_normals, self._env.lookat_vectors[self.cam_name]) > 0
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
          plt.savefig(f"tmp/state_{self.cam_name}.jpeg", bbox_inches='tight', pad_inches=0)
          #plt.pause(0.01)

        except Exception as e:
          print(f"ERROR: Failed to get state, try again - {e}")
  
  def vec2quat(self, vec):
    """
    将向量转换为四元数。
    
    参数:
    vector (np.array): 一个三维向量，表示方向。
    
    返回:
    np.array: 表示旋转的四元数。
    """
    vec = vec / np.linalg.norm(vec)

    # 目标方向是z轴
    target = np.array([0, 0, 1])

    # 计算旋转
    # 使用Rotation.from_rotvec来获取从z轴到v的旋转
    rotation = R.align_vectors([vec], [target])[0]

    # 获取四元数
    quat = rotation.as_quat()  # 返回四元数 [x, y, z, w] 格式
    return quat

  def get_ee_pos(self):
    return self._world_to_voxel(self._env.get_ee_pos())
  
  def execute(self, movable_obs, affordance_map=None, avoidance_map=None, rotation_map=None,
              velocity_map=None, gripper_map=None):
    """
    First use planner to generate waypoint path, then use controller to follow the waypoints.

    Args:
      movable_obs: a dictionary of movable objects
      affordance_map: callable function that generates a 3D numpy array, the target voxel map
      avoidance_map: callable function that generates a 3D numpy array, the obstacle voxel map
      rotation_map: callable function that generates a 4D numpy array, the rotation voxel map (rotation is represented by a quaternion *in world frame*)
      velocity_map: callable function that generates a 3D numpy array, the velocity voxel map
      gripper_map: callable function that generates a 3D numpy array, the gripper voxel map
    """
    # initialize default voxel maps if not specified
    if rotation_map is None:
      rotation_map = self._get_default_voxel_map('rotation')()
    if velocity_map is None:
      velocity_map = self._get_default_voxel_map('velocity')()
    if gripper_map is None:
      gripper_map = self._get_default_voxel_map('gripper')()
    if avoidance_map is None:
      avoidance_map = self._get_default_voxel_map('obstacle')()

    # 如果需要移动的是末端执行器则 object_centric=False，否则为True
    object_centric = (not movable_obs['name'] in EE_ALIAS)
    stop = True
    if affordance_map is not None:
      # evaluate voxel maps such that we use latest information
      _affordance_map = affordance_map
      _avoidance_map = avoidance_map
      _rotation_map = rotation_map
      _velocity_map = velocity_map
      _gripper_map = gripper_map
      # preprocess avoidance map
      _avoidance_map = self._preprocess_avoidance_map(_avoidance_map, _affordance_map, movable_obs)
      # start planning
      start_pos = movable_obs['position']
      if movable_obs['name'] == "gripper":
        start_pos = self.get_ee_pos()

      next_pos, stop = self._planner.optimize(start_pos, _affordance_map, _avoidance_map, object_centric=object_centric)
      path_voxel = [next_pos]
      # convert voxel path to world trajectory, and include rotation, velocity, and gripper information
      traj_world = self._path2traj(path_voxel, _rotation_map, _velocity_map, _gripper_map)

      # execute path
      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] start executing path via controller ({len(traj_world)} waypoints){bcolors.ENDC}')

      for i, waypoint in enumerate(traj_world):
        # check if the movement is finished
        if np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0]) <= 0.01:
          print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached last waypoint; curr_xyz={movable_obs['_position_world']}, target={traj_world[-1][0]} (distance: {np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0]):.3f})){bcolors.ENDC}")
          break

        # waypoint : world_xyz, rotation, velocity, gripper
        controller_info = self._controller.execute(movable_obs, waypoint)

        dist2target = np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0])
        if not object_centric and controller_info['mp_info'] == -1:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] failed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}); mp info: {controller_info["mp_info"]}{bcolors.ENDC}')
        else:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] completed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}){bcolors.ENDC}')
        
        controller_info['controller_step'] = i
        controller_info['target_waypoint'] = waypoint


    # make sure we are at the final target position and satisfy any additional parametrization
    # (skip if we are specifying object-centric motion)
    if not object_centric:
      try:
        # traj_world: world_xyz, rotation, velocity, gripper
        ee_pos_world = traj_world[-1][0]
        ee_rot_world = traj_world[-1][1]
        ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
        ee_speed = traj_world[-1][2]
        gripper_state = traj_world[-1][3]
      except:
        # evaluate latest voxel map
        _rotation_map = rotation_map
        _velocity_map = velocity_map
        _gripper_map = gripper_map
        # get last ee pose
        ee_pos_world = self._env.get_ee_pos()
        ee_pos_voxel = self.get_ee_pos()
        ee_rot_world = _rotation_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
        ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
        ee_speed = _velocity_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
        gripper_state = _gripper_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
      # move to the final target
      self._env.apply_action(np.concatenate([ee_pose_world, [gripper_state]]))

    return stop
  
  def cm2index(self, cm, direction):
    if isinstance(direction, str) and direction == 'x':
      x_resolution = self._resolution[0] * 100  # resolution is in m, we need cm
      return int(cm / x_resolution)
    elif isinstance(direction, str) and direction == 'y':
      y_resolution = self._resolution[1] * 100
      return int(cm / y_resolution)
    elif isinstance(direction, str) and direction == 'z':
      z_resolution = self._resolution[2] * 100
      return int(cm / z_resolution)
    else:
      # calculate index along the direction
      assert isinstance(direction, np.ndarray) and direction.shape == (3,)
      direction = normalize_vector(direction)
      x_cm = cm * direction[0]
      y_cm = cm * direction[1]
      z_cm = cm * direction[2]
      x_index = self.cm2index(x_cm, 'x')
      y_index = self.cm2index(y_cm, 'y')
      z_index = self.cm2index(z_cm, 'z')
      return np.array([x_index, y_index, z_index])
  
  def index2cm(self, index, direction=None):
    if direction is None:
      average_resolution = np.mean(self._resolution)
      return index * average_resolution * 100  # resolution is in m, we need cm
    elif direction == 'x':
      x_resolution = self._resolution[0] * 100
      return index * x_resolution
    elif direction == 'y':
      y_resolution = self._resolution[1] * 100
      return index * y_resolution
    elif direction == 'z':
      z_resolution = self._resolution[2] * 100
      return index * z_resolution
    else:
      raise NotImplementedError
    
  def pointat2quat(self, vector):
    assert isinstance(vector, np.ndarray) and vector.shape == (3,), f'vector: {vector}'
    return pointat2quat(vector)    # append the last waypoint a few more times for the robot to stabilize
    for _ in range(1):
      traj.append((world_xyz, rotation, velocity, gripper))

  def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
    """given a 3D np array, set the value of the voxel at voxel_xyz to value. If radius is specified, set the value of all voxels within the radius to value."""
    voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
    if radius_cm > 0:
      radius_x = self.cm2index(radius_cm, 'x')
      radius_y = self.cm2index(radius_cm, 'y')
      radius_z = self.cm2index(radius_cm, 'z')
      # simplified version - use rectangle instead of circle (because it is faster)
      min_x = max(0, voxel_xyz[0] - radius_x)
      max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
      min_y = max(0, voxel_xyz[1] - radius_y)
      max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
      min_z = max(0, voxel_xyz[2] - radius_z)
      max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
      voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
    return voxel_map
  
  def get_empty_affordance_map(self):
    return self._get_default_voxel_map('target')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

  def get_empty_avoidance_map(self):
    return self._get_default_voxel_map('obstacle')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_rotation_map(self):
    return self._get_default_voxel_map('rotation')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_velocity_map(self):
    return self._get_default_voxel_map('velocity')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_gripper_map(self):
    return self._get_default_voxel_map('gripper')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
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
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size)) #* self._env.get_last_gripper_action()
      elif type == 'rotation':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size, 4))
        voxel_map[:, :, :] = self._env.get_ee_quat()
      else:
        raise ValueError('Unknown voxel map type: {}'.format(type))
      voxel_map = VoxelIndexingWrapper(voxel_map)
      return voxel_map
    return fn_wrapper
  
  def _path2traj(self, path, rotation_map, velocity_map, gripper_map):
    """
    convert path (generated by planner) to trajectory (used by controller)
    path only contains a sequence of voxel coordinates, while trajectory parametrize the motion of the end-effector with rotation, velocity, and gripper on/off command
    """
    # convert path to trajectory
    traj = []
    for i in range(len(path)):
      # get the current voxel position
      voxel_xyz = path[i]
      # get the current world position
      world_xyz = self._voxel_to_world(voxel_xyz)
      voxel_xyz = np.round(voxel_xyz).astype(int)
      # get the current rotation (in world frame)
      rotation = rotation_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current velocity
      velocity = velocity_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current on/off
      gripper = gripper_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # LLM might specify a gripper value change, but sometimes EE may not be able to reach the exact voxel, so we overwrite the gripper value if it's close enough (TODO: better way to do this?)
      if (i == len(path) - 1) and not (np.all(gripper_map == 1) or np.all(gripper_map == 0)):
        # get indices of the less common values
        less_common_value = 1 if np.sum(gripper_map == 1) < np.sum(gripper_map == 0) else 0
        less_common_indices = np.where(gripper_map == less_common_value)
        less_common_indices = np.array(less_common_indices).T
        # get closest distance from voxel_xyz to any of the indices that have less common value
        closest_distance = np.min(np.linalg.norm(less_common_indices - voxel_xyz[None, :], axis=0))
        # if the closest distance is less than threshold, then set gripper to less common value
        if closest_distance <= 3:
          gripper = less_common_value
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] overwriting gripper to less common value for the last waypoint{bcolors.ENDC}')
      # add to trajectory
      traj.append((world_xyz, rotation, velocity, gripper))

    return traj
  
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
  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np,
      'euler2quat': transforms3d.euler.euler2quat,
      'quat2euler': transforms3d.euler.quat2euler,
      'qinverse': transforms3d.quaternions.qinverse,
      'qmult': transforms3d.quaternions.qmult,
  }  # external library APIs
  variable_vars = {
      k: getattr(lmp_env, k)
      for k in dir(lmp_env) if callable(getattr(lmp_env, k)) and not k.startswith("_")
  }  # our custom APIs exposed to LMPs

  # creating the LMP that deals w/ high-level language commands
  task_planner = LMP(
      'planner', lmps_config, fixed_vars, variable_vars, debug, env_name
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