from openai import OpenAI
from utils import load_prompt,normalize_vector
from VLM_demo import encode_image,read_state
import json
import os
import numpy as np
from transforms3d.euler import euler2quat,quat2euler
from transforms3d.quaternions import qinverse,qmult
from scipy.spatial.transform import Rotation as R


class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._planner_prompt = load_prompt(f"{env}/{self._cfg['planner_prompt_fname']}.txt")
        self._action_state_prompt = load_prompt(f"{env}/{self._cfg['vision_prompt_fname']}.txt")

        self._stop_tokens = [self._cfg['stop']]
        self._context = None
        self.mask_path = "./tmp/masks/"
        self.image_path = "./tmp/images/"
        self.state_json_path = "./tmp/state_front.json"
        #set your api_key Qwen
        self.api_key= "sk-6c92e8dc39534beea619a0470d8a2571"
        self.base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

    def get_last_filename(self,folder):
        filenames = os.listdir(folder)
        filename = filenames[-1]
        return f"{folder}{filename}"

    def generate_planning(self, query):
        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'

        planner_prompt = self._planner_prompt

        if self._context :
            user_query = f"# Objects : {self._context}\n" + user_query

        client = OpenAI(api_key=self.api_key,base_url=self.base_url)
        
        filepath = self.get_last_filename(self.image_path)
        base64_image = encode_image(filepath)

        completion = client.chat.completions.create(
            model=self._cfg['vision_model'],
            messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene image.\n{planner_prompt}\nThe above are some examples of planning, please give the corresponding planning according to the image I gave you next:\n{user_query}. The output format likely is\n" + "planner : ['', '', '', '']\nOther than that, don't give me any superfluous information and hints"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}],
        )

        planner = completion.choices[0].message.content

        planning = json.loads(planner.split(":")[-1].strip())

        return planning
    
    def get_state(self, state_json_path,lock):
      state = read_state(state_json_path,lock)
      return state

    def _vlmapi_call(self,image_path, query, planner ,action, objects):
        client = OpenAI(api_key=self.api_key,base_url=self.base_url)

        base64_image = encode_image(image_path)

        prompt = self._action_state_prompt

        completion = client.chat.completions.create(
            model=self._cfg['vision_model'],  
            messages=[{"role": "user","content": [
                    {"type": "text","text": f"This is a robotic arm operation scene." + f"The format of output should be like {prompt}.\n Objects : {objects}\nQuery : {query}\nPlanner : {planner}\nAction : {action}\nPlease just give me the corresponding json, no explanation and no text required"},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                    }
                    ]}]
            )

        resstr = completion.choices[0].message.content.replace("```","").replace("json","")

        state = json.loads(resstr)

        return state
    

    def __execute_action_state(self, action_state, lock, lmp_env):
        global _map_size, _resolution
        _map_size = lmp_env._map_size
        _resolution = lmp_env._resolution
        while True:
            object_state = self.get_state(self.state_json_path,lock)

            affordable_map = None
            rotation_map = lmp_env._get_default_voxel_map('rotation')()
            velocity_map = lmp_env._get_default_voxel_map('velocity')()
            gripper_map = lmp_env._get_default_voxel_map('gripper')()
            avoidance_map = lmp_env._get_default_voxel_map('obstacle')()

            affordable = action_state["affordable"]
            avoidance = action_state["avoid"]
            gripper = action_state["gripper"]
            rotation = action_state["rotation"]
            velocity = action_state["velocity"]

            affordable_set = affordable["set"]
            avoidance_set = avoidance["set"]
            gripper_set = gripper["set"]
            rotation_set = rotation["set"]
            velocity_set = velocity["set"]

            movable = action_state["movable"]
            movable_var = object_state[movable]["obs"]
            
            if affordable_set != "default" :
                affordable_map = lmp_env._get_default_voxel_map('target')()
                affordable_var = affordable["object"]
                object = object_state[affordable_var]["obs"]
                center_x, center_y, center_z = eval(affordable["center_x, center_y, center_z"])
                (min_x, min_y, min_z), (max_x, max_y, max_z) = eval(affordable["(min_x, min_y, min_z), (max_x, max_y, max_z)"])
                x = eval(affordable["x"])
                y = eval(affordable["y"])
                z = eval(affordable["z"])
                target_affordance = affordable["target_affordance"]
                affordable_map[x,y,z] = target_affordance

            if avoidance_set != "default" :
                avoidance_var = action_state["avoid"]["object"]
                if avoidance_var not in object_state.keys():
                    print(f"Object {avoidance_var} not found in scene in this step.")
                    pass
                object = object_state[avoidance_var]["obs"]
                center_x, center_y, center_z = eval(avoidance["center_x, center_y, center_z"])
                (min_x, min_y, min_z), (max_x, max_y, max_z) = eval(avoidance["(min_x, min_y, min_z), (max_x, max_y, max_z)"])
                x = eval(avoidance["x"])
                y = eval(avoidance["y"])
                z = eval(avoidance["z"])
                radius_cm = avoidance["radius_cm"]
                value = avoidance["value"]
                avoidance_map = set_voxel_by_radius(avoidance_map, [x,y,z], radius_cm, value)
                
            if gripper_set != "default" :
                if "object" not in action_state["gripper"].keys():
                    gripper_map[:, :, :] = 1
                    pass
                gripper_var = action_state["gripper"]["object"]
                object = object_state[gripper_var]["obs"]
                center_x, center_y, center_z = eval(gripper["center_x, center_y, center_z"])
                (min_x, min_y, min_z), (max_x, max_y, max_z) = eval(gripper["(min_x, min_y, min_z), (max_x, max_y, max_z)"])
                x = eval(gripper["x"])
                y = eval(gripper["y"])
                z = eval(gripper["z"])
                radius_cm = gripper["radius_cm"]
                value = gripper["value"]
                gripper_map = set_voxel_by_radius(gripper_map, [x,y,z], radius_cm, value)

            '''if rotation_set != "default" :
                rotation_var = action_state["rotation"]["object"]
                if rotation_var not in object_state.keys():
                    print(f"Object {rotation_var} not found in scene in this step.")
                    pass
                object = object_state[rotation_var]["obs"]
                target_rotation = eval(rotation["target_rotation"])
                rotation_map[:, :, :] = target_rotation'''

            if velocity_set != "default" :
                target_velocity = velocity["target_velocity"]
                velocity_map[:] = target_velocity

            stop = lmp_env.execute(movable_var,affordable_map,avoidance_map,rotation_map,velocity_map,gripper_map)
            # 到达目标位置
            if stop:
                break


    def __call__(self, query, lock, lmp_env):
        planning = self.generate_planning(query)
        planning_ = planning.copy()
        action = planning.pop(0)
        while len(planning) != 0:
            print(f"Action: {action}")
            filepath = self.get_last_filename(self.mask_path)
            action_state  = self._vlmapi_call(filepath,query=query,planner=planning_,action=action,objects=self._context)
            print(action_state)
            self.__execute_action_state(action_state, lock, lmp_env)
            action = planning.pop(0)


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e


def cm2index(cm, direction):
    global _map_size, _resolution
    if isinstance(direction, str) and direction == 'x':
      x_resolution = _resolution[0] * 100  # resolution is in m, we need cm
      return int(cm / x_resolution)
    elif isinstance(direction, str) and direction == 'y':
      y_resolution = _resolution[1] * 100
      return int(cm / y_resolution)
    elif isinstance(direction, str) and direction == 'z':
      z_resolution = _resolution[2] * 100
      return int(cm / z_resolution)
    else:
      # calculate index along the direction
      assert isinstance(direction, np.ndarray) and direction.shape == (3,)
      direction = normalize_vector(direction)
      x_cm = cm * direction[0]
      y_cm = cm * direction[1]
      z_cm = cm * direction[2]
      x_index = cm2index(x_cm, 'x')
      y_index = cm2index(y_cm, 'y')
      z_index = cm2index(z_cm, 'z')
      return np.array([x_index, y_index, z_index])
  
def index2cm(index, direction=None):
    global _map_size, _resolution
    if direction is None:
      average_resolution = np.mean(_resolution)
      return index * average_resolution * 100  # resolution is in m, we need cm
    elif direction == 'x':
      x_resolution = _resolution[0] * 100
      return index * x_resolution
    elif direction == 'y':
      y_resolution = _resolution[1] * 100
      return index * y_resolution
    elif direction == 'z':
      z_resolution = _resolution[2] * 100
      return index * z_resolution
    else:
      raise NotImplementedError
    
def pointat2quat(vector):
    assert isinstance(vector, np.ndarray) and vector.shape == (3,), f'vector: {vector}'
    return pointat2quat(vector)    # append the last waypoint a few more times for the robot to stabilize

def vec2quat(vec):
    vec = vec / np.linalg.norm(vec)
    # 目标方向是z轴
    target = np.array([0, 0, 1])
    # 使用Rotation.from_rotvec来获取从z轴到v的旋转
    rotation = R.align_vectors([vec], [target])[0]
    # 获取四元数
    quat = rotation.as_quat()  # 返回四元数 [x, y, z, w] 格式
    return quat


def set_voxel_by_radius(voxel_map, voxel_xyz, radius_cm=0, value=1):
    """given a 3D np array, set the value of the voxel at voxel_xyz to value. If radius is specified, set the value of all voxels within the radius to value."""
    global _map_size, _resolution
    voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
    if radius_cm > 0:
      radius_x = cm2index(radius_cm, 'x')
      radius_y = cm2index(radius_cm, 'y')
      radius_z = cm2index(radius_cm, 'z')
      # simplified version - use rectangle instead of circle (because it is faster)
      min_x = max(0, voxel_xyz[0] - radius_x)
      max_x = min(_map_size, voxel_xyz[0] + radius_x + 1)
      min_y = max(0, voxel_xyz[1] - radius_y)
      max_y = min(_map_size, voxel_xyz[1] + radius_y + 1)
      min_z = max(0, voxel_xyz[2] - radius_z)
      max_z = min(_map_size, voxel_xyz[2] + radius_z + 1)
      voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
    return voxel_map
