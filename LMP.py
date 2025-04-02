from openai import RateLimitError, APIConnectionError,OpenAI
from utils import load_prompt
import time
from VLM_demo import encode_image,read_state
import json
import os
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qinverse
import numpy as np
from scipy.spatial.transform import Rotation as R


class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._planner_prompt = load_prompt(f"{env}/{self._cfg['planner_prompt_fname']}.txt")
        self._stop_tokens = [self._cfg['stop']]
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self._context = None
        self.mask_path = "./tmp/masks/"
        self.image_path = "./tmp/images/"
        self.state_json_path = "./tmp/state_front.json"
        #set your api_key Qwen
        self.api_key= "sk-df55df287b2c420285feb77137467576"
        self.base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

    def get_last_filename(self,folder):
        filenames = os.listdir(folder)
        filename = filenames[-1]
        return f"{folder}{filename}"

    def generate_planning(self, query, model):
        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'

        planner_prompt = self._planner_prompt

        if self._context :
            user_query = f"# Objects : {self._context}\n" + user_query

        client = OpenAI(api_key=self.api_key,base_url=self.base_url)
        
        filepath = self.get_last_filename(self.image_path)
        base64_image = encode_image(filepath)

        completion = client.chat.completions.create(
            model=model,
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
        client = OpenAI(
            api_key="sk-df55df287b2c420285feb77137467576",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        base64_image = encode_image(image_path)

        prompt = load_prompt(f"vlm_rlbench/state.txt")

        completion = client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",  
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
                x = eval(avoidance["x"])
                y = eval(avoidance["y"])
                z = eval(avoidance["z"])
                radius_cm = avoidance["radius_cm"]
                value = avoidance["value"]
                avoidance_map = lmp_env.set_voxel_by_radius(avoidance_map, [x,y,z], radius_cm, value)
                
            if gripper_set != "default" :
                if "object" not in action_state["gripper"].keys():
                    gripper_map[:, :, :] = 1
                    pass
                gripper_var = action_state["gripper"]["object"]
                object = object_state[gripper_var]["obs"]
                x = eval(gripper["x"])
                y = eval(gripper["y"])
                z = eval(gripper["z"])
                radius_cm = gripper["radius_cm"]
                value = gripper["value"]
                gripper_map = lmp_env.set_voxel_by_radius(gripper_map, [x,y,z], radius_cm, value)

            if rotation_set != "default" :
                rotation_var = action_state["rotation"]["object"]
                if rotation_var not in object_state.keys():
                    print(f"Object {rotation_var} not found in scene in this step.")
                    pass
                object = object_state[rotation_var]["obs"]
                target_rotation = eval(rotation["target_rotation"])
                rotation_map[:, :, :] = target_rotation

            if velocity_set != "default" :
                target_velocity = velocity["target_velocity"]
                velocity_map[:] = target_velocity

            stop = lmp_env.execute(movable_var,affordable_map,avoidance_map,rotation_map,velocity_map,gripper_map)
            # 到达目标位置
            if stop:
                break


    def __call__(self, query, lock, lmp_env):
        planning = self.generate_planning(query,self._cfg['planner_model'])
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
    

def vec2quat(vec):
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