from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks
import threading
import os
import time

#load config
config_path = "configs/vlm_rlbench_config.yaml"
config = get_config(config_path=config_path)

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
env.load_task(tasks.PutRubbishInBin)
#env.load_task(tasks.LampOff)
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

bbox_entities = lmp_env.update_box()

def update_state(bbox_entities):
    print("thread 1 running...")
    lmp_env.update_mask_entities(bbox_entities)

def run_voxposer_ui(instruction):
    print("thread 2 running...")
    try:
        voxposer_ui(instruction)
        print("thread 2 done.")
    except Exception as e:
        print(f"Error in thread 2: {e}")


thread1 = threading.Thread(target=update_state, args=(bbox_entities,))
thread2 = threading.Thread(target=run_voxposer_ui, args=(instruction,))

thread1.start()

while not os.path.exists(f"./tmp/state_{lmp_env.cam_name}.json"):
    time.sleep(1)

thread2.start()

thread2.join()

os.remove(f"./tmp/state_{lmp_env.cam_name}.json")

os._exit(0)
