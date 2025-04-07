from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks
import threading
import time
import os
import time
import queue
import shutil 

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

# 创建锁
lock = threading.Lock()
q = queue.Queue()

def update_state(lock,q):
    lmp_env.update_mask_entities(lock,q)
    shutil.rmtree("tmp/images")
    shutil.rmtree("tmp/masks")
    os.remove(config["json_path"])

def run_voxposer_ui(instruction,lock,lmp_env,q):
    voxposer_ui(instruction,lock,lmp_env)
    q.put(0)
    

thread1 = threading.Thread(target=update_state, args=(lock,q,))
thread2 = threading.Thread(target=run_voxposer_ui, args=(instruction,lock,lmp_env,q,))

thread1.start()
while not os.path.exists(config["json_path"]):
    time.sleep(1)
thread2.start()
thread2.join()
thread1.join()
