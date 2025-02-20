~~~
1.用户输入高级prompt指令(把垃圾放到垃圾桶中)至LMP对象
2.LMP对象planner首先加载context，生成skill级composer规划LMP对象
(抓住垃圾、回到原位、移动到垃圾桶上方10cm、松开夹抓)
3.递归逐行执行高级规划即执行composer(skill)
	1）使用自定义API获得目标位姿等信息
	(在现实世界使用与VLM交互实现)
	2）处理得到的三维价值图中的价值
	(使用欧式距离获得每个点到目标点的距离得到每个点的价值，平滑到0-1之间)
	3）使用贪婪搜索算法获得价值点轨迹
	4）更新点云图显示每个点价值和轨迹
	5）使用control.apply_actiion逐点移动
4.执行完所有动作结束
~~~

#一、加载配置 (arguments.py) config = get_config('rlbench')

加载configs/rlbench_config.yaml文件

定义了planner,controller,visualizer,lmp的参数

#二、将可视化参数配置传入到点云可视化中 (visualizers.py) visualizer = ValueMapVisualizer(config['visualizer'])

只进行init，定义点云图保存路径，质量，尺寸大小，进行update_quality定义下采样比例，最多点的数量，不透明度，代价地图表面计数

#三、初始化VoxPoserRLBench环境 (envs/rlbench_env.py) env = VoxPoserRLBench(visualizer=visualizer)

定义visualizer工作区间边界，使用visualizers中的update_bounds更新边界，绘图边界分别缩小和向外扩展0.15,计算出每个轴方向的比例系数，转换为缩放比例存储；

定义了5个相机的名称，并获取相机的外参矩阵4X4(3X3是旋转矩阵，1X3三平移向量)，旋转矩阵*前向向量，将前向向量转换到当前相机的前向方法，获得视线方向(look-at向量)并归一化使其长度为1

保存点云图为json文件

#四、配置LMP (interfaces.py) lmps, lmp_env = setup_LMP(env, config, debug=False)

首先加载全部配置，初始化LMP_interface对象，在其中初始化PathPlanner planners.py(加载配置)和Controller controllers.py对象(加载配置以及动力学模型PushingDynamicsModel dynamics_models.py)，然后计算每个体素的大小，输出为：

~~~
##################################################
## voxel resolution: [0.0105   0.0131 0.01  ]
##################################################
~~~

创建LMPs可以交互的第三方库API(np,euler2quat,quat2euler,qinverse,qmult)

创建拓展的API(在interfaces.py中LMP_interface中可以调用的用户自定义库API方法其中不以_开头的，例如get_ee_pos,detect,execute,cm2index,index2cm,pointat2quat,
set_voxel_by_radius,get_empty_affordance_map等;_开头的方法是帮助辅助方法如_world_to_voxel等)

创建LMPs可以调用其他LMPs中的API(在rlbench_config.yaml中例如parse_query_obj,get_affordance_map等)

将交互API、拓展API、其他API作为LMP对象一起作为low_level_lmps；创建LMP composer技能；创建LMP 任务规划task_planner高级语言命令。将所有的LMP对象组合成一个字典作为lmps返回，将LMP_interface对象作为lmp_env返回。

#五、得到voxposer_ui(四中的高级任务规划task_planner) voxposer_ui = lmps['plan_ui']

#六、加载任务(放垃圾到垃圾桶) env.load_task(tasks.PutRubbishInBin)

调用VoxPoserRLBench对象的load_task函数，布置tasks.PutRubbishInBin中的场景，并记录与任务相关的机器人和物体的掩码ID为后续操作提供基础。它还建立了物体名称和ID的映射，方便在后续任务执行时通过名称快速找到对应的物体。

#七、随机选取任务描述并更新可视化 descriptions, obs = env.reset()

选取该任务的所有描述;得到场景变量(五个相机的rgb和depth，以及关节速度等)并将场景变量四元数作变换保持一致性(_process_obs)，初始化可视化env环境的初始状态和最终状态，更新场景(得到场景变量在3d图中的点和颜色，调用visualizeis.py中的update_scene_points更新点云图)

返回任务的所有描述以及场景变量

#八、设置被voxposer使用的目标名set_lmp_objects(lmps,env.get_object_names())

lmps是所有LMP可以调用的API方法字典,env.get_object_names()是该任务场景中所有的目标名

set_lmp_objects中如果lmps是字典，则得到所有lmps的值即所有API的LMP对象(LMP.py)，设置每个对象的上下文_context目标为传入的当前场景目标名

#九、随机选取一个任务字符串 instruction = np.random.choice(descriptions)

#十、将任务描述进行高级任务规划 voxposer_ui(instruction)

voxposer_ui是高级任务规划器,调用LMP类中的__call__()进行使用

首先构建高级命令的prompt:</br>
base_prompt为planner_prompt.txt的内容</br>
首先从utils导入所有的库并进行prompt替换库</br>
判断是否保持执行以及包含上下文</br>
定义user_qurey为标准形式，在prompt末尾加入user_query并返回</br>

进行大模型API调用得到代码字符串</br>
new_query为用户的输入即user2，user1为上下文并添加一些对LLM的限制条件prompt，判断user1中是否有物体，如果有则添加到user2前面，组成messages</br>
判断该prompt曾经生成过cache(LLM_cache.py实现),如果有则使用cache，否则调用API生成代码，并保存到cache中</br>
显示调用花费多少时间：</br>
~~~
*** OpenAI API call took 3.73s ***
~~~
首次调用planner上下文为场景目标列表，将生成代码放到user_query后，输出</br>
~~~
########################################
## "planner" generated code
## context: "objects = ['bin', 'rubbish', 'tomato1', 'tomato2']"
########################################
objects = ['bin', 'rubbish', 'tomato1', 'tomato2']
# Query: drop the rubbish into the bin.
composer("grasp the rubbish")
composer("back to default pose")
composer("move to 10cm above the bin")
composer("open gripper")
# done
~~~
记录执行历史为字符串形式，安全执行生成代码，传入代码字符串及其他LMP对象，以及planner中LLM参数</br>
exec调用，exec(code_str, custom_gvars, lvars)，第一个是字符串，第二个是可以调用的全局变量，这里是LMP对象或函数API，第三个是局部变量即LLM参数

**然后会依次调用上述代码框中的LMP对象，首次调用composer("grasp the rubbish")**

同样先加载prompt然后与user_query拼接得到LLM输出代码
~~~
*** OpenAI API call took 5.74s ***
########################################
## "composer" generated code
########################################
# Query: grasp the rubbish.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point at the center of the rubbish')
gripper_map = get_gripper_map('open everywhere except 1cm around the rubbish')
execute(movable, affordance_map=affordance_map, gripper_map=gripper_map)
~~~
这里调用比较多的是自定义API，一直调用直到execute(movable, affordance_map=affordance_map, gripper_map=gripper_map)

**注意：这里的LMP如果不是planner和composer则修改为def函数，返回值为该函数，直到execute调用这些变量，这些函数会依次调用，保持姿态的同一性**

#十一、各个自定义API的含义

detect：如果是末端执行器或者桌面则返回其名字、3D图位置、AABB(两个点坐标)包围盒,世界位置坐标；如果是其他物品则根据物体的名称从多个摄像头获取该物体的真实世界3D点云观测值及法向量，然后转化到3D体素图的点，同时计算AABB包围盒坐标。这里返回的position,aabb,occupancy_map为体素世界中的点,其他为世界坐标系即VREP环境。

get_affordance_map：根据对应的图返回对应的空体素坐标，全为0或者全为1。

execute：首先使用planner生成轨迹点的路径坐标，然后使用控制器使其按照路径移动

planner.optimize(得到轨迹点的路径坐标)：根据目标地图与障碍物地图生成成本地图，初始化路径和当前位置，不断计算当前附近的体素，选择成本最低的体素作为下一个位置，避免回退，增加当前位置的体素值来强制改变路径

~~~
[planners.py | 21:53:40.656] start
[planners.py | 21:53:40.844] start optimizing, start_pos: [52 49 71]
[planners.py | 21:53:40.897] optimization finished; path length: 36
[planners.py | 21:53:40.898] after postprocessing, path length: 36
[planners.py | 21:53:40.898] last waypoint: [62.        76.1782044  1.       ]
~~~


controller.execute：如果移动的是末端执行器则不考虑动力学否则考虑动力学，移动使用apply_action(rlbench_env.py)，移动

~~~
[interfaces.py | 21:53:40] planner time: 0.245s
[interfaces.py | 21:53:40] overwriting gripper to less common value for the last waypoint
** saving visualization to ./visualizations/21:53:40.html ...
** saving visualization to ./visualizations/latest.html ...
** save to ./visualizations/21:53:40.html
[interfaces.py | 21:53:41] start executing path via controller (38 waypoints)
[interfaces.py | 21:53:44] completed waypoint 1 (wp: [ 0.274 -0.002  1.469], actual: [ 0.274 -0.002  1.469], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.798)
[interfaces.py | 21:53:44] completed waypoint 2 (wp: [0.301 0.019 1.449], actual: [0.3   0.018 1.449], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.769)
[interfaces.py | 21:53:45] completed waypoint 3 (wp: [0.325 0.043 1.429], actual: [0.324 0.043 1.428], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.737)
[interfaces.py | 21:53:45] completed waypoint 4 (wp: [0.343 0.069 1.409], actual: [0.343 0.068 1.409], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.708)
[interfaces.py | 21:53:45] completed waypoint 5 (wp: [0.358 0.096 1.388], actual: [0.357 0.095 1.389], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.678)
[interfaces.py | 21:53:45] completed waypoint 6 (wp: [0.369 0.124 1.368], actual: [0.369 0.123 1.368], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.649)
[interfaces.py | 21:53:45] completed waypoint 7 (wp: [0.378 0.153 1.348], actual: [0.377 0.151 1.348], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.62)
[interfaces.py | 21:53:46] completed waypoint 8 (wp: [0.383 0.182 1.328], actual: [0.383 0.18  1.328], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.592)
[interfaces.py | 21:53:46] completed waypoint 9 (wp: [0.386 0.21  1.308], actual: [0.386 0.208 1.308], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.564)
[interfaces.py | 21:53:46] completed waypoint 10 (wp: [0.388 0.237 1.287], actual: [0.388 0.236 1.287], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.538)
[interfaces.py | 21:53:46] completed waypoint 11 (wp: [0.388 0.273 1.257], actual: [0.388 0.272 1.257], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.502)
[interfaces.py | 21:53:47] completed waypoint 12 (wp: [0.387 0.293 1.237], actual: [0.387 0.293 1.237], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.478)
[interfaces.py | 21:53:47] completed waypoint 13 (wp: [0.386 0.311 1.217], actual: [0.386 0.311 1.216], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.456)
[interfaces.py | 21:53:47] completed waypoint 14 (wp: [0.384 0.325 1.196], actual: [0.384 0.325 1.196], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.435)
[interfaces.py | 21:53:47] completed waypoint 15 (wp: [0.383 0.337 1.176], actual: [0.383 0.336 1.176], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.414)
[interfaces.py | 21:53:47] completed waypoint 16 (wp: [0.383 0.345 1.156], actual: [0.383 0.345 1.157], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.395)
[interfaces.py | 21:53:48] completed waypoint 17 (wp: [0.383 0.352 1.136], actual: [0.382 0.352 1.135], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.373)
[interfaces.py | 21:53:48] completed waypoint 18 (wp: [0.383 0.356 1.116], actual: [0.382 0.356 1.116], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.354)
[interfaces.py | 21:53:48] completed waypoint 19 (wp: [0.383 0.358 1.095], actual: [0.382 0.358 1.095], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.333)
[interfaces.py | 21:53:48] completed waypoint 20 (wp: [0.383 0.358 1.075], actual: [0.382 0.358 1.076], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.314)
[interfaces.py | 21:53:48] completed waypoint 21 (wp: [0.383 0.358 1.055], actual: [0.382 0.358 1.055], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.293)
[interfaces.py | 21:53:49] completed waypoint 22 (wp: [0.383 0.358 1.035], actual: [0.382 0.358 1.035], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.273)
[interfaces.py | 21:53:49] completed waypoint 23 (wp: [0.383 0.358 1.015], actual: [0.382 0.358 1.015], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.253)
[interfaces.py | 21:53:49] completed waypoint 24 (wp: [0.383 0.361 0.994], actual: [0.382 0.36  0.994], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.232)
[interfaces.py | 21:53:49] completed waypoint 25 (wp: [0.383 0.364 0.974], actual: [0.382 0.363 0.974], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.212)
[interfaces.py | 21:53:50] completed waypoint 26 (wp: [0.383 0.367 0.954], actual: [0.382 0.367 0.954], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.192)
[interfaces.py | 21:53:50] completed waypoint 27 (wp: [0.383 0.369 0.944], actual: [0.382 0.368 0.943], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.182)
[interfaces.py | 21:53:50] completed waypoint 28 (wp: [0.383 0.372 0.924], actual: [0.382 0.371 0.924], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.162)
[interfaces.py | 21:53:50] completed waypoint 29 (wp: [0.383 0.374 0.904], actual: [0.382 0.374 0.904], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.143)
[interfaces.py | 21:53:50] completed waypoint 30 (wp: [0.383 0.376 0.883], actual: [0.382 0.375 0.883], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.123)
[interfaces.py | 21:53:51] completed waypoint 31 (wp: [0.383 0.376 0.863], actual: [0.382 0.376 0.863], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.103)
[interfaces.py | 21:53:51] completed waypoint 32 (wp: [0.383 0.375 0.843], actual: [0.382 0.375 0.843], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.083)
[interfaces.py | 21:53:51] completed waypoint 33 (wp: [0.383 0.372 0.823], actual: [0.382 0.372 0.823], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.063)
[interfaces.py | 21:53:51] completed waypoint 34 (wp: [0.383 0.368 0.803], actual: [0.382 0.368 0.803], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.043)
[interfaces.py | 21:53:51] completed waypoint 35 (wp: [0.383 0.362 0.782], actual: [0.382 0.362 0.782], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.022)
[interfaces.py | 21:53:51] skip waypoint 36 because it is moving in opposite direction of the final target
[interfaces.py | 21:53:51] skip waypoint 37 because it is moving in opposite direction of the final target
[interfaces.py | 21:53:52] completed waypoint 38 (wp: [0.383 0.353 0.762], actual: [0.383 0.353 0.763], target: [0.383 0.353 0.762], start: [ 0.274 -0.002  1.469], dist2target: 0.001)
[interfaces.py | 21:53:52] reached target; terminating 
[interfaces.py | 21:53:52] finished executing path via controller
~~~

#十二、点云价值的平滑

~~~
#返回每个点到最近非零点的距离
target_map = distance_transform_edt(1 - target_map)
#平滑价值，使其的值为0-1之间
target_map = normalize_map(target_map)
~~~


#十二、一段完整的运行日志
~~~
##################################################
## voxel resolution: [0.0105 0.0131 0.01  ]
##################################################


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "planner" generated code
## context: "objects = ['bin', 'rubbish', 'tomato1', 'tomato2']"
########################################
objects = ['bin', 'rubbish', 'tomato1', 'tomato2']
# Query: drop the rubbish into the bin.
composer("grasp the rubbish")
composer("back to default pose")
composer("move to 10cm above the bin")
composer("open gripper")
# done


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "composer" generated code
########################################
# Query: grasp the rubbish.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point at the center of the rubbish')
gripper_map = get_gripper_map('open everywhere except 1cm around the rubbish')
execute(movable, affordance_map=affordance_map, gripper_map=gripper_map)


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "parse_query_obj" generated code
## context: "objects = ['bin', 'rubbish', 'tomato1', 'tomato2']"
########################################
objects = ['bin', 'rubbish', 'tomato1', 'tomato2']
# Query: gripper.
gripper = detect('gripper')
ret_val = gripper


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "get_affordance_map" generated code
########################################
# Query: a point at the center of the rubbish.
affordance_map = get_empty_affordance_map()
rubbish = parse_query_obj('rubbish')
x, y, z = rubbish.position
affordance_map[x, y, z] = 1
ret_val = affordance_map


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "get_gripper_map" generated code
########################################
# Query: open everywhere except 1cm around the rubbish.
gripper_map = get_empty_gripper_map()
# open everywhere
gripper_map[:, :, :] = 1
# close when 1cm around the rubbish
rubbish = parse_query_obj('rubbish')
set_voxel_by_radius(gripper_map, rubbish.position, radius_cm=1, value=0)
ret_val = gripper_map


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "parse_query_obj" generated code
## context: "objects = ['bin', 'rubbish', 'tomato1', 'tomato2']"
########################################
objects = ['bin', 'rubbish', 'tomato1', 'tomato2']
# Query: rubbish.
rubbish = detect('rubbish')
ret_val = rubbish


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "parse_query_obj" generated code
## context: "objects = ['bin', 'rubbish', 'tomato1', 'tomato2']"
########################################
objects = ['bin', 'rubbish', 'tomato1', 'tomato2']
# Query: rubbish.
rubbish = detect('rubbish')
ret_val = rubbish


[planners.py | 11:2:25.158] start
[planners.py | 11:2:25.355] start optimizing, start_pos: [52 49 71]
[planners.py | 11:2:25.410] optimization finished; path length: 36
[planners.py | 11:2:25.411] after postprocessing, path length: 36
[planners.py | 11:2:25.411] last waypoint: [36.70559006 50.22382835  1.        ]
[interfaces.py | 11:2:25] planner time: 0.255s
[interfaces.py | 11:2:25] overwriting gripper to less common value for the last waypoint
** saving visualization to ./visualizations/11-2-25.html ...
** saving visualization to ./visualizations/latest.html ...
** save to ./visualizations/11-2-25.html
[interfaces.py | 11:2:25] start executing path via controller (38 waypoints)
[interfaces.py | 11:2:26] completed waypoint 1 (wp: [ 2.870e-01 -1.000e-03  1.469e+00], actual: [ 0.287 -0.002  1.469], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.727)
[interfaces.py | 11:2:26] completed waypoint 2 (wp: [0.255 0.002 1.449], actual: [0.256 0.002 1.449], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.701)
[interfaces.py | 11:2:26] completed waypoint 3 (wp: [0.227 0.004 1.429], actual: [0.228 0.004 1.429], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.676)
[interfaces.py | 11:2:27] completed waypoint 4 (wp: [0.203 0.006 1.409], actual: [0.204 0.006 1.408], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.652)
[interfaces.py | 11:2:27] completed waypoint 5 (wp: [0.183 0.007 1.388], actual: [0.183 0.007 1.388], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.63)
[interfaces.py | 11:2:27] completed waypoint 6 (wp: [0.165 0.007 1.368], actual: [0.166 0.007 1.368], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.608)
[interfaces.py | 11:2:27] completed waypoint 7 (wp: [0.151 0.007 1.348], actual: [0.151 0.007 1.348], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.587)
[interfaces.py | 11:2:28] completed waypoint 8 (wp: [0.139 0.007 1.328], actual: [0.14  0.007 1.328], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.567)
[interfaces.py | 11:2:28] completed waypoint 9 (wp: [0.13  0.007 1.308], actual: [0.131 0.007 1.308], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.546)
[interfaces.py | 11:2:28] completed waypoint 10 (wp: [0.124 0.007 1.287], actual: [0.124 0.007 1.287], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.525)
[interfaces.py | 11:2:28] completed waypoint 11 (wp: [0.116 0.007 1.257], actual: [0.116 0.007 1.257], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.495)
[interfaces.py | 11:2:29] completed waypoint 12 (wp: [0.114 0.009 1.237], actual: [0.114 0.009 1.237], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.475)
[interfaces.py | 11:2:29] completed waypoint 13 (wp: [0.113 0.011 1.217], actual: [0.113 0.011 1.217], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.455)
[interfaces.py | 11:2:29] completed waypoint 14 (wp: [0.113 0.015 1.196], actual: [0.113 0.015 1.197], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.435)
[interfaces.py | 11:2:29] completed waypoint 15 (wp: [0.114 0.018 1.176], actual: [0.114 0.018 1.177], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.415)
[interfaces.py | 11:2:29] completed waypoint 16 (wp: [0.116 0.023 1.156], actual: [0.116 0.023 1.156], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.395)
[interfaces.py | 11:2:30] completed waypoint 17 (wp: [0.117 0.029 1.136], actual: [0.117 0.029 1.136], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.375)
[interfaces.py | 11:2:30] completed waypoint 18 (wp: [0.116 0.035 1.116], actual: [0.116 0.034 1.116], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.355)
[interfaces.py | 11:2:30] completed waypoint 19 (wp: [0.115 0.04  1.095], actual: [0.115 0.04  1.096], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.335)
[interfaces.py | 11:2:30] completed waypoint 20 (wp: [0.113 0.046 1.075], actual: [0.113 0.046 1.076], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.316)
[interfaces.py | 11:2:31] completed waypoint 21 (wp: [0.111 0.051 1.055], actual: [0.111 0.051 1.056], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.296)
[interfaces.py | 11:2:31] completed waypoint 22 (wp: [0.109 0.055 1.035], actual: [0.109 0.055 1.035], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.277)
[interfaces.py | 11:2:31] completed waypoint 23 (wp: [0.107 0.059 1.015], actual: [0.107 0.059 1.015], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.258)
[interfaces.py | 11:2:31] completed waypoint 24 (wp: [0.105 0.06  0.994], actual: [0.104 0.06  0.994], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.238)
[interfaces.py | 11:2:31] completed waypoint 25 (wp: [0.102 0.062 0.974], actual: [0.102 0.062 0.974], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.219)
[interfaces.py | 11:2:32] completed waypoint 26 (wp: [0.099 0.063 0.954], actual: [0.099 0.063 0.954], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.2)
[interfaces.py | 11:2:32] completed waypoint 27 (wp: [0.098 0.062 0.944], actual: [0.098 0.062 0.944], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.19)
[interfaces.py | 11:2:32] completed waypoint 28 (wp: [0.096 0.06  0.924], actual: [0.096 0.06  0.924], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.171)
[interfaces.py | 11:2:32] completed waypoint 29 (wp: [0.094 0.057 0.904], actual: [0.094 0.057 0.904], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.151)
[interfaces.py | 11:2:33] completed waypoint 30 (wp: [0.094 0.053 0.883], actual: [0.093 0.053 0.883], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.131)
[interfaces.py | 11:2:33] completed waypoint 31 (wp: [0.094 0.048 0.863], actual: [0.094 0.048 0.863], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.11)
[interfaces.py | 11:2:34] completed waypoint 32 (wp: [0.095 0.042 0.843], actual: [0.095 0.042 0.843], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.089)
[interfaces.py | 11:2:34] completed waypoint 33 (wp: [0.098 0.035 0.823], actual: [0.098 0.035 0.823], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.068)
[interfaces.py | 11:2:34] completed waypoint 34 (wp: [0.102 0.028 0.803], actual: [0.102 0.028 0.803], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.046)
[interfaces.py | 11:2:35] completed waypoint 35 (wp: [0.107 0.019 0.782], actual: [0.107 0.019 0.782], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.024)
[interfaces.py | 11:2:35] completed waypoint 36 (wp: [0.114 0.01  0.762], actual: [0.115 0.009 0.764], target: [0.114 0.01  0.762], start: [ 2.870e-01 -1.000e-03  1.469e+00], dist2target: 0.002)
[interfaces.py | 11:2:35] reached last waypoint; curr_xyz=[0.11472605 0.00857013 0.76379836], target=[0.11430171 0.00957793 0.7621009 ] (distance: 0.002))
[interfaces.py | 11:2:35] reached target; terminating 
[interfaces.py | 11:2:35] finished executing path via controller
(using cache) *** OpenAI API call took 0.00s ***
########################################
## "composer" generated code
########################################
# Query: back to default pose.
reset_to_default_pose()


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "composer" generated code
########################################
# Query: move to 10cm above the bin.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 10cm above the bin')
execute(movable, affordance_map=affordance_map)


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "parse_query_obj" generated code
## context: "objects = ['bin', 'rubbish', 'tomato1', 'tomato2']"
########################################
objects = ['bin', 'rubbish', 'tomato1', 'tomato2']
# Query: gripper.
gripper = detect('gripper')
ret_val = gripper


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "get_affordance_map" generated code
########################################
# Query: a point 10cm above the bin.
affordance_map = get_empty_affordance_map()
bin = parse_query_obj('bin')
(min_x, min_y, min_z), (max_x, max_y, max_z) = bin.aabb
center_x, center_y, center_z = bin.position
# 10cm above so we add to z-axis
x = center_x
y = center_y
z = max_z + cm2index(10, 'z')
affordance_map[x, y, z] = 1
ret_val = affordance_map


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "parse_query_obj" generated code
## context: "objects = ['bin', 'rubbish', 'tomato1', 'tomato2']"
########################################
objects = ['bin', 'rubbish', 'tomato1', 'tomato2']
# Query: bin.
bin = detect('bin')
ret_val = bin


[planners.py | 11:2:40.275] start
[planners.py | 11:2:40.464] start optimizing, start_pos: [52 49 71]
[planners.py | 11:2:40.502] optimization finished; path length: 22
[planners.py | 11:2:40.502] after postprocessing, path length: 22
[planners.py | 11:2:40.502] last waypoint: [37.54150198 31.08853755 29.        ]
[interfaces.py | 11:2:40] planner time: 0.231s
** saving visualization to ./visualizations/11-2-40.html ...
** saving visualization to ./visualizations/latest.html ...
** save to ./visualizations/11-2-40.html
[interfaces.py | 11:2:40] start executing path via controller (24 waypoints)
[interfaces.py | 11:2:40] completed waypoint 1 (wp: [0.287 0.006 1.469], actual: [0.287 0.006 1.469], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.518)
[interfaces.py | 11:2:41] completed waypoint 2 (wp: [ 0.255 -0.031  1.449], actual: [ 0.255 -0.031  1.449], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.475)
[interfaces.py | 11:2:42] completed waypoint 3 (wp: [ 0.227 -0.065  1.429], actual: [ 0.227 -0.065  1.429], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.436)
[interfaces.py | 11:2:42] completed waypoint 4 (wp: [ 0.203 -0.096  1.409], actual: [ 0.204 -0.096  1.409], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.401)
[interfaces.py | 11:2:42] completed waypoint 5 (wp: [ 0.183 -0.124  1.388], actual: [ 0.184 -0.123  1.388], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.369)
[interfaces.py | 11:2:42] completed waypoint 6 (wp: [ 0.167 -0.149  1.368], actual: [ 0.168 -0.148  1.368], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.34)
[interfaces.py | 11:2:43] completed waypoint 7 (wp: [ 0.154 -0.171  1.348], actual: [ 0.154 -0.171  1.348], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.313)
[interfaces.py | 11:2:43] completed waypoint 8 (wp: [ 0.144 -0.19   1.328], actual: [ 0.144 -0.19   1.328], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.289)
[interfaces.py | 11:2:43] completed waypoint 9 (wp: [ 0.136 -0.207  1.308], actual: [ 0.136 -0.206  1.307], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.265)
[interfaces.py | 11:2:43] completed waypoint 10 (wp: [ 0.13  -0.221  1.287], actual: [ 0.131 -0.22   1.288], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.244)
[interfaces.py | 11:2:44] completed waypoint 11 (wp: [ 0.125 -0.237  1.257], actual: [ 0.125 -0.236  1.258], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.213)
[interfaces.py | 11:2:44] completed waypoint 12 (wp: [ 0.123 -0.244  1.237], actual: [ 0.123 -0.244  1.237], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.192)
[interfaces.py | 11:2:44] completed waypoint 13 (wp: [ 0.123 -0.246  1.227], actual: [ 0.123 -0.246  1.227], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.182)
[interfaces.py | 11:2:44] completed waypoint 14 (wp: [ 0.124 -0.249  1.207], actual: [ 0.124 -0.249  1.207], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.162)
[interfaces.py | 11:2:44] completed waypoint 15 (wp: [ 0.126 -0.25   1.186], actual: [ 0.126 -0.249  1.187], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.142)
[interfaces.py | 11:2:45] completed waypoint 16 (wp: [ 0.128 -0.249  1.166], actual: [ 0.128 -0.249  1.166], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.122)
[interfaces.py | 11:2:45] completed waypoint 17 (wp: [ 0.13  -0.248  1.146], actual: [ 0.13  -0.248  1.146], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.101)
[interfaces.py | 11:2:45] completed waypoint 18 (wp: [ 0.132 -0.246  1.126], actual: [ 0.131 -0.246  1.126], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.081)
[interfaces.py | 11:2:45] completed waypoint 19 (wp: [ 0.132 -0.245  1.106], actual: [ 0.132 -0.245  1.106], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.061)
[interfaces.py | 11:2:45] completed waypoint 20 (wp: [ 0.131 -0.244  1.085], actual: [ 0.131 -0.243  1.086], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.041)
[interfaces.py | 11:2:46] completed waypoint 21 (wp: [ 0.128 -0.243  1.065], actual: [ 0.128 -0.243  1.065], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.021)
[interfaces.py | 11:2:46] skip waypoint 22 because it is moving in opposite direction of the final target
[interfaces.py | 11:2:46] skip waypoint 23 because it is moving in opposite direction of the final target
[interfaces.py | 11:2:46] completed waypoint 24 (wp: [ 0.123 -0.244  1.045], actual: [ 0.123 -0.243  1.045], target: [ 0.123 -0.244  1.045], start: [0.287 0.006 1.469], dist2target: 0.0)
[interfaces.py | 11:2:46] reached target; terminating 
[interfaces.py | 11:2:46] finished executing path via controller
(using cache) *** OpenAI API call took 0.00s ***
########################################
## "composer" generated code
########################################
# Query: open gripper.
movable = parse_query_obj('gripper')
gripper_map = get_gripper_map('open everywhere')
execute(movable, gripper_map=gripper_map)


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "parse_query_obj" generated code
## context: "objects = ['bin', 'rubbish', 'tomato1', 'tomato2']"
########################################
objects = ['bin', 'rubbish', 'tomato1', 'tomato2']
# Query: gripper.
gripper = detect('gripper')
ret_val = gripper


(using cache) *** OpenAI API call took 0.00s ***
########################################
## "get_gripper_map" generated code
########################################
# Query: open everywhere.
gripper_map = get_empty_gripper_map()
gripper_map[:, :, :] = 1
ret_val = gripper_map


[interfaces.py | 11:2:46] finished executing path via controller
~~~