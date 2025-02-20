

An embodied grasping method with spatial state awareness

**1.Introduction**

目前，LLM已经在各个方面取得了飞速的进展，融合了视觉模态的视觉语言模型VLMs**[GPT4V,LLaVa]**可以通过输入图像在推理能力与对世界认识的泛化方面有了巨大提升。Qwen-2.5-VL具有非常优越的json输出能力，将VLMs、LLMs和机器人进行结合，使机器人可以与现实世界进行积极的物理交互完成更加广泛的任务,目前主要的方法有两类：

第一类方法为**modular hierarchical policies**模块化分层规划**[VoxPoser,CoPa,MOKA,ManipLLM]**首先利用VLMs对世界进行对齐，然后使用LLMs生成规划(代码**(code as policies)**或者结构化语言**(SayCan,Statler)**)，使用底层原语实现机器人运动；优点是其具有可解释性，规划的路径比较细致；缺点是大模型LLM反应时间较长，不容易根据目前的场景状态实时更新反馈目前路径。

第二类方法为 **end-to-end policies **端到端策略，也叫**VLA(Vision-Language-Action)**模型，**[RT-1]**使用encoder-decoder类似架构从头训练，将机器人状态和视觉观测作为latent condition，然后用action query-baseed Transformer decoder解码出动作；**[RT-2,OpenVLA]**使用预训练的LLM/VLM，将action当成token直接预测，借鉴已经比较成熟的语言模型；**[DiffusionPolicy]**使用Diffusion Model多步降噪生成运动轨迹；**[Octo]**使用LLM压缩多模态表征，Diffusion作为action expert精细化输出action trajectories。

同时，最近有学者提出了**[LCB]**:结合两种方式的优点进行操作，既使用了LLM的高级推理能力，具有可解释性，同时通过学习<**token>**令牌，输入Policy Network生成低级控制动作，克服仅依赖语言作为接口层的固有限制。

我们受到自动驾驶领域DriveVLM工作的启发，得到了一种思路:使用VLM实现对世界状态的感知，同时根据感知编排动作，最后实现路径的规划使用到具身智能中，具体来说，首先在模块化方法中的VLMs使用Qwen-2.5-VL得到物体框然后使用SAM2进行对齐Grounding，找到目标，同时得到各个物体的状态，跟踪mask，使用VoxPoser的方式生成低频轨迹作为路径，



本文的主要贡献主要有：

1.在本研究中，我们的目的是使用模块化分层策略生成的路径作为参考路径与端到端快速生成的路径进行结合，使机械臂路径更快，且异步进行，使机械臂路径倾向于模块化路径的稳定。

2.设计了一种Benchmark，实现了复杂富含运动的场景，并对已有的方法和我们的方法进行对比评估，说明了我们路径融合方法的优越性。

3.将该路径融合技术使用到太空技术中，在解决复杂太空抓取任务中具有先进性。

**2.Relatedwork**

modular hierarchical policies-介绍一下分层策略等方法，最后提到自己使用的分层策略方法。</br>
end-to-end policies -介绍一下端到端等方法，最后提到自己的端到端方法。</br>
融合方法：引入一些融合方法的论文，并引出自己的融合策略**(目前暂定为使用LLM评估使用路径)**。

**3.Method**

3.1 问题描述</br>

3.2 模型框架</br>

**4.BenchMark**

**5.Experiments**

5.1 实验设置</br>

5.2 对真实世界操作</br>

5.3 对比实验</br>

5.4 第三个贡献</br>

**6.Conclusion**

在本文的工作中，文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本。







已经存在的工作使用3D点云生成抓取姿态**[GraspNet,CoPa]**，但是仍然没有广泛运用到机器人领域真实世界的推理与对齐，最近，一些工作使用3D对齐的方法**[PointLLM,VLM-Grounder,3D-LLM]**实现了对点云目标的对齐与推理，这样就提出了问题：如何使用已有的对3D点云的对齐，实现真正对真实世界的任务规划。

5.Discussion & Limitations