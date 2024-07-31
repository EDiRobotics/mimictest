<div align=center><img width = '600' height ='350' src="https://github.com/user-attachments/assets/8d27de3f-bc5d-40b9-b1d1-2129b10a128f"/></div>


<div align="center">
	
# Mimictest

</div>

A simple testbed for robotics manipulation policies based on [robomimic](https://robomimic.github.io/). It uses Diffusion Policy's [dataset](https://diffusion-policy.cs.columbia.edu/data/training/). All policies are rewritten in a simple way. We may further expand it to the [libero](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark, which is also based on [robosuite](https://github.com/ARISE-Initiative/robosuite) simulator.

We also have policies trained and tested on the [CALVIN](https://github.com/mees/calvin) benchmark, e.g., [GR1-Training](https://github.com/EDiRobotics/GR1-Training) which is the current SOTA on the hardest ABC->D task of CALVIN.

<details>
  <summary> We also recommend other good frameworks / comunities for robotics policy learning. </summary>

- HuggingFace's [LeRobot](https://github.com/huggingface/lerobot), which currently have ACT, Diffusion Policy (only simple pusht task), TDMPC, and VQ-BeT. LeRobot has a nice robotics learning community on this [discord server](https://discord.com/invite/s3KuuzsPFb).

- [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) which implements multiple diffusion algorithms for imitation learning and reinforcement learning. Our implementation of diffusion algorithms is different from CleanDiffuser, but we thank the help of their team members.

- Dr. Mu Yao organizes a nice robitics learning community for Chinese researchers, see [DeepTimber website](https://gamma.app/public/DeepTimber-Robotics-Innovations-Community-A-Community-for-Multi-m-og0uv8mswl1a3q7?mode=doc) and [知乎](https://zhuanlan.zhihu.com/p/698664022).

</details>

**Please remember we build systems for you ヾ(^▽^*)). Feel free to ask [me](zhuohengli@foxmail.com) if you have any question!**

## News
**[2024.7.30]** Add Florence policy with MLP action head & diffusion action head. Add RT-1 policy.  

**[2024.7.16]** Add transformer version of Diffusion Policy.

**[2024.7.15]** Initial release which only contains UNet version of Diffusion Policy.

## Features

<details>
  <summary> Unified State and Action Space. </summary>

- All policies share the same data pre-processing pipeline and predict actions in 3D Cartesian translation  + [6D rotation](https://zhouyisjtu.github.io/project_rotation/rotation.html) + gripper open/close. The 3D translation can be relative to current gripper position (`abs_mode=False`) or the world coordinate (`abs_mode=True`).

- They perceive `obs_horizon` historical observations, generate `chunk_size` future actions, and execute `test_chunk_size` predicted actions. An example with `obs_horizon=3, chunk_size=4, test_chunk_size=2`:

```
Policy sees: 		|o|o|o|
Policy predicts: 	| | |a|a|a|a|
Policy executes:	| | |a|a|
```

- They use image input from both static and wrist cameras.
</details>

<details>
  <summary> Multi-GPU training and simulation. </summary>

- We achieve multi-GPU / multi-machine training with HuggingFace accelerate.
  
- We achieve parallel simulation with asynchronized environment provided by stable-baseline3. In practice, we train and evaluate the model on multiple GPUs. For each GPU training process, there are several parallel environments running on different CPU.

</details>

<details>
  <summary> Optimizing data loading pipeline and profiling. </summary>

- We implement a simple GPU data prefetching mechanism.

- Image preprocessing are performed on GPU, instead of CPU.

- You can perform detailed profiling of the training pipeline by setting `do_profile=True` and check the trace log with `torch_tb_profiler`. Introduction to the [pytorch profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html).

</details>

<details>
  <summary> Sorry...but you should tune the learning rate manually. </summary>
	
- We try new algorithms here so we are not sure when the algorithm will converge before we run it. Thus, we use a simple constant learning rate schduler with warmup. To get the best performance, you should set the learning rate manually: a high learning rate at the beginning and a lower learning rate at the end.

- Sometimes you need to freeze the visual encoder at the first training stage, and unfreeze the encoder when the loss converges in the first stage. It's can be done by setting `freeze_vision_tower=<True/False>` in the script.

</details>

## Supported Policies

We implement the following algorithms:

<details>
  <summary> Google's RT1. </summary>

- [Original implementation](https://github.com/google-research/robotics_transformer).

- Our implementation supports EfficientNet v1/v2 and you can directly load pretrained weights by torchvision API. Google's implementation only supports EfficientNet v1. 

- You should choose a text encoder in [Sentence Transformers](https://sbert.net/) to generate text embeddings and sent them to RT1.

- Our implementation predicts multiple continuous actions (see above) instead of a single discrete action. We find our setting has better performance.

- **To get better performance, you should freeze the EfficientNet visual encoder in the 1st training stage, and unfreeze it in the 2nd stage.**
</details>

<details>
  <summary> Chi Cheng's Diffusion Policy (UNet / Transformer). </summary>

- [Original implementation]([https://github.com/google-research/robotics_transformer](https://github.com/real-stanford/diffusion_policy)).

- Our architecture is a copy of Chi Cheng's network. We test it in our pipeline and it has the same performance. Note that diffusion policy trains 2 resnet visual encoders for 2 camera views from scratch, so we never freeze the visual encoders.
	
- We also support predict actions in episilon / sample / v-space and other diffusion schedulers. The `DiffusionPolicy` wrapper can easily adapt to different network designs.
</details>

<details>
  <summary> Florence Policy developed on Microsoft's Florence2 VLM, which is trained with VQA, OCR, detection and segmentation tasks on 900M images. </summary>

- We develop the policy on the [pretrained model](https://huggingface.co/microsoft/Florence-2-base).

- Unlike [OpenVLA](https://github.com/openvla/openvla) and [RT2](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/), Florence2 is much smaller with 0.23B (Florence-2-base) or 0.7B (Florence-2-large) parameters.
	
- Unlike [OpenVLA](https://github.com/openvla/openvla) and [RT2](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/) which generate discrete actions, our Florence policy generates continuous actions with a linear action head or a diffusion transformer action head.

- The following figure illustrates the architecture of the Florence policy. We always freeze the DaViT visual encoder of Florence2, which is so good that unfreezing it does not improve the success rate.

<div align=center>
	<img width = '500' height ='350' src = "https://github.com/user-attachments/assets/54a236d1-492b-49fd-ab5f-59e53e88d259"/></div>
<div align="center">
<div align=center>
	Original Florence2 Network</div>
<div align="center">

<div align=center>
	<img width = '500' height ='350' src = "https://github.com/user-attachments/assets/cde63327-cc1c-4b12-8ef1-40f3ed21d26d"/></div>
<div align="center">
<div align=center>
	Florence policy with a linear action head</div>
<div align="center">

<div align=center>
	<img width = '550' height ='350' src = "https://github.com/user-attachments/assets/7ab7a387-e223-4dcd-947b-d3dadb03794f"/></div>
<div align="center">
<div align=center>
	Florence policy with a diffusion transformer action head</div>
<div align="center">

</details>

## Performance on Example Task

Square task with professional demos:

<div align=center><img src ="README_md_files/ee649200-4e85-11ef-b431-ef7e324e13ae.jpeg?v=1"/></div>

<div align="center">
	
| Policy | Success Rate | Checkpoint | Model Size | Failure Cases |
|--|--|--|--|--|
| RT-1 | 62% | [HuggingFace](https://huggingface.co/EDiRobotics/Mimictest_logs/blob/main/RT1_square/RT1.pth) | 23.8M | [HuggingFace](https://huggingface.co/EDiRobotics/Mimictest_logs/tree/main/RT1_square) | 
| Diffusion Policy (UNet) | 88.5% | [HuggingFace](https://huggingface.co/EDiRobotics/Mimictest_logs/blob/main/unet_square/unet.pth) | 329M | [HuggingFace](https://huggingface.co/EDiRobotics/Mimictest_logs/blob/main/unet_square) |
| Diffusion Policy (Transformer) | 90.5% | [HuggingFace](https://huggingface.co/EDiRobotics/Mimictest_logs/blob/main/DiffusionTransformer_square/DiffusionTransformer.pth) | 31.5M | [HuggingFace](https://huggingface.co/EDiRobotics/Mimictest_logs/blob/main/DiffusionTransformer_square) |
| Florence (linear head) | 88.5% | [HuggingFace]() | 270.8M | [HuggingFace]() |
| Florence (diffusion head) | 92.7% | link | xM | link |

</div>

*The success rate is measured with an average of 3 latest checkpoints. Each checkpoint is evaluated with 96 rollouts.
*For diffusion models, we save both the trained model and the exponential moving average (EMA) of the trained model in a checkpoint

<details>
  <summary> Failure analysis </summary>

- RT-1:
	- Failure to grasp an object after picking it up and the object falls: 1
 	- Pause before picking the object: 6
	- Pause before inserting object into the target: 2
 	- It thought the gripper picked up the object, but actually not: 3
  	- When inserting the object into target, the object gets stuck halfway through, and the policy doesn't know how to fix it: 1
- Diffusion Policy (UNet):
  	- Failure to grasp an object after picking it up and the object falls: 2
	- Pause before picking the object: 2
   	- It thought the gripper picked up the object, but actually not: 1
	- When inserting the object into target, the object gets stuck halfway through, and the policy doesn't know how to fix it: 3
 	- It successfylly inserts the object into target but suddenly lifts and throws the object away: 1
- Diffusion Policy (Transformer):
	- Pause before picking the object: 1 (In the third-person view, objects are obscured by the gripper)
  	- It thought the gripper picked up the object, but actually not: 1
	- Pause before inserting object into the target: 2
 - Florence (linear head):
	- c

</details>

## Installation

Other python versions may also work, like 3.8

```
conda create -n mimic python=3.9
conda activate mimic
https://github.com/EDiRobotics/mimictest
cd mimictest
bash setup.bash
```

You should also download dataset that contains `env_meta` [here](https://diffusion-policy.cs.columbia.edu/data/training/).

## Multi-GPU Train & Evaluation
1. You shall first run `accelerate config` to set environment parameters (number of GPUs, precision, etc). We recommend to use `bf16`.
2. Download and unzip this [dataset](https://diffusion-policy.cs.columbia.edu/data/training/).
3. Please check and modify the settings (e.g, train or eval, and the corresponding settings) in the scripts you want to run, under the `Script` directory. Each script represent a configuration of an algorithm.
4. Please then run
```
accelerate launch Script/<the script you choose>.py
```

## Possible Installation Problems

[](https://github.com/StarCycle/ParallelRobomimic#possible-problems)

```
ImportError: /opt/conda/envs/test/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1)
```

Please check [this link](https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin).
