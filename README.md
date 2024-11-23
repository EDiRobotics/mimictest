<div align=center><img width = '600' height ='350' src="https://github.com/user-attachments/assets/8d27de3f-bc5d-40b9-b1d1-2129b10a128f"/></div>


<div align="center">
	
# Mimictest

</div>

A simple testbed for robotics manipulation policies based on [robomimic](https://robomimic.github.io/). All policies are rewritten in a simple way. We may further expand it to the [robocasa](https://github.com/robocasa/robocasa) benchmark, which is also based on [robosuite](https://github.com/ARISE-Initiative/robosuite) simulator.

We also have policies trained and tested on the [CALVIN](https://github.com/mees/calvin) benchmark, e.g., [GR1-Training](https://github.com/EDiRobotics/GR1-Training) which is the current SOTA on the hardest ABC->D task of CALVIN.

<details>
  <summary> We also recommend other good frameworks / comunities for robotics policy learning. </summary>

- HuggingFace's [LeRobot](https://github.com/huggingface/lerobot), which currently have ACT, Diffusion Policy (only simple pusht task), TDMPC, and VQ-BeT. LeRobot has a nice robotics learning community on this [discord server](https://discord.com/invite/s3KuuzsPFb).

- [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) which implements multiple diffusion algorithms for imitation learning and reinforcement learning. Our implementation of diffusion algorithms is different from CleanDiffuser, but we thank the help of their team members.

- Dr. Mu Yao organizes a nice robitics learning community for Chinese researchers, see [DeepTimber website](https://gamma.app/public/DeepTimber-Robotics-Innovations-Community-A-Community-for-Multi-m-og0uv8mswl1a3q7?mode=doc) and [知乎](https://zhuanlan.zhihu.com/p/698664022).

</details>

**Please remember we build systems for you ヾ(^▽^*)).** Feel free to ask [@StarCycle](https://github.com/StarCycle) if you have any question!

## News
**[2024.11.1]** Update the performance of PushT environment.

**[2024.8.9]** Several updates below. And we are merging the base Florence policy to HuggingFace LeRobot.
- Add Florence policy with DiT diffusion action head from [MDT](https://github.com/intuitive-robots/mdt_policy) developed by [intuitive robot lab](https://github.com/intuitive-robots) at KIT. 
- Switch from tensorboard to wandb.
- Heavily optimize training speed of Florence-series models.
- Support compilation.

<details>
  <summary> Profiling result of compiled Florence MDT DiT policy </summary>

![图片](https://github.com/user-attachments/assets/0b7e1b60-eed0-495c-b00c-6a02ea9a0d43)

</details>

**[2024.7.30]** Add Florence policy with MLP action head & diffusion transformer action head (from Cheng Chi's Diffusion policy). Add RT-1 policy.  

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

- [Original implementation](https://github.com/real-stanford/diffusion_policy).

- Our architecture is a copy of Chi Cheng's network. We test it in our pipeline and it has the same performance. Note that diffusion policy trains 2 resnet visual encoders for 2 camera views from scratch, so we never freeze the visual encoders.
	
- We also support predict actions in episilon / sample / v-space and other diffusion schedulers. The `DiffusionPolicy` wrapper can easily adapt to different network designs.
</details>

<details>
  <summary> Florence Policy developed on Microsoft's Florence2 VLM, which is trained with VQA, OCR, detection and segmentation tasks on 900M images. </summary>

- We develop the policy on the [pretrained model](https://huggingface.co/microsoft/Florence-2-base).

- Unlike [OpenVLA](https://github.com/openvla/openvla) and [RT2](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/), Florence2 is much smaller with 0.23B (Florence-2-base) or 0.7B (Florence-2-large) parameters.
	
- Unlike [OpenVLA](https://github.com/openvla/openvla) and [RT2](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/) which generate discrete actions, our Florence policy generates continuous actions with a linear action head / a diffusion transformer action head from Cheng Chi's Diffusion Policy / a DiT action head from MDT policy.

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

**Square task with professional demos:**

<div align=center><img src ="README_md_files/ee649200-4e85-11ef-b431-ef7e324e13ae.jpeg?v=1"/></div>

<div align="center">
	
| Policy | Success Rate | Model Size |
|--|--|--|
| RT-1 | 62% | 23.8M | 
| Diffusion Policy (UNet) | 88.5% | 329M |
| Diffusion Policy (Transformer) | 90.5% | 31.5M |
| Florence (linear head) | 88.5% | 270.8M | 
| Florence (diffusion head - MDT DiT) | 93.75% | 322.79M |

</div>

*The success rate is measured with an average of 3 latest checkpoints. Each checkpoint is evaluated with 96 rollouts.
*For diffusion models, we save both the trained model and the exponential moving average (EMA) of the trained model in a checkpoint

**PushT task:**

<div align=center><img src ="https://github.com/user-attachments/assets/21b477a1-3f0d-440a-9a56-1224cc0136fb"/>

| Policy | Success Rate | Model Size |
|--|--|--|
| RT-1 | 52% | 23.8M |
| Diffusion Policy (UNet) | 64.5% | 76M |
| Florence (linear head) | 53% | 270.8M |
| Florence (diffusion head - MDT DiT) | 64% | 322.79M |

</div>

*Each checkpoint is evaluated with 96 rollouts. 

*A success in the PushT environment requires a final IoU > 95% (which is difficult to locate under low resolution). If you raise the resolution or reduce the threshold, the succes rate will be much higher.

## Installation

You can use [mirror sites of Github](https://github.com/runningcheese/MirrorSite) to avoid the connection problem in some regions. With different simulators, it's recommended to use different python versions, which will be mentioned below.
```
conda create -n mimic python=3.x
conda activate mimic
apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common net-tools unzip vim virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf cmake
git clone https://github.com/EDiRobotics/mimictest
cd mimictest
pip install -e .
```

Now, depending on the environment and model you want, Please perform the following steps.

<details>
  <summary> For Robomimic experiments. </summary>

The recommended python version is 3.9. You need to install `robomimic` and `robosuite` via:
```
pip install robosuite@https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz
pip install robomimic
```

Recent robosuite has turned to the DeepMind's Mujoco 3 backend but we are still using the old version with Mujoco 2.1. This is because the dataset is recorded in Mujoco 2.1, which has slighlyly dynamics difference with Mujoco 3. 

You should also download dataset that contains `robomimic_image.zip` or `robomimic_lowdim.zip` from the [official link](https://diffusion-policy.cs.columbia.edu/data/training/) or [HuggingFace](https://huggingface.co/datasets/EDiRobotics/mimictest_data). In this example, I use the tool of [HF-Mirror](https://hf-mirror.com/). You can set the environment variable `export HF_ENDPOINT=https://hf-mirror.com` to avoid the connection problem in some regions.

```
apt install git-lfs aria2
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
./hfd.sh EDiRobotics/mimictest_data --dataset --tool aria2c -x 9
```

If you only want to download a subset of the data, e.g., the square task with image input:

```
./hfd.sh EDiRobotics/mimictest_data --dataset --tool aria2c -x 9 --include robomimic_image/square.zip
```

</details>

<details>
  <summary> For the PushT experiment. </summary>

The recommended python version is 3.19. You can install the environment via 

```
pip install gym-pusht
```

Then you can download the PushT dataset from the [official link](https://diffusion-policy.cs.columbia.edu/data/training/).

</details>

<details>
  <summary> For Florence-based models. </summary>

To use florence-based models, you should download one of it from HuggingFace, for example:
```
./hfd.sh microsoft/Florence-2-base --model --tool aria2c -x 9
```

And then set `model_path` in the script, for example:
```
# in Script/FlorenceImage.py
model_path = "/path/to/downloaded/florence/folder"
```

You need to install florence-specific dependencies, e.g., flash-attention. You can achieve it with:
```
pip install -e .[florence]
```

</details>

## Multi-GPU Train & Evaluation
1. You shall first run `accelerate config` to set environment parameters (number of GPUs, precision, etc). We recommend to use `bf16`.
2. Download and unzip the dataset mentioned above.
3. Please check and modify the settings (e.g, train or eval, and the corresponding settings) in the scripts you want to run, under the `Script` directory. **Each script represents a configuration of an algorithm.**
4. Please then run
```
accelerate launch Script/<the script you choose>.py
```

## Possible Installation Problems

1. `GLIBCXX_3.4.30' not found
```
ImportError: /opt/conda/envs/test/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1)
```
You can try `conda install -c conda-forge gcc=12.1` which is a magical command that automatically install some dependencies. 

Also check [this link](https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin).

2. Spend too much time compiling flash-attn
   
You can download a pre-build wheel from [official release](https://github.com/Dao-AILab/flash-attention/releases), instead of building a wheel by yourself. For example (you should choose a wheel depending on your system):
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiTRUE-cp39-cp39-linux_x86_64.whl
pip install flash_attn-2.6.3+cu118torch2.4cxx11abiTRUE-cp39-cp39-linux_x86_64.whl
```

When installing pytorch, make sure the torch cuda version and your cuda driver version are the same (e.g., 11.8).

3. Cannot initialize a EGL device display
```
Cannot initialize a EGL device display. This likely means that your EGL driver does not support the PLATFORM_DEVICE extension, which is required for creating a headless rendering context.
```
You can try `conda install -c conda-forge gcc=12.1`.

4. fatal error: GL/osmesa.h: No such file or directory
```
 /tmp/pip-install-rsxccpmh/mujoco-py/mujoco_py/gl/osmesashim.c:1:23: fatal error: GL/osmesa.h: No such file or directory
    compilation terminated.
    error: command 'gcc' failed with exit status 1
```
You can try `conda install -c conda-forge mesalib glew glfw` or check this [link](https://github.com/ethz-asl/reinmav-gym/issues/35).

5. cannot find -lGL
```
 /home/ubuntu/anaconda3/compiler_compat/ld: cannot find -lGL
  collect2: error: ld returned 1 exit status
  error: command 'gcc' failed with exit status 1
```
You can try `conda install -c conda-forge mesa-libgl-devel-cos7-x86_64` or check this [link](https://stackoverflow.com/questions/59016606/ld-cant-find-lgl-error-during-installation).

6. `SystemError: initialization of _internal failed without raising an exception`.

You can simply `pip -U numba` or this [link](https://stackoverflow.com/questions/74947992/how-to-remove-the-error-systemerror-initialization-of-internal-failed-without).

7. `ImportError: libGL.so.1: cannot open shared object file`
```
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```
Or check this [link](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo).

8. `failed to EGL with glad`

The core problem seems to be lack of `libEGL.so.1`. You may try `apt-get update && apt-get install libegl1`. If you find other packages not installed during installing `libegl1`, please install them.

9. `No such file or directory: ‘patchelf’ `

```
apt install patchelf
```

10. `from numba.np.ufunc import _internal SystemError: initialization of _internal failed without raising an exception`

```
pip install -U numba
```
