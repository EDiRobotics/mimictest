<div align=center><img width = '500' height ='250' src="https://github.com/user-attachments/assets/8d27de3f-bc5d-40b9-b1d1-2129b10a128f"/></div>

# Mimictest
A simple testbed for robotics manipulation policies based on [robomimic](https://robomimic.github.io/). It uses Diffusion Policy's [dataset](https://diffusion-policy.cs.columbia.edu/data/training/). All policies are rewritten in a simple way.

### News
**[2024.7.30]** Add Florence policy with MLP action head & diffusion action head. Add RT-1 policy.  

**[2024.7.16]** Add transformer version of Diffusion Policy.

**[2024.7.15]** Initial release which only contains UNet version of Diffusion Policy.

### Features

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

### Supported Policies

We implement the following algorithms:

<details>
  <summary> Google's RT1. </summary>

- [Original implementation](https://github.com/google-research/robotics_transformer).

- Our implementation supports EfficientNet v1/v2 and you can directly load pretrained weights by torchvision API. Google's implementation only supports EfficientNet v1. 

- You should choose a text encoder in [Sentence Transformers](https://sbert.net/) to generate text embeddings and sent them to RT1.

- Our implementation predicts multiple continuous actions (see above) instead of a single discrete action. We find our setting has better performance.
</details>

<details>
  <summary> Chi Cheng's Diffusion Policy (UNet / Transformer). </summary>

- [Original implementation]([https://github.com/google-research/robotics_transformer](https://github.com/real-stanford/diffusion_policy)).

- Our architecture is a copy of Chi Cheng's network. We test it in our pipeline and it has the same performance.
	
- We also support predict actions in episilon / sample / v-space and other diffusion schedulers. The `DiffusionPolicy` wrapper can easily adapt to different network designs.
</details>

<details>
  <summary> Florence Policy developed on Microsoft's Florence2 VLM, which is trained with VQA, OCR, detection and segmentation tasks on 900M images. </summary>

- We develop the policy on the [pretrained model](https://huggingface.co/microsoft/Florence-2-base).

- Unlike [OpenVLA](https://github.com/openvla/openvla) and [RT2](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/), Florence2 is much smaller with 0.23B (Florence-2-base) or 0.7B (Florence-2-large) parameters.
	
- Unlike [OpenVLA](https://github.com/openvla/openvla) and [RT2](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/) which generate discrete actions, our Florence policy generates continuous actions with an MLP action head or a diffusion transformer action head.

- The following figure illustrates the architecture of the Florence policy:
</details>

### Performance on Example Task

Square task with professional demos:

<div align=center><img src ="README_md_files/ee649200-4e85-11ef-b431-ef7e324e13ae.jpeg?v=1"/></div>

<div align="center">
	
| Policy | Success Rate | Checkpoint | Failure Cases |
|--|--|--|--|
| RT-1 | 62% | link | link | 
| Diffusion Policy (UNet) | 88.5% | link | link |
| Diffusion Policy (Transformer) | 90.5% | link | link |
| Florence (MLP head) | 88.5% | link | link |
| Florence (diffusion head) | 92.7% | link | link |

</div>

*The success rate is measured with an average of 3 latest checkpoints. Each checkpoint is evaluated with 96 rollouts.

### Installation

Other python versions may also work, like 3.8

```
conda create -n mimic python=3.9
conda activate mimic
https://github.com/EDiRobotics/mimictest
cd mimictest
bash setup.bash
```

You should also download dataset that contains `env_meta` [here](https://diffusion-policy.cs.columbia.edu/data/training/).

### Multi-GPU Train & Evaluation
1. You shall first run `accelerate config` to set environment parameters (number of GPUs, precision, etc). We recommend to use `bf16`.
2. Download and unzip this [dataset](https://diffusion-policy.cs.columbia.edu/data/training/).
3. Please check and modify the settings (e.g, train or eval, and the corresponding settings) in the scripts you want to run, under the `Script` directory. Each script represent a configuration of an algorithm.
4. Please then run
```
accelerate launch Script/<the script you choose>.py
```

### Possible Installation Problems

[](https://github.com/StarCycle/ParallelRobomimic#possible-problems)

```
ImportError: /opt/conda/envs/test/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1)
```

Please check [this link](https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin).
