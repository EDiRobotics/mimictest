# mimictest
A simple testbed for robotics manipulation policies based on [robomimic](https://robomimic.github.io/). It uses this [dataset](https://diffusion-policy.cs.columbia.edu/data/training/). All policies are rewritten in a simple way.

### News
**[2024.7.30]** Add Florence policy with MLP action head & diffusion action head. Add RT-1 policy.  

**[2024.7.16]** Add transformer version of Diffusion Policy.

**[2024.7.15]** Initial release which only contains UNet version of Diffusion Policy.

### Unified State and Action Space
- All policies share the same data pre-processing pipeline and predict actions in 3D Cartesian translation  + [6D rotation](https://zhouyisjtu.github.io/project_rotation/rotation.html) + gripper open/close. The 3D translation can be relative to current gripper position (`abs_mode=False`) or the world coordinate (`abs_mode=True`).
- They perceive `obs_horizon` historical observations, generate `chunk_size` future actions, and execute `test_chunk_size` predicted actions. An example with `obs_horizon=3, chunk_size=4, test_chunk_size=2`:
```
Policy sees: 		|o|o|o|
Policy predicts: 	| | |a|a|a|a|
Policy executes:	| | |a|a|
```
- They use image input from both static and wrist cameras.

### Supported Policies
- Google's [RT1](https://github.com/google-research/robotics_transformer).

	- Our implementation supports EfficientNet v1/v2 and you can directly load pretrained weights by torchvision API. Google's implementation only supports EfficientNet v1. 
	- You should choose a text encoder in [Sentence Transformers](https://sbert.net/) to generate text embeddings and sent them to RT1.
	-   Our implementation predicts multiple continuous actions (see above) instead of a single discrete action. We find our setting has better performance.

- Chi Cheng's [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (UNet / Transformer).
	- The architecture is the same as Chi Cheng's implementation. We test ours and it has the same performance.
	- We also support predict actions in episilon / sample / v-space and other diffusion schedulers. The `DiffusionPolicy` wrapper can easily adapt to different network designs.
- Florence Policy developed on Microsoft's [Florence2](https://huggingface.co/microsoft/Florence-2-base) VLM, which is trained with VQA, OCR, detection and segmentation tasks on 900M images.
	- Unlike [OpenVLA](https://github.com/openvla/openvla) and [RT2](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/), Florence2 is much smaller with 0.23B (Florence-2-base) or 0.7B (Florence-2-large) parameters.
	- Unlike [OpenVLA](https://github.com/openvla/openvla) and [RT2](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/) which generate discrete actions, our Florence policy generates continuous actions with an MLP action head or a diffusion transformer action head. 

### Performance on Example Task

Square task with professional demos:

![](README_md_files/ee649200-4e85-11ef-b431-ef7e324e13ae.jpeg?v=1&type=image)
 
| Policy | Success Rate | Checkpoint | Failure Cases |
|--|--|--|--|
| RT-1 | 62% | link | link | 
| Diffusion Policy (UNet) | 88.5% | link | link |
| Diffusion Policy (Transformer) | 90.5% | link | link |
| Florence (MLP head) | 88.5% | link | link |
| Florence (diffusion head) | 92.7% | link | link |

*The success rate is measured with an average of 3 latest checkpoints. Each checkpoint is evaluated with 96 rollouts.

### Installation

[](https://github.com/StarCycle/ParallelRobomimic#installation)

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
2. Please check and modify the settings (e.g, train or eval, and the corresponding settings) in the scripts you want to run, under the `Script` directory. Each script represent a configuration of an algorithm.
3. Please then run
```
accelerate launch Script/<the script you choose>.py
```

### Possible Installation Problems

[](https://github.com/StarCycle/ParallelRobomimic#possible-problems)

```
ImportError: /opt/conda/envs/test/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1)

```

Please check [this link](https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin).
