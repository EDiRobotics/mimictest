# mimictest
A simple testbed for robotics manipulation policies based on [robomimic](https://robomimic.github.io/). It uses this [dataset](https://diffusion-policy.cs.columbia.edu/data/training/). All policies are rewritten in a simple way.

### News
**[2024.7.15]** Initial release which only contains UNet version of Diffusion Policy.

### Supported Policies
- Google's [RT1](https://github.com/google-research/robotics_transformer).
- Chi Cheng's [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (UNet / Transformer).
- Florence Policy, under development.
- Magvit2 Policy, under development.

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
