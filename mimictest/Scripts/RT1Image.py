import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from mimictest.Utils.AccelerateFix import AsyncStep
from mimictest.Utils.PreProcess import PreProcess
from mimictest.Utils.RobomimicDataset import CustomMimicDataset, DataPrefetcher
from mimictest.Utils.ComputeLimit import ComputeLimit
from mimictest.Wrappers.SimplePolicy import SimplePolicy
from mimictest.Nets.RT1 import RT1
from mimictest.Simulation.ParallelEnv import ParallelMimic
from mimictest.Train import train
from mimictest.Evaluation import Evaluation

if __name__ == '__main__':
    # Script-specific settings (general settings stored in 
    mode = 'train' # 'train' or 'eval'

    # Saving path
    save_path = './Save/'

    # Dataset
    abs_mode = True # relative EE action space or absolute EE action space
    if abs_mode:
        file_name = 'image_abs.hdf5'
    else:
        file_name = 'image.hdf5'
    dataset_path = f'/root/dataDisk/robomimic/datasets/square/ph/' + file_name
    bs_per_gpu = 432
    desired_rgb_shape = 84
    crop_shape = 76
    workers_per_gpu = 8
    cache_ratio = 2

    # Space
    num_actions = 7
    num_actions_6d = 10
    obs_horizon = 2
    chunk_size = 16
    limits = ComputeLimit(dataset_path, abs_mode)

    # Network
    efficientnet_version = "efficientnet_v2_s"
    FiLM_cond_channel = 1 # Since we dont use it 
    depth = 8
    vision_token_dim = 512
    ff_dim = 128
    n_heads = 2
    head_dim = 64
    max_T = 128
    token_learner_num_output_tokens = 8
    drop_prob = 0.1
    freeze_vision_tower = False

    # Training
    num_training_epochs = 5000
    save_interval = 50 
    load_epoch_id = 0
    gradient_accumulation_steps = 1
    lr_max = 2e-5
    warmup_steps = 5
    weight_decay = 1e-4
    print_interval = 60

    # Testing (num_envs*num_eval_ep*num_GPU epochs)
    num_envs = 16
    num_eval_ep = 6
    test_chunk_size = 1
    max_test_ep_len = 400
    smooth_factor = 0.01

    # Preparation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # kwargs_handlers=[kwargs],
    )
    device = acc.device
    preprocessor = PreProcess(
        desired_rgb_shape,
        crop_shape,
        limits['low_dim_max'],
        limits['low_dim_min'],
        limits['action_max'],
        limits['action_min'],
        abs_mode,
        device,
    )
    dataset = CustomMimicDataset(dataset_path, obs_horizon, chunk_size, start_ratio=0, end_ratio=1)
    loader = DataLoader(
        dataset=dataset,
        sampler=None, 
        batch_size=bs_per_gpu,
        shuffle=True,
        num_workers=workers_per_gpu,
        drop_last=True,     
    )
    net = RT1(
        efficientnet_version=efficientnet_version,
        FiLM_cond_channel=FiLM_cond_channel,
        lowdim_obs_num=len(limits['low_dim_max']),
        num_actions=num_actions_6d,
        chunk_size=chunk_size,
        depth=depth,
        vision_token_dim=vision_token_dim,
        ff_dim=ff_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        max_T=max_T,
        token_learner_num_output_tokens=token_learner_num_output_tokens,
        drop_prob=drop_prob,
        freeze_vision_tower=freeze_vision_tower,
    ).to(device)
    policy = SimplePolicy(
        net=net,
        loss_func=torch.nn.functional.l1_loss,
    )
    if os.path.isfile(save_path+f'policy_{load_epoch_id}.pth'):
        policy.load_pretrained(acc, save_path+f'policy_{load_epoch_id}.pth')
    if os.path.isfile(save_path+'step.json'):
        with open(save_path+'step.json', 'r') as json_file:
            step = json.load(open(save_path+'step.json'))
    else:
        step = 0
    optimizer = torch.optim.AdamW(policy.net.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    policy.net, optimizer, loader = acc.prepare(
        policy.net, 
        optimizer, 
        loader, 
        device_placement=[True, True, False],
    )
    optimizer.step = AsyncStep
    prefetcher = DataPrefetcher(loader, device)
    envs = ParallelMimic(dataset_path, num_envs, abs_mode)
    eva = Evaluation(
        envs, 
        num_envs, 
        preprocessor, 
        obs_horizon,
        test_chunk_size, 
        num_actions, 
        smooth_factor, 
        acc.device,
    )
    writer = SummaryWriter(save_path + 'logs')

    if mode == 'train':
        train(
            acc=acc, 
            prefetcher=prefetcher, 
            preprocessor=preprocessor,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            num_training_epochs=num_training_epochs,
            eva=eva,
            num_eval_ep=num_eval_ep, 
            max_test_ep_len=max_test_ep_len,
            device=device,
            writer=writer,
            save_path=save_path,
            load_epoch_id=load_epoch_id,
            save_interval=save_interval,
            step=step,
            print_interval=print_interval,
            bs_per_gpu=bs_per_gpu,
            do_profile = False,
        )
    elif mode == 'eval':
        avg_reward  = torch.tensor(eva.evaluate_on_env(policy, num_eval_ep, max_test_ep_len, save_path=save_path, record_video=True)).to(device)
        avg_reward = acc.gather_for_metrics(avg_reward).mean(dim=0)
        acc.print(f'chunk size {test_chunk_size}, success rate {avg_reward}')
