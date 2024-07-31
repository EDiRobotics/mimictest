import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from mimictest.Utils.AccelerateFix import AsyncStep
from mimictest.Utils.PreProcess import PreProcess
from mimictest.Utils.RobomimicDataset import CustomMimicDataset, DataPrefetcher
from mimictest.Utils.ComputeLimit import ComputeLimit
from mimictest.Wrappers.DiffusionPolicy import DiffusionPolicy
from mimictest.Nets.FlorenceOctoNet import FlorenceOctoNet
from mimictest.Simulation.ParallelEnv import ParallelMimic
from mimictest.Train import train
from mimictest.Evaluation import Evaluation

if __name__ == '__main__':
    # Script-specific settings (general settings stored in 
    mode = 'train' # or 'eval'

    # Saving path
    save_path = './Save/'

    # Dataset
    abs_mode = True # relative EE action space or absolute EE action space
    if abs_mode:
        file_name = 'image_abs.hdf5'
    else:
        file_name = 'image.hdf5'
    dataset_path = f'/root/dataDisk/robomimic/datasets/square/ph/' + file_name
    bs_per_gpu = 360
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
    model_path = "/root/dataDisk/Florence-2-base"
    freeze_vision_tower = True
    max_T = chunk_size
    n_layer = 8
    n_cond_layers = 0  # >0: use transformer encoder for cond, otherwise use MLP
    n_head = 4
    n_emb = 256
    p_drop_emb = 0.0
    p_drop_attn = 0.3
    causal_attn = True
    obs_as_cond = True
    time_as_cond = True # if false, use BERT like encoder only arch, time as input

    # Diffusion
    diffuser_train_steps = 100
    diffuser_infer_steps = 100
    diffuser_solver = "ddpm"
    beta_schedule = "squaredcos_cap_v2"
    prediction_type = 'epsilon'
    clip_sample = True
    loss_func = torch.nn.functional.mse_loss

    # Training
    num_training_epochs = 5000
    save_interval = 200 
    load_epoch_id = 0
    gradient_accumulation_steps = 1
    lr_max = 1e-4
    warmup_steps = 5
    weight_decay = 1e-4
    print_interval = 79

    # Testing (num_envs*num_eval_ep*num_GPU epochs)
    num_envs = 16
    num_eval_ep = 6
    test_chunk_size = 8
    max_test_ep_len = 50
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
    net = FlorenceOctoNet(
        path=model_path,
        freeze_vision_tower=freeze_vision_tower,
        lowdim_obs_dim=len(limits['low_dim_max']),
        num_actions=num_actions_6d,
        max_T=max_T,
        n_layer=n_layer,
        n_head=n_head,
        n_emb=n_emb,
        p_drop_emb=p_drop_emb,
        p_drop_attn=p_drop_attn,
        causal_attn=causal_attn,
        time_as_cond=time_as_cond,
        obs_as_cond=obs_as_cond,
        n_cond_layers=n_cond_layers,
    ).to(device)
    policy = DiffusionPolicy(
        net=net,
        num_actions=num_actions_6d,
        chunk_size=chunk_size,
        scheduler_name=diffuser_solver,
        num_train_steps=diffuser_train_steps,
        num_infer_steps=diffuser_infer_steps,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
        loss_func=loss_func,
    )
    if os.path.isfile(save_path+f'policy_{load_epoch_id}.pth'):
        policy.load_pretrained(acc, save_path+f'policy_{load_epoch_id}.pth')
    if os.path.isfile(save_path+'step.json'):
        with open(save_path+'step.json', 'r') as json_file:
            step = json.load(open(save_path+'step.json'))
    else:
        step = 0
    optimizer = torch.optim.AdamW(policy.net.parameters(), lr=lr_max, weight_decay=weight_decay, fused=False) # TODO
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    policy.net, policy.ema_net, optimizer, loader = acc.prepare(
        policy.net, 
        policy.ema_net, 
        optimizer, 
        loader, 
        device_placement=[True, True, True, False],
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