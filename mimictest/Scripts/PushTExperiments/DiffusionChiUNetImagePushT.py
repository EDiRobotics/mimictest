import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from mimictest.Utils.AccelerateFix import AsyncStep
from mimictest.Utils.PreProcess import PreProcess
from mimictest.Datasets.PushTDataset import PushTImageDataset
from mimictest.Datasets.DataPrefetcher import DataPrefetcher
from mimictest.Simulation.ParallelPushT import ParallelPushT
from mimictest.Wrappers.DiffusionPolicy import DiffusionPolicy
from mimictest.Nets.Chi_UNet1D import Chi_UNet1D
from mimictest.Train import train
from mimictest.Evaluation import Evaluation

if __name__ == '__main__':
    # Script-specific settings (general settings stored in 
    mode = 'train' # or 'eval'

    # Saving path
    save_path = Path('./Save/')
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    abs_mode = True # relative EE action space or absolute EE action space
    file_name = 'pusht_cchi_v7_replay.zarr'
    dataset_path = Path('/root/dataDisk/pusht/') / file_name
    bs_per_gpu = 800
    desired_rgb_shape = 96
    crop_shape = 96
    workers_per_gpu = 8
    cache_ratio = 2

    # Space
    camera_num = 2
    num_actions = 2
    obs_horizon = 1
    chunk_size = 16

    # Network
    resnet_name = 'resnet18'
    diffusion_step_embed_dim = 128
    down_dims = [512, 1024, 2048]
    kernel_size = 5
    n_groups = 8
    do_compile = False
    do_profile = False

    # Diffusion
    diffuser_train_steps = 100
    diffuser_infer_steps = 100
    diffuser_solver = "ddpm"
    beta_schedule = "squaredcos_cap_v2"
    prediction_type = 'epsilon'
    clip_sample = True
    ema_interval = 10
    loss_func = torch.nn.functional.mse_loss

    # Training
    num_training_epochs = 1000
    save_interval = 50 
    load_epoch_id = 0
    gradient_accumulation_steps = 1
    lr_max = 1e-4
    warmup_steps = 5
    weight_decay = 1e-4
    max_grad_norm = 10
    print_interval = 29
    do_watch_parameters = False
    record_video = True

    # Testing (num_envs*num_eval_ep*num_GPU epochs)
    num_envs = 16
    num_eval_ep = 1
    test_chunk_size = 8
    max_test_ep_len = 200

    # Preparation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        # kwargs_handlers=[kwargs],
    )
    device = acc.device
    dataset = PushTImageDataset(dataset_path, chunk_size, obs_horizon, chunk_size)
    preprocessor = PreProcess(
        desired_rgb_shape=desired_rgb_shape,
        crop_shape=crop_shape,
        low_dim_max=torch.from_numpy(dataset.stats['agent_pos']['max']),
        low_dim_min=torch.from_numpy(dataset.stats['agent_pos']['min']),
        action_max=torch.from_numpy(dataset.stats['action']['max']),
        action_min=torch.from_numpy(dataset.stats['action']['min']),
        enable_6d_rot=False,
        abs_mode=abs_mode,
        device=device,
    )
    envs = ParallelPushT(num_envs)
    eva = Evaluation(
        envs, 
        num_envs, 
        preprocessor, 
        obs_horizon,
        test_chunk_size, 
        num_actions, 
        save_path,
        device,
    )
    loader = DataLoader(
        dataset=dataset,
        sampler=None, 
        batch_size=bs_per_gpu,
        shuffle=True,
        num_workers=workers_per_gpu,
        drop_last=True,     
    )
    unet = Chi_UNet1D(
        camera_num=camera_num,
        obs_horizon=obs_horizon,
        lowdim_obs_dim=len(dataset.stats['agent_pos']['max']),
        num_actions=num_actions,
        resnet_name=resnet_name,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
    ).to(device)
    policy = DiffusionPolicy(
        net=unet,
        loss_func=loss_func,
        do_compile=do_compile,
        num_actions=num_actions,
        chunk_size=chunk_size,
        scheduler_name=diffuser_solver,
        num_train_steps=diffuser_train_steps,
        num_infer_steps=diffuser_infer_steps,
        ema_interval=ema_interval,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )
    policy.load_pretrained(acc, save_path, load_epoch_id)
    policy.load_wandb(acc, save_path, do_watch_parameters, save_interval)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
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
            save_path=save_path,
            load_epoch_id=load_epoch_id,
            save_interval=save_interval,
            print_interval=print_interval,
            bs_per_gpu=bs_per_gpu,
            max_grad_norm=max_grad_norm,
            record_video=record_video,
            do_profile=do_profile,
        )
    elif mode == 'eval':
        avg_reward = torch.tensor(eva.evaluate_on_env(
            acc=acc, 
            policy=policy, 
            epoch=0,
            num_eval_ep=num_eval_ep, 
            max_test_ep_len=max_test_ep_len, 
            record_video=True)
        ).to(device)
        avg_reward = acc.gather_for_metrics(avg_reward).mean(dim=0)
        acc.print(f'chunk size {test_chunk_size}, success rate {avg_reward}')