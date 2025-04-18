import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from mimictest.Utils.AccelerateFix import AsyncStep
from mimictest.Utils.PreProcess import PreProcess
from mimictest.Datasets.RobomimicDataset import CustomMimicDataset, ComputeLimit
from mimictest.Datasets.DataPrefetcher import DataPrefetcher
from mimictest.Wrappers.DiffusionPolicy import DiffusionPolicy
from mimictest.Nets.Chi_Transformer import Chi_Transformer
from mimictest.Simulation.ParallelMimic import ParallelMimic
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
    if abs_mode:
        file_name = 'image_abs.hdf5'
    else:
        file_name = 'image.hdf5'
    dataset_path = Path('/root/dataDisk/square/ph/') / file_name
    bs_per_gpu = 1280
    desired_rgb_shape = 84
    crop_shape = 76
    workers_per_gpu = 8
    cache_ratio = 2

    # Space
    camera_num = 2
    num_actions = 7
    num_actions_6d = 10
    obs_horizon = 2
    chunk_size = 16
    limits = ComputeLimit(dataset_path, abs_mode)

    # Network
    resnet_name = 'resnet18'
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
    load_batch_id = 0
    gradient_accumulation_steps = 1
    lr_max = 3e-4
    warmup_steps = 5
    weight_decay = 1e-4
    max_grad_norm = 10
    print_interval = 22
    eval_interval = print_interval * 20
    use_wandb = False
    wandb_name = "robomimic"
    do_watch_parameters = False
    record_video = False

    # Testing (num_envs*num_eval_ep*num_GPU epochs)
    num_envs = 16
    num_eval_ep = 6
    action_horizon = [0, 8]
    max_test_ep_len = 400

    # Preparation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        # kwargs_handlers=[kwargs],
    )
    device = acc.device
    preprocessor = PreProcess(
        desired_rgb_shape=desired_rgb_shape,
        crop_shape=crop_shape,
        low_dim_max=limits['low_dim_max'],
        low_dim_min=limits['low_dim_min'],
        action_max=limits['action_max'],
        action_min=limits['action_min'],
        enable_6d_rot=True,
        abs_mode=abs_mode,
        device=device,
    )
    envs = ParallelMimic(dataset_path, num_envs, abs_mode)
    eva = Evaluation(
        envs, 
        num_envs, 
        preprocessor, 
        obs_horizon,
        action_horizon, 
        num_actions, 
        save_path,
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
    transformer = Chi_Transformer(
        camera_num=camera_num,
        obs_horizon=obs_horizon,
        lowdim_obs_dim=len(limits['low_dim_max']),
        num_actions=num_actions_6d,
        max_T=max_T,
        resnet_name=resnet_name,
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
        net=transformer,
        loss_func=loss_func,
        do_compile=do_compile,
        num_actions=num_actions_6d,
        chunk_size=chunk_size,
        scheduler_name=diffuser_solver,
        num_train_steps=diffuser_train_steps,
        num_infer_steps=diffuser_infer_steps,
        ema_interval=ema_interval,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )
    policy.load_pretrained(acc, save_path, load_batch_id) 
    if use_wandb:
        policy.load_wandb(acc, wandb_name, save_path, do_watch_parameters, save_interval)
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
            load_batch_id=load_batch_id,
            save_interval=save_interval,
            print_interval=print_interval,
            eval_interval=eval_interval,
            bs_per_gpu=bs_per_gpu,
            max_grad_norm=max_grad_norm,
            use_wandb=use_wandb,
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
        acc.print(f'action hoziron {action_horizon}, success rate {avg_reward}')
