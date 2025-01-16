import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from mimictest.Utils.AccelerateFix import AsyncStep
from mimictest.Utils.PreProcess import PreProcess
from mimictest.Datasets.PushTDataset import PushTImageDataset
from mimictest.Datasets.DataPrefetcher import DataPrefetcher
from mimictest.Wrappers.DiffusionPolicy import DiffusionPolicy
from mimictest.Nets.FlorenceMDTNet import FlorenceMDTNet
from mimictest.Simulation.ParallelPushT import ParallelPushT
from mimictest.Train import train
from mimictest.Evaluation import Evaluation

if __name__ == '__main__':
    # Script-specific settings 
    mode = 'eval' # or 'eval'

    # Saving path
    save_path = Path('./Save/')
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    abs_mode = True # relative EE action space or absolute EE action space
    file_name = 'pusht_cchi_v7_replay.zarr'
    dataset_path = Path('/root/autodl-tmp/pusht/') / file_name
    bs_per_gpu = 64
    workers_per_gpu = 12
    cache_ratio = 2

    # Space
    num_actions = 2
    lowdim_obs_dim = 2
    obs_horizon = 2
    chunk_size = 16
    process_configs = {
        'rgb': {
            'rgb_shape': (96, 96),
            'crop_shape': (84, 84),
            'max': torch.tensor(1.0),
            'min': torch.tensor(0.0),
        },
        'low_dim': {
            'max': None, # to be filled
            'min': None,
        },
        'action': {
            'max': None, # to be filled
            'min': None,
        },
    }

    # Network
    model_path = Path("microsoft/Florence-2-base")
    freeze_vision_tower = False
    freeze_florence = False
    num_action_query = 64
    max_T = 128
    ff_dim = 128
    n_heads = 2
    attn_pdrop = 0.3
    resid_pdrop = 0.1
    mlp_pdrop = 0.05
    n_layers = 4
    do_compile = False
    do_profile = False

    # Diffusion
    diffuser_train_steps = 10
    diffuser_infer_steps = 10
    diffuser_solver = "ddim"
    beta_schedule = "squaredcos_cap_v2"
    prediction_type = 'epsilon'
    clip_sample = True
    ema_interval = 10

    # Training
    num_training_epochs = 400
    save_interval = 50
    load_epoch_id = 0
    gradient_accumulation_steps = 1
    lr_max = 1e-4
    warmup_steps = 5
    weight_decay = 1e-4
    max_grad_norm = 10
    print_interval = 360
    use_wandb = False
    do_watch_parameters = False
    record_video = False
    loss_configs = {
        'action': {
            'loss_func': torch.nn.functional.l1_loss,
            'type': 'diffusion',
            'weight': 1.0,
            'shape': (chunk_size, num_actions),
        },
    }

    # Testing (num_envs*num_eval_ep*num_GPU epochs)
    num_envs = 16
    num_eval_ep = 6
    action_horizon = [1, 9]
    max_test_ep_len = 300

    # Preparation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        kwargs_handlers=[kwargs],
    )
    device = acc.device
    dataset = PushTImageDataset(
        dataset_path, 
        chunk_size, 
        obs_horizon, 
        action_horizon[1] - action_horizon[0],
    )
    process_configs['low_dim']['max'] = torch.from_numpy(dataset.stats['agent_pos']['max'])
    process_configs['low_dim']['min'] = torch.from_numpy(dataset.stats['agent_pos']['min'])
    process_configs['action']['max'] = torch.from_numpy(dataset.stats['action']['max'])
    process_configs['action']['min'] = torch.from_numpy(dataset.stats['action']['min'])
    preprocessor = PreProcess(
        process_configs=process_configs,
        device=device,
    )
    envs = ParallelPushT(num_envs, max_test_ep_len)
    eva = Evaluation(
        envs, 
        num_envs, 
        preprocessor, 
        obs_horizon,
        action_horizon,
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
    net = FlorenceMDTNet(
        path=model_path,
        freeze_vision_tower=freeze_vision_tower,
        freeze_florence=freeze_florence,
        num_actions=num_actions,
        num_action_query=num_action_query,
        lowdim_obs_dim=lowdim_obs_dim,
        max_T=max_T,
        ff_dim=ff_dim,
        n_heads=n_heads,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        n_layers=n_layers,
        mlp_pdrop=mlp_pdrop,
    ).to(device)
    policy = DiffusionPolicy(
        net=net,
        loss_configs=loss_configs,
        do_compile=do_compile,
        scheduler_name=diffuser_solver,
        num_train_steps=diffuser_train_steps,
        num_infer_steps=diffuser_infer_steps,
        ema_interval=ema_interval,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )
    policy.load_pretrained(acc, save_path, load_epoch_id)
    if use_wandb:
        policy.load_wandb(acc, save_path, do_watch_parameters, save_interval)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr_max, weight_decay=weight_decay, fused=False)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=num_training_epochs,
    )
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
        acc.print(f'action horizon {action_horizon}, success rate {avg_reward}')
