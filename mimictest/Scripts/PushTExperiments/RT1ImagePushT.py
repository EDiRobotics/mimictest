import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from mimictest.Utils.AccelerateFix import AsyncStep
from mimictest.Utils.PreProcess import PreProcess
from mimictest.Datasets.PushTDataset import PushTImageDataset
from mimictest.Datasets.DataPrefetcher import DataPrefetcher
from mimictest.Wrappers.BasePolicy import BasePolicy
from mimictest.Nets.RT1 import RT1
from mimictest.Simulation.ParallelPushT import ParallelPushT
from mimictest.Train import train
from mimictest.Evaluation import Evaluation

if __name__ == '__main__':
    # Script-specific settings (general settings stored in 
    mode = 'train' # 'train' or 'eval'

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
    # select from https://pytorch.org/vision/main/models/efficientnetv2.html
    # or https://pytorch.org/vision/main/models/efficientnet.html
    efficientnet_version = "efficientnet_v2_s"
    FiLM_cond_channel = 1 # We don't use it in Robomimic but you can enable it with language-conditioned tasks 
    depth = 8
    vision_token_dim = 512
    ff_dim = 128
    n_heads = 2
    head_dim = 64
    max_T = 128
    token_learner_num_output_tokens = 8
    drop_prob = 0.1
    freeze_vision_tower = False
    do_compile = False
    do_profile = False

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
    do_watch_parameters = False
    record_video = False
    loss_configs = {
        'action': {
            'loss_func': torch.nn.functional.l1_loss,
            'type': 'simple',
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
        # kwargs_handlers=[kwargs],
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
    net = RT1(
        efficientnet_version=efficientnet_version,
        FiLM_cond_channel=FiLM_cond_channel,
        lowdim_obs_dim=lowdim_obs_dim,
        num_actions=num_actions,
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
    policy = BasePolicy(
        net=net,
        loss_configs=loss_configs,
        do_compile=do_compile,
    )
    policy.load_pretrained(acc, save_path, load_epoch_id)
    policy.load_wandb(acc, save_path, do_watch_parameters, save_interval)
    optimizer = torch.optim.AdamW(policy.net.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=num_training_epochs,
    )
    policy.net, optimizer, loader = acc.prepare(
        policy.net, 
        optimizer, 
        loader, 
        device_placement=[True, True, False],
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
        acc.print(f'action horizon {action_horizon}, success rate {avg_reward}')