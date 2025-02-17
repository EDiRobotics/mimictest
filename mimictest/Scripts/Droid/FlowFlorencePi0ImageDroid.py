import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from mimictest.Utils.AccelerateFix import AsyncStep
from mimictest.Utils.PreProcess import PreProcess
from mimictest.Datasets.DroidLMDBDataset import DroidLMDBDataset
from mimictest.Datasets.DataPrefetcher import DataPrefetcher
from mimictest.Wrappers.DiffusionPolicy import DiffusionPolicy
from mimictest.Nets.FlorencePi0Net import FlorencePi0Net
from mimictest.Train import train
from mimictest.DroidEvaluation import Evaluation

if __name__ == '__main__':
    # Script-specific settings 
    mode = 'train' # or 'eval'

    # Saving path
    save_path = Path('./Save/')
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    abs_mode = True # relative EE action space or absolute EE action space
    folder_name = 'droid_lmdb'
    dataset_path = Path('/root/autodl-tmp/') / folder_name
    bs_per_gpu = 1 # TODO
    workers_per_gpu = 12 
    cache_ratio = 2

    # Space
    num_actions = 7
    num_actions_6d = 10
    lowdim_obs_dim = 0
    obs_horizon = 1
    chunk_size = 20
    process_configs = {
        'rgb': {
            'rgb_shape': (320, 320), # Initial resolution is (180, 320)
            'crop_shape': (280, 280),
            'max': torch.tensor(1.0),
            'min': torch.tensor(0.0),
        },
        'action': {
            'enable_6d_rot': True,
            'abs_mode': abs_mode,
            'max': None, # to be filled
            'min': None,
        },
        'inst_token': {},
        'mask': {},
    }

    # Network
    model_path = Path("microsoft/Florence-2-base")
    freeze_vision_tower = True
    freeze_florence = False
    do_compile = False
    do_profile = False

    # Diffusion
    diffuser_train_steps = 10
    diffuser_infer_steps = 10
    diffuser_solver = "flow_euler"
    beta_schedule = None
    prediction_type = None
    clip_sample = None
    ema_interval = 10

    # Training
    num_training_epochs = 50
    save_interval = 5000
    load_batch_id = 0
    gradient_accumulation_steps = 2
    lr_max = 2e-5
    warmup_steps = 5
    weight_decay = 1e-4
    max_grad_norm = 10
    print_interval = 100
    eval_interval = 100
    use_wandb = False
    wandb_name = "droid"
    do_watch_parameters = False
    record_video = True
    loss_configs = {
        'action': {
            'loss_func': torch.nn.functional.l1_loss,
            'type': 'flow',
            'weight': 1.0,
            'shape': (chunk_size, num_actions_6d),
        },
    }

    # Preparation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        kwargs_handlers=[kwargs],
    )
    device = acc.device
    train_dataset = DroidLMDBDataset(
        dataset_path=dataset_path, 
        obs_horizon=obs_horizon, 
        chunk_size=chunk_size, 
        start_ratio=0,
        end_ratio=0.9,
    )
    test_dataset = DroidLMDBDataset(
        dataset_path=dataset_path, 
        obs_horizon=obs_horizon, 
        chunk_size=chunk_size, 
        start_ratio=0.9,
        end_ratio=1,
    )
    limit = train_dataset.get_action_range(abs_mode)
    process_configs['action']['max'] = limit['action_max']
    process_configs['action']['min'] = limit['action_min']
    preprocessor = PreProcess(
        process_configs=process_configs,
        device=device,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=None, 
        batch_size=bs_per_gpu,
        shuffle=True,
        num_workers=workers_per_gpu,
        drop_last=True,     
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        sampler=None, 
        batch_size=bs_per_gpu,
        shuffle=True,
        num_workers=workers_per_gpu,
        drop_last=True,     
    )
    net = FlorencePi0Net(
        path=model_path,
        freeze_vision_tower=freeze_vision_tower,
        num_actions=num_actions_6d,
        lowdim_obs_dim=lowdim_obs_dim,
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
    policy.load_pretrained(acc, save_path, load_batch_id)
    if use_wandb:
        policy.load_wandb(acc, wandb_name, save_path, do_watch_parameters, save_interval)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
    )
    policy.net, policy.ema_net, optimizer, train_loader, test_loader = acc.prepare(
        policy.net, 
        policy.ema_net, 
        optimizer, 
        train_loader, 
        test_loader, 
        device_placement=[True, True, True, False, False],
    )
    optimizer.step = AsyncStep
    train_prefetcher = DataPrefetcher(train_loader, device)
    test_prefetcher = DataPrefetcher(test_loader, device)
    eva = Evaluation(
        preprcs=preprocessor,
        prefetcher=test_prefetcher,
        save_path=save_path,
        device=device,
    )
    num_eval_ep = None
    max_test_ep_len = None

    if mode == 'train':
        train(
            acc=acc, 
            prefetcher=train_prefetcher, 
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
