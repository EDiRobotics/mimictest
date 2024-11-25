import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from mimictest.Utils.AccelerateFix import AsyncStep
from mimictest.Utils.PreProcess import PreProcess
from mimictest.Datasets.DroidFlowDataset import DroidFlowDataset
from mimictest.Datasets.DataPrefetcher import DataPrefetcher
from mimictest.Wrappers.DiffusionPolicy import DiffusionPolicy
from mimictest.Nets.FlorenceMDTNet import FlorenceMDTNet
from mimictest.Train import train
from mimictest.FlowEvaluation import Evaluation

if __name__ == '__main__':
    # Script-specific settings 
    mode = 'train' # or 'eval'

    # Saving path
    save_path = Path('./Save/')
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    abs_mode = True # relative EE action space or absolute EE action space
    folder_name = 'lmdb_droid'
    dataset_path = Path('/root/autodl-tmp/') / folder_name
    bs_per_gpu = 160
    workers_per_gpu = 15 
    cache_ratio = 2

    # Space
    num_actions = 2
    lowdim_obs_dim = 0
    obs_horizon = 1
    chunk_size = 100
    process_configs = {
        'rgb': {
            'rgb_shape': (320, 320), # Initial resolution is (180, 320)
            'crop_shape': (280, 280),
            'max': torch.tensor(1.0),
            'min': torch.tensor(0.0),
        },
        'action': {
            'max': torch.tensor([50, 50]), 
            'min': torch.tensor([-50, -50]), 
        },
        'inst_token': {},
        'mask': {},
        'flow_start': {},
    }

    # Network
    model_path = Path("microsoft/Florence-2-base")
    freeze_vision_tower = True
    freeze_florence = False
    num_action_query = 64
    max_T = 512
    ff_dim = 256
    n_heads = 4
    attn_pdrop = 0.3
    resid_pdrop = 0.1
    mlp_pdrop = 0.05
    n_layers = 8
    do_compile = False
    do_profile = False

    # Diffusion
    diffuser_train_steps = 10
    diffuser_infer_steps = 10
    diffuser_solver = "ddim"
    beta_schedule = "squaredcos_cap_v2"
    prediction_type = 'epsilon'
    clip_sample = False
    ema_interval = 10

    # Training
    num_training_epochs = 50
    save_interval = 5000
    load_batch_id = 85000
    gradient_accumulation_steps = 2
    lr_max = 1e-4
    warmup_steps = 5
    weight_decay = 1e-4
    max_grad_norm = 10
    print_interval = 500
    do_watch_parameters = False
    record_video = True
    loss_configs = {
        'action': {
            'loss_func': torch.nn.functional.l1_loss,
            'type': 'diffusion',
            'weight': 1.0,
            'shape': (chunk_size, num_actions),
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
    train_dataset = DroidFlowDataset(
        dataset_path=dataset_path, 
        obs_horizon=obs_horizon, 
        chunk_size=chunk_size, 
        start_ratio=0,
        end_ratio=0.9,
    )
    test_dataset = DroidFlowDataset(
        dataset_path=dataset_path, 
        obs_horizon=obs_horizon, 
        chunk_size=chunk_size, 
        start_ratio=0.9,
        end_ratio=1,
    )
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
    policy.load_pretrained(acc, save_path, load_batch_id)
    policy.load_wandb(acc, save_path, do_watch_parameters, save_interval)
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
            bs_per_gpu=bs_per_gpu,
            max_grad_norm=max_grad_norm,
            record_video=record_video,
            do_profile=do_profile,
        )
