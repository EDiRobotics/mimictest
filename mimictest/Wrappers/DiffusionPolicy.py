import os
import copy
import math
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import EMAModel
from mimictest.Wrappers.BasePolicy import BasePolicy

class DiffusionPolicy(BasePolicy):
    def __init__(
            self,
            net,
            loss_configs,
            do_compile,
            scheduler_name,
            num_train_steps,
            num_infer_steps,
            beta_schedule,
            clip_sample,
            prediction_type,
            ema_interval,
        ):
        super().__init__(net, loss_configs, do_compile)
        self.use_ema = True
        self.ema_interval = ema_interval
        self.ema = EMAModel(
            parameters=self.net.parameters(),
            power=0.9999)
        self.ema_net = copy.deepcopy(self.net) 

        self.scheduler_name = scheduler_name
        self.prediction_type = prediction_type
        self.num_infer_steps = num_infer_steps
        if scheduler_name == 'ddpm':
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_steps,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample,
                prediction_type=prediction_type,
            )
        elif scheduler_name == 'ddim':
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_steps,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample,
                prediction_type=prediction_type,
            )
        elif scheduler_name == "flow_euler":
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=num_train_steps,
            )
    
    def compute_loss(self, batch):
        device = batch['rgb'].device
        B = batch['rgb'].shape[0]

        # sample a diffusion iteration for each data point
        batch['timesteps'] = torch.randint(
            1, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device,
        ).long()

        noise = {}
        batch['noisy_inputs'] = {}
        for key in self.loss_configs:
            if self.loss_configs[key]['type'] == 'diffusion':
                noise[key] = torch.randn((B,)+self.loss_configs[key]['shape'], device=device)
                batch['noisy_inputs'][key] = self.noise_scheduler.add_noise(
                    sample=batch[key], 
                    noise=noise[key], 
                    timestep=batch['timesteps'],
                )
            elif self.loss_configs[key]['type'] == 'flow':
                noise[key] = torch.randn((B,)+self.loss_configs[key]['shape'], device=device)
                batch['noisy_inputs'][key] = self.noise_scheduler.scale_noise(
                    sample=batch[key], 
                    noise=noise[key], 
                    timestep=batch['timesteps'],
                )

        batch['obs_features'] = None

        pred, _ = self.net(batch)

        loss = {'total_loss': 0}
        for key in self.loss_configs:
            loss_func = self.loss_configs[key]['loss_func']
            weight = self.loss_configs[key]['weight']

            # Deal with different diffusion losses
            if self.loss_configs[key]['type'] == 'diffusion':
                if self.prediction_type == 'epsilon':
                    loss[key] = loss_func(pred[key], noise[key], reduction="none")
                elif self.prediction_type == 'sample':
                    loss[key] = loss_func(pred[key], batch[key], reduction="none")
                elif self.prediction_type == 'v_prediction':
                    target = self.noise_scheduler.get_velocity(batch[key], noise[key], batch['timesteps'])
                    loss[key] = loss_func(pred[key], target, reduction="none")
            elif self.loss_configs[key]['type'] == 'flow':
                loss[key] = loss_func(pred[key], noise[key] - batch[key], reduction="none")
            elif self.loss_configs[key]['type'] == 'simple':
                loss[key] = loss_func(pred[key], batch[key], reduction="none")

            # Deal with masking
            data_shape = pred[key].shape
            if "mask" in batch:
                mask_shape = batch["mask"].shape
                for _ in range(len(data_shape) - len(mask_shape)):
                    new_mask = batch["mask"].unsqueeze(-1)
                loss[key] = (loss[key] * new_mask).sum() / (new_mask.sum() * math.prod(data_shape[len(mask_shape):]))
            else:
                loss[key] = loss[key].sum() / math.prod(data_shape)

            loss['total_loss'] += loss[key] * weight
        return loss

    def infer(self, batch):
        device = batch['rgb'].device
        B = batch['rgb'].shape[0]

        batch['noisy_inputs'] = {}
        for key in self.loss_configs:
            if self.loss_configs[key]['type'] == 'diffusion' or self.loss_configs[key]['type'] == 'flow':
                batch['noisy_inputs'][key] = torch.randn((B,)+self.loss_configs[key]['shape'], device=device)
        ones = torch.ones((B,), device=device).long()

        batch['obs_features'] = None
        self.noise_scheduler.set_timesteps(self.num_infer_steps)
        for k in self.noise_scheduler.timesteps:
            # A special fix because diffusers scheduler will add up _step_index, but we want it to kep the same at every k
            if "flow" in self.scheduler_name:
                current_step_index = self.noise_scheduler._step_index
            batch['timesteps'] = k*ones
            out, batch['obs_features'] = self.ema_net(batch)
            for key in self.loss_configs:
                if self.loss_configs[key]['type'] == 'diffusion' or self.loss_configs[key]['type'] == 'flow':
                    if "flow" in self.scheduler_name:
                        self.noise_scheduler._step_index = current_step_index
                    batch['noisy_inputs'][key] = self.noise_scheduler.step(
                        model_output=out[key],
                        timestep=k,
                        sample=batch['noisy_inputs'][key],
                    ).prev_sample

        pred = {}
        for key in out:
            if self.loss_configs[key]['type'] == 'diffusion' or self.loss_configs[key]['type'] == 'flow':
                pred[key] = batch['noisy_inputs'][key]
            elif self.loss_configs[key]['type'] == 'simple':
                pred[key] = out[key]
        return pred
                        
    def save_pretrained(self, acc, path, epoch_id):
        acc.wait_for_everyone()
        if hasattr(acc.unwrap_model(self.net), '_orig_mod'): # the model has been compiled
            ckpt = {"net": acc.unwrap_model(self.net)._orig_mod.state_dict()}
            if self.use_ema:
                ckpt["ema"] = acc.unwrap_model(self.ema_net)._orig_mod.state_dict()
        else:
            ckpt = {"net": acc.unwrap_model(self.net).state_dict()}
            if self.use_ema:
                ckpt["ema"] = acc.unwrap_model(self.ema_net).state_dict()
        acc.save(ckpt, path / f'policy_{epoch_id}.pth')

    def load_pretrained(self, acc, path, load_epoch_id):
        if os.path.isfile(path / f'policy_{load_epoch_id}.pth'):
            ckpt = torch.load(path / f'policy_{load_epoch_id}.pth', map_location='cpu', weights_only=True)
            if self.do_compile:
                missing_keys, unexpected_keys = self.net._orig_mod.load_state_dict(ckpt["net"])
                if self.use_ema:
                    missing_keys, unexpected_keys = self.ema_net._orig_mod.load_state_dict(ckpt["ema"])
            else:
                missing_keys, unexpected_keys = self.net.load_state_dict(ckpt["net"])
                if self.use_ema:
                    missing_keys, unexpected_keys = self.ema_net.load_state_dict(ckpt["ema"])
            acc.print('load ', path / f'policy_{load_epoch_id}.pth', '\nmissing ', missing_keys, '\nunexpected ', unexpected_keys)
        else: 
            acc.print(path / f'policy_{load_epoch_id}.pth', 'does not exist. Initialize new checkpoint')

    def update_ema(self):
        self.ema.step(self.net.parameters())

    def copy_ema_to_ema_net(self):
        self.ema.copy_to(self.ema_net.parameters())
