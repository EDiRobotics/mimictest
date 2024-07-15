import copy
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

class DiffusionPolicy():
    def __init__(self,
                net,
                num_actions,
                chunk_size,
                scheduler_name,
                num_train_steps,
                num_infer_steps,
                beta_schedule,
                clip_sample,
                prediction_type,
                loss_func = None,
        ):
        
        self.num_actions = num_actions
        self.chunk_size = chunk_size
        self.net = net 
        self.ema = EMAModel(
            parameters=self.net.parameters(),
            power=0.9999)
        self.ema_net = copy.deepcopy(self.net) 

        self.prediction_type = prediction_type
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
        self.noise_scheduler.set_timesteps(num_infer_steps)
        self.loss_func = loss_func

    def compute_loss(self, rgb, low_dim, actions):
        # sample noise to add to actions
        noise = torch.randn(actions.shape, device=rgb.device)

        # sample a diffusion iteration for each data point
        B = rgb.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=rgb.device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            actions, noise, timesteps)

        pred = self.net(rgb, low_dim, noisy_actions, timesteps)

        if self.prediction_type == 'epsilon':
            loss = self.loss_func(pred, noise, reduction='none')
        elif self.prediction_type == 'sample':
            loss = self.loss_func(pred, actions, reduction='none')
        elif self.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(actions, noise, timesteps)
            loss = self.loss_func(pred, target, reduction='none')
        return loss.sum(dim=(-1,-2)).mean()

    def infer(self, rgb, low_dim):
        B = rgb.shape[0]
        noisy_actions = torch.randn((B, self.chunk_size, self.num_actions), device=rgb.device)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.ema_net(rgb, low_dim, noisy_actions, k)
            noisy_actions = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_actions,
            ).prev_sample
        return noisy_actions
                        
    def update_ema(self):
        self.ema.step(self.net.parameters())
