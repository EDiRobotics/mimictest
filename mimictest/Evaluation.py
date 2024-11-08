import os
import math
import numpy as np
import torch
from einops import rearrange
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

class Evaluation():
    def __init__(self, envs, num_envs, preprcs, obs_horizon, action_horizon, save_path, device):
        self.envs = envs
        self.num_envs = num_envs
        self.preprcs = preprcs
        self.device = device
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.save_path = save_path
        return None

    def fill_buffer(self, obs):
        rgb = torch.from_numpy(obs['rgb'])
        rgb = rearrange(rgb, 'b v h w c -> b v c h w').contiguous()
        low_dim = torch.from_numpy(obs['low_dim']).float()
        if len(self.rgb_buffer) == 0: # Fill the buffer 
            for i in range(self.obs_horizon):
                self.rgb_buffer.append(rgb)
                self.low_dim_buffer.append(low_dim)
        elif len(self.rgb_buffer) == self.obs_horizon: # Update the buffer
            self.rgb_buffer.pop(0)
            self.rgb_buffer.append(rgb)
            self.low_dim_buffer.pop(0)
            self.low_dim_buffer.append(low_dim)
        else:
            raise ValueError(f"Evaluation.py: buffer len {len(self.rgb_buffer)}")

    def evaluate_on_env(self, acc, policy, batch_idx, num_eval_ep, max_test_ep_len, record_video=False):
        if policy.use_ema:
            policy.ema_net.eval()
        else:
            policy.net.eval()
        total_rewards = np.zeros((self.num_envs)) 
        with torch.no_grad():
            for ep in range(num_eval_ep):
                rewards = np.zeros((self.num_envs))
                self.rgb_buffer = [] 
                self.low_dim_buffer = []
                obs = self.envs.reset()
                self.fill_buffer(obs)
                self.num_cameras = obs['rgb'].shape[1]
                if record_video:
                    videos = []
                    for camera_id in range(self.num_cameras):
                        videos.append([[] for i in range(self.num_envs)])

                t = 0
                progress_bar = tqdm(total=max_test_ep_len, desc=f"run episode {ep+1} of {num_eval_ep}", disable=not acc.is_main_process)
                while t < max_test_ep_len:
                    batch = {
                        'rgb': torch.stack(self.rgb_buffer, dim=1).to(self.device),
                        'low_dim': torch.stack(self.low_dim_buffer, dim=1).to(self.device),
                    }
                    batch = self.preprcs.process(batch, train=False)
                    pred = policy.infer(batch)
                    pred_actions = self.preprcs.back_process(pred)['action'].cpu().numpy()
                    for action_id in range(self.action_horizon[0], self.action_horizon[1]):
                        obs, rw, done, info = self.envs.step(pred_actions[:, action_id])
                        self.fill_buffer(obs)
                        for env_id in range(self.num_envs):
                            if rewards[env_id] == 0 and rw[env_id] == 1:
                                rewards[env_id] = 1
                                print(f'gpu{acc.process_index}_episode{ep}_env{env_id}: get reward! step {t}')
                        if record_video:
                            for env_id in range(self.num_envs):
                                if rewards[env_id] == 0:
                                    for camera_id in range(self.num_cameras):
                                        img = obs['rgb'][env_id, camera_id].astype(np.uint8).copy()
                                        videos[camera_id][env_id].append(img)
                        t += 1
                        progress_bar.update(1)
                        if t >= max_test_ep_len:
                            break
                progress_bar.close()

                # If there are multiple episodes in max_test_ep_len, only conut the 1st episode
                rewards = np.where(rewards > 0, 1, rewards)
                total_rewards += rewards 
                print(f'gpu{acc.process_index}_epidose{ep}: rewards {rewards}')
                if record_video:
                    for env_id in range(self.num_envs):
                        prefix = f'batch{batch_idx}_gpu{acc.process_index}_episode{ep}_env{env_id}_reward{rewards[env_id]}'
                        for camera_id in range(self.num_cameras):
                            clip = ImageSequenceClip(videos[camera_id][env_id], fps=30)
                            clip.write_gif(self.save_path / (prefix+f'_camera{camera_id}.gif'), fps=30)
            total_rewards /= num_eval_ep
        return total_rewards.mean() 
