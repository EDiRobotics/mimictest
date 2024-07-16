import os
import math
import numpy as np
import torch
from einops import rearrange
import av
import cv2

def save_video(file, video):
    container = av.open(file, 'w', 'mp4')
    stream = container.add_stream('libx264', rate=5)
    stream.pix_fmt = 'yuv420p'
    for i in range(len(video)):
        frame = av.VideoFrame.from_ndarray(video[i], format='rgb24')
        frame = frame.reformat(format=stream.pix_fmt)
        for packet in stream.encode(frame):
            container.mux(packet)
    # Flush stream
    for packet in stream.encode():
        container.mux(packet)
    container.close()

class ReplayEnv():
    def __init__(self, dataset, read_step):
        self.read_step = read_step
        self.dataset = dataset

    def get_obs(self):
        rgbs, inst, actions, percentage = self.dataset[self.read_step]
        rgb = rearrange(rgbs, 'c h w -> 1 h w c')
        obs = {
            'rgb': rgb.contiguous().numpy(),
            'instruction': inst,
            'actions': actions.numpy(),
        }
        return obs 

    def reset(self, read_step=None):
        if read_step is not None:
            self.read_step = read_step
        obs = self.get_obs()
        self.read_step += 1
        return obs

    def step(self, action): # I dont care the actual action
        obs = self.get_obs()
        self.read_step += 1
        return obs, 0, None, None 

class Evaluation():
    def __init__(self, envs, num_envs, preprcs, obs_horizon, chunk_size, num_actions, smooth_factor, device):
        self.envs = envs
        self.num_envs = num_envs
        self.preprcs = preprcs
        self.device = device
        self.obs_horizon = obs_horizon
        self.chunk_size = chunk_size
        self.num_actions = num_actions
        self.chunk = np.zeros((num_envs, chunk_size, chunk_size, num_actions))
        self.cur_step = 0
        self.exp_weights = np.exp(-smooth_factor * np.arange(chunk_size)) 
        return None

    def action_chunking(self, new_actions):
        self.cur_step += 1
        self.chunk[:, 1:, :] = self.chunk[:, :-1, :]
        self.chunk[:, 0, :] = new_actions # newest action
        num_steps = min(self.chunk_size, self.cur_step)
        current_action = self.chunk[:, 0, 0] * self.exp_weights[num_steps-1]
        for i in range(1, num_steps):
            current_action += self.chunk[:, i, i] * self.exp_weights[num_steps-i-1]
        current_action /= self.exp_weights[:num_steps].sum()
        return current_action

    def evaluate_on_env(self, policy, num_eval_ep, max_test_ep_len, save_path=None, record_video=False):
        policy.ema_net.eval()
        # TODO: criterion = torch.nn.XXLoss(reduction='none').to(acc.device)
        total_rewards = np.zeros((self.num_envs)) 
        with torch.no_grad():
            for ep in range(num_eval_ep):
                self.cur_step = 0
                self.chunk = np.zeros_like(self.chunk)
                rewards = np.zeros((self.num_envs))
                rgb_buffer = [] 
                low_dim_buffer = []
                obs = self.envs.reset()
                if record_video:
                    video_hand = [[] for i in range(self.num_envs)]
                    video_agent = [[] for i in range(self.num_envs)]
                
                for t in range(max_test_ep_len):
                    # Add RGB image to placeholder 
                    rgb = np.stack((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1)
                    rgb = torch.from_numpy(rgb)
                    rgb = rearrange(rgb, 'b v h w c -> b v c h w').contiguous()
                    rgb = self.preprcs.rgb_process(rgb, train=False).to(self.device)
                    low_dim = np.concatenate((obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos']), axis=1)
                    low_dim = torch.from_numpy(low_dim).float()
                    low_dim = self.preprcs.low_dim_normalize(low_dim.to(self.device))
                    if len(rgb_buffer) == 0: # Fill the buffer 
                        for i in range(self.obs_horizon):
                            rgb_buffer.append(rgb)
                            low_dim_buffer.append(low_dim)
                    elif len(rgb_buffer) == self.obs_horizon: # Update the buffer
                        rgb_buffer.pop(0)
                        rgb_buffer.append(rgb)
                        low_dim_buffer.pop(0)
                        low_dim_buffer.append(low_dim)
                    else:
                        raise ValueError(f"Evaluation.py: buffer len {len(rgb_buffer)}")
                    pred_actions = policy.infer(torch.stack(rgb_buffer, dim=1), torch.stack(low_dim_buffer, dim=1))
                    pred_actions = self.preprcs.action_back_normalize(pred_actions).cpu().numpy()
                    # TODO: recon_loss = criterion(pred_actions.cpu(), torch.from_numpy(obs['actions']).unsqueeze(0)).mean(dim=0).sum() 
                    # TODO: smooth_action = self.action_chunking(pred_actions[:, :self.chunk_size])
                    for action_id in range(self.chunk_size):
                        obs, rw, done, info = self.envs.step(pred_actions[:, action_id])
                        for env_id in range(self.num_envs):
                            if rewards[env_id] == 0 and rw[env_id] == 1:
                                rewards[env_id] = 1
                                print(f'get reward! step {t*self.chunk_size+action_id}')
                        if record_video:
                            for env_id in range(self.num_envs):
                                if rewards[env_id] == 0:
                                    img = obs['robot0_eye_in_hand_image'][env_id].astype(np.uint8).copy()
                                    video_hand[env_id].append(img)
                                    img = obs['agentview_image'][env_id].astype(np.uint8).copy()
                                    # TODO: cv2.putText(img, f'loss {recon_loss}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                    video_agent[env_id].append(img)

                # If there are multiple episodes in max_test_ep_len, only conut the 1st episode
                rewards = np.where(rewards > 0, 1, rewards)
                total_rewards += rewards 
                print(f'{self.device}_epidose{ep}: rewards {rewards}')
                if record_video:
                    for env_id in range(self.num_envs):
                        save_video(cfg.save_path+f'gpu{acc.process_index}_episode{ep}_rank{env_id}_hand.mp4', video_hand[i])
                        save_video(cfg.save_path+f'gpu{acc.process_index}_episode{ep}_rank{env_id}_agent.mp4', video_agent[i])
                        print(f'gpu{acc.process_index}_video{ep}_rank{env_id}: reward {rewards[env_id]}')
            total_rewards /= num_eval_ep
        return total_rewards.mean() 
