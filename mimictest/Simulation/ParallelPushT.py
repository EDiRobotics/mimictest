import argparse
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
import gym_pusht

def make_env_initializers(num_envs, max_episode_steps):
    initializers = []
    for i in range(num_envs):
        def thunk():
            env = gym.make(
                "gym_pusht/PushT-v0", 
                obs_type="pixels_agent_pos", 
                render_mode="rgb_array",
                max_episode_steps=max_episode_steps,
            )
            return env
        initializers.append(thunk)
    return initializers

class ParallelPushT():
    def __init__(self, num_envs, max_episode_steps):
        self.num_envs = num_envs
        initializers = make_env_initializers(num_envs, max_episode_steps)
        self.envs = SubprocVecEnv(initializers)

    def reset(self):
        obs = self.envs.reset()
        obs['rgb'] = obs['pixels']
        obs['rgb'] = obs['rgb'][:, np.newaxis] # (b h w c) -> (b v h w c)
        obs['low_dim'] = obs['agent_pos']
        return obs 

    def step(self, action):
        obs, rw, done, info = self.envs.step(action)
        obs['rgb'] = obs['pixels']
        obs['rgb'] = obs['rgb'][:, np.newaxis] # (b h w c) -> (b v h w c)
        obs['low_dim'] = obs['agent_pos']
        return obs, rw, done, info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run multiple pusht environments")
    parser.add_argument('--num_envs', type=int, help='number of environments you want to run')
    cfg = parser.parse_args()
    envs = ParallelPushT(cfg.num_envs)
    obs = envs.reset()
    actions = np.random.randn(cfg.num_envs, 2)
    obs, rw, done, info = envs.step(actions)
    import pdb; pdb.set_trace()