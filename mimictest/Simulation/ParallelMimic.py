import argparse
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils
from mimictest.Simulation.robosuite_gym_wrapper import GymWrapper

def make_env_initializers(num_envs, env_meta):
    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)
    initializers = []
    for i in range(num_envs):
        def thunk():
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_meta["env_name"],
                render=False,
                render_offscreen=True,
                use_image_obs=True,
            ).env
            env = GymWrapper(env)
            return env
        initializers.append(thunk)
    return initializers

class ParallelMimic():
    def __init__(self, dataset_path, num_envs, abs_mode):
        self.num_envs = num_envs
        self.render = False
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        if env_meta['env_kwargs']['use_camera_obs']:
            self.render = True
        if abs_mode:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        initializers = make_env_initializers(num_envs, env_meta)
        self.envs = SubprocVecEnv(initializers)

    def combine_obs(self, obs):
        new_obs = {}
        for key in obs[0].keys():
            new_obs[key] = np.stack([obs[i][key] for i in range(self.num_envs)])
        return new_obs

    def reset(self):
        obs = self.envs.reset()
        obs = self.combine_obs(obs)
        obs['low_dim'] = np.concatenate((obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos']), axis=1)
        if self.render:
            obs['agentview_image'] = np.flip(obs['agentview_image'], axis=1)
            obs['robot0_eye_in_hand_image'] = np.flip(obs['robot0_eye_in_hand_image'], axis=1)
            obs['rgb'] = np.stack((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1)
        return obs 

    def step(self, action):
        obs, rw, done, info = self.envs.step(action)
        obs = self.combine_obs(obs)
        obs['low_dim'] = np.concatenate((obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos']), axis=1)
        if self.render:
            obs['agentview_image'] = np.flip(obs['agentview_image'], axis=1)
            obs['robot0_eye_in_hand_image'] = np.flip(obs['robot0_eye_in_hand_image'], axis=1)
            obs['rgb'] = np.stack((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1)
        return obs, rw, done, info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run multiple robosuite environments based on the env_meta in the robomimic dataset provided.")
    parser.add_argument('--dataset_path', type=str, help='path of the robomimic dataset')
    parser.add_argument('--num_envs', type=int, help='number of environments you want to run')
    parser.add_argument('--abs_mode', action='store_true', help='speficy it if you use absolute action space')
    cfg = parser.parse_args()
    envs = ParallelMimic(cfg.dataset_path, cfg.num_envs, cfg.abs_mode)
    obs = envs.reset()
    actions = np.random.randn(cfg.num_envs, 7)
    obs, rw, done, info = envs.step(actions)
    import pdb; pdb.set_trace()
