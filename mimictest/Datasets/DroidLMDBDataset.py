import os
import json
import lmdb
from pickle import loads, dumps
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
from mimictest.Utils.PreProcess import action_axis_to_6d, action_6d_to_axis, action_euler_to_6d, action_6d_to_euler
from tqdm import tqdm

RES = (180, 320)
MAX_LAN_TOKEN_NUM = 100

class DroidReader():

    def __init__(self, lmdb_dir):
        if isinstance(lmdb_dir, str):
            lmdb_dir = Path(lmdb_dir)
        self.lmdb_dir = lmdb_dir
        self.envs = []
        self.txns = []
        self.max_steps = json.load(open(lmdb_dir/'split.json', 'r'))
        split_num = len(self.max_steps)
        self.min_steps = [0] + [self.max_steps[split_id]+1 for split_id in range(split_num-1)]
        self.dataset_len = self.max_steps[-1] + 1

    def __len__(self):
        return self.dataset_len

    def open_lmdb(self, write=False):
        for split_id, split in enumerate(self.max_steps):
            split_path = self.lmdb_dir / str(split_id)
            env = lmdb.open(str(split_path), readonly=not write, create=False, lock=False, map_size=int(3e12))
            txn = env.begin(write=write)
            self.envs.append(env)
            self.txns.append(txn)

    def close_lmdb(self):
        for txn in self.txns:
            txn.commit()
        for env in self.envs:
            env.close()
        self.envs = []
        self.txns = []

    def get_split_id(self, idx, array):
        left, right = 0, len(self.max_steps) - 1
        while left < right:
            mid = (left + right) // 2
            if array[mid] > idx:
                right = mid
            else:
                left = mid + 1
        return left

    def get_episode(self, idx):
        if self.envs == []:
            self.open_lmdb()
        split_id = self.get_split_id(idx, self.max_steps)
        cur_episode = loads(self.txns[split_id].get(f'cur_episode_{idx}'.encode()))
        return cur_episode

    def get_img(self, idx):
        if self.envs == []:
            self.open_lmdb()
        split_id = self.get_split_id(idx, self.max_steps)
        img = loads(self.txns[split_id].get(f'img_{idx}'.encode()))
        img['exterior_image_1_left'] = decode_jpeg(img['exterior_image_1_left'])
        img['exterior_image_2_left'] = decode_jpeg(img['exterior_image_2_left'])
        img['wrist_image_left'] = decode_jpeg(img['wrist_image_left'])
        return img
    
    def get_langs(self, idx):
        if self.envs == []:
            self.open_lmdb()
        split_id = self.get_split_id(idx, self.max_steps)
        ep_id = self.get_episode(idx)
        langs = loads(self.txns[split_id].get(f'lang_{ep_id}'.encode()))
        return langs
    
    def get_lang_tokens(self, idx):
        if self.envs == []:
            self.open_lmdb()
        split_id = self.get_split_id(idx, self.max_steps)
        ep_id = self.get_episode(idx)
        lang_tokens = loads(self.txns[split_id].get(f'lang_token_{ep_id}'.encode()))
        return lang_tokens

    def get_others(self, idx):
        if self.envs == []:
            self.open_lmdb()
        split_id = self.get_split_id(idx, self.max_steps)
        others = loads(self.txns[split_id].get(f'others_{idx}'.encode()))
        return others
    
    def write_lang_token_id(self, tokenizer):
        if self.envs == []:
            self.open_lmdb(write=True)
        last_episode_id = -1
        for idx in tqdm(range(self.dataset_len - 1), desc="Write lang token id"):
            episode_id = self.get_episode(idx)
            if episode_id != last_episode_id:
                langs = self.get_langs(idx)
                lang_tokens = []
                for lang in langs:
                    lang_token = tokenizer(
                        lang.decode('utf-8'),
                        return_tensors="pt",
                        padding=False,
                        max_length=None,
                        truncation=None,
                        return_token_type_ids=False,
                    )['input_ids'][0].numpy()
                    lang_tokens.append(lang_token)
                split_id = self.get_split_id(idx, self.max_steps)
                self.txns[split_id].put(
                    f'lang_token_{episode_id}'.encode(),
                    dumps(lang_tokens),
                )
                last_episode_id = episode_id      
        self.close_lmdb()

class DroidLMDBDataset(Dataset):

    def __init__(self, dataset_path, obs_horizon, chunk_size, start_ratio, end_ratio):
        self.obs_horizon = obs_horizon
        self.chunk_size = chunk_size
        self.reader = DroidReader(dataset_path)
        self.dummy_rgb = torch.zeros((obs_horizon, 3, 3) + RES, dtype=torch.uint8) # (t v c h w)
        self.dummy_inst_token = torch.ones(MAX_LAN_TOKEN_NUM, dtype=torch.int) 
        self.dummy_action = torch.zeros((chunk_size, 7))
        self.dummy_mask = torch.zeros(chunk_size)
        self.start_step = int(self.reader.dataset_len * start_ratio)
        self.end_step = int(self.reader.dataset_len * end_ratio) - chunk_size - obs_horizon
    
    def __len__(self):
        return self.end_step - self.start_step
    
    def get_action_range(self, abs_mode):
        actions = []
        for idx in range(10000):
            others = self.reader.get_others(idx)
            action = torch.from_numpy(
                np.concatenate((others['action_dict']['cartesian_position'], others['action_dict']['gripper_position']))
            ).to(torch.float32)
            rot = action[3:6]
            if abs_mode:
                action = torch.cat((action[:3], action_axis_to_6d(rot), action[6:]), dim=-1)
            else:
                action = torch.cat((action[:3], action_euler_to_6d(rot), action[6:]), dim=-1)
            actions.append(action)
        actions = torch.stack(actions)
        return {
            "action_max": actions.max(dim=0)[0],
            "action_min": actions.min(dim=0)[0],
        }

    def __getitem__(self, idx):
        idx = idx + self.start_step

        rgb = self.dummy_rgb.clone()
        inst_token = self.dummy_inst_token.clone()
        actions = self.dummy_action.clone()
        mask = self.dummy_mask.clone()
        
        episode_id = self.reader.get_episode(idx)
        lang_tokens = self.reader.get_lang_tokens(idx)
        if lang_tokens != []:
            lang_token = random.choice(lang_tokens)
            inst_token[:len(lang_token)] = torch.from_numpy(lang_token)

        for obs_idx in range(self.obs_horizon):
            img = self.reader.get_img(idx + obs_idx)
            if self.reader.get_episode(idx + obs_idx) == episode_id:
                rgb[obs_idx, 0] = img['exterior_image_1_left']
                rgb[obs_idx, 1] = img['exterior_image_2_left']
                rgb[obs_idx, 2] = img['wrist_image_left']
        
        action_start = idx + self.obs_horizon - 1
        for action_idx in range(self.chunk_size):
            others = self.reader.get_others(action_start + action_idx)
            if self.reader.get_episode(action_start + action_idx) == episode_id:
                actions[action_idx, :-1] = torch.from_numpy(others['action_dict']['cartesian_position'])
                actions[action_idx, -1] = torch.from_numpy(others['action_dict']['gripper_position'])
                mask[action_idx] = 1
        
        return {
            "rgb": rgb,
            "inst_token": inst_token,
            "action": actions,
            "mask": mask,
        }

if __name__ == "__main__":
    reader = DroidReader("/root/autodl-tmp/droid_lmdb")
    from transformers import AutoProcessor
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    tokenizer = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).tokenizer
    reader.write_lang_token_id(tokenizer)
    import pdb; pdb.set_trace()