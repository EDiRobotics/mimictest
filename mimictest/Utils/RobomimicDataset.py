import numpy as np
from time import time
import torch
from einops import rearrange
from torch.utils.data import Dataset
from robomimic.utils.dataset import SequenceDataset

class DataPrefetcher():
    def __init__(self, loader, device):
        self.device = device
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            # Dataloader will prefetch data to cpu so this step is very quick
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return 
        with torch.cuda.stream(self.stream):
            for key in self.batch:
                self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if batch[key] is not None:
                    batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch, time()-clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time

class CustomMimicDataset(Dataset):
    def __init__(self, dataset_path, obs_horizon, chunk_size, start_ratio, end_ratio):
        self.obs_dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=(            
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
                "agentview_image",
                "robot0_eye_in_hand_image",
            ),
            dataset_keys=(
                "actions",
            ),
            load_next_obs=False,
            frame_stack=obs_horizon,
            seq_length=1,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode="all",
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
        )
        self.action_dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=(            
            ),
            dataset_keys=(
                "actions",
            ),
            load_next_obs=False,
            frame_stack=1,
            seq_length=chunk_size,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode="all",
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
        )
        self.start_step = int(len(self.obs_dataset) * start_ratio)
        self.end_step = int(len(self.obs_dataset) * end_ratio) - chunk_size

    def __getitem__(self, idx):
        idx = idx + self.start_step
        batch = self.obs_dataset[idx]
        action_batch = self.action_dataset[idx]
        rgbs = torch.from_numpy(np.stack((batch['obs']['agentview_image'], batch['obs']['robot0_eye_in_hand_image']), axis=1))
        low_dims = np.concatenate((batch['obs']['robot0_eef_pos'], batch['obs']['robot0_eef_quat'], batch['obs']['robot0_gripper_qpos']), axis=-1).astype(np.float32)
        return {
            'rgbs': rearrange(rgbs, 't v h w c -> t v c h w').contiguous(),
            'low_dims': low_dims,
            'actions': action_batch['actions'],
        }

    def __len__(self):
        return self.end_step - self.start_step
