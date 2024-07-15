import numpy as np
import torch
from robomimic.utils.dataset import SequenceDataset
from mimictest.Utils.PreProcess import action_euler_to_6d, action_axis_to_6d

def ComputeLimit(dataset_path, abs_mode):
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=(            
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        dataset_keys=(
            "actions",
        ),
        load_next_obs=False,
        frame_stack=1,
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
    low_dims = []
    actions = []
    for i in range(len(dataset)):
        low_dim = np.concatenate((dataset[i]['obs']['robot0_eef_pos'], dataset[i]['obs']['robot0_eef_quat'], dataset[i]['obs']['robot0_gripper_qpos']), axis=-1)[0].astype(np.float32)
        low_dims.append(torch.from_numpy(low_dim))
        if abs_mode:
            action = action_axis_to_6d(torch.from_numpy(dataset[i]['actions'][0]))
        else:
            action = action_euler_to_6d(torch.from_numpy(dataset[i]['actions'][0]))
        actions.append(action)
    low_dims = torch.stack(low_dims)
    actions = torch.stack(actions)
    return {
        "action_max": actions.max(dim=0)[0],
        "action_min": actions.min(dim=0)[0],
        "low_dim_max": low_dims.max(dim=0)[0],
        "low_dim_min": low_dims.min(dim=0)[0],
    }
