import os
from random import choice
import json
import lmdb
from pickle import loads, dumps
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
from tqdm import tqdm

RES = (180, 320)
MAX_LAN_TOKEN_NUM = 128
RESOLUTION_SCALE_DOWN = 4
EXTERNAL_CAMERA_INTRINSIC = np.array([
    [520., 0, 640.],
    [0, 520., 360.],
    [0, 0, 1.],
]) # Not real intrinsic of each camera, just an estimated number
WRIST_CAMERA_INTRINSIC = np.array([
    [729., 0, 640.],
    [0, 729., 360.],
    [0, 0, 1.],
]) # Not real intrinsic of each camera, just an estimated number

def get_ext_matrix(pose):
    trans = pose[:3]
    rot = Rotation.from_euler("xyz", np.array(pose[3:])).as_matrix()

    # Get extrinsics matrix from rotation and translation
    cam_to_world = np.zeros((4,4))
    cam_to_world[:3, :3] = rot
    cam_to_world[:3, 3] = trans
    cam_to_world[3, 3] = 1
    world_to_cam = np.linalg.inv(cam_to_world)
    return world_to_cam

def project(world_traj, world_to_cam, intrinsic):
    # Get 3D traj in camera frame
    ones = np.ones((world_traj.shape[0], 1))
    world_traj = np.concatenate([world_traj, ones], axis=1)
    camera_traj = (world_to_cam @ world_traj.T)[:3]

    # Get pixel traj
    z = camera_traj[2]
    pixel_traj = (intrinsic @ camera_traj) / z
    pixel_traj = pixel_traj[:2] / RESOLUTION_SCALE_DOWN

    return pixel_traj.T

class DroidReader():

    def __init__(self, lmdb_dir):
        if isinstance(lmdb_dir, str):
            lmdb_dir = Path(lmdb_dir)
        self.lmdb_dir = lmdb_dir
        self.split_set = json.load(open(lmdb_dir/'split.json', 'r'))
        self.envs = []
        self.txns = []
        self.max_steps = []
        self.max_episode_ids = []
        for split_id, split in enumerate(self.split_set):
            self.max_steps.append(split["max_step"])
            self.max_episode_ids.append(split["cur_episode_id"])
        split_num = len(self.split_set)
        self.min_steps = [0] + [self.max_steps[split_id]+1 for split_id in range(split_num-1)]
        self.dataset_len = self.split_set[-1]["max_step"] + 1

    def __len__(self):
        return self.dataset_len
    
    def open_lmdb(self, write=False):
        for split_id, split in enumerate(self.split_set):
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
        left, right = 0, len(self.max_steps)
        while left < right:
            mid = (left + right) // 2
            if array[mid] > idx:
                right = mid
            else:
                left = mid + 1
        return left

    def get(self, idx, load_image, left):
        if self.envs == []:
            self.open_lmdb()

        split_id = self.get_split_id(idx, self.max_steps)

        episode_id = loads(self.txns[split_id].get(f'episode_id_{idx}'.encode()))
        data = {
            "episode_id": episode_id,
            "inst_token": choice(loads(self.txns[split_id].get(f'inst_token_{episode_id}'.encode()))),
            "gripper_pos": loads(self.txns[split_id].get(f'gripper_pos_{idx}'.encode())),
        }

        # Pick left or right image 
        if left:
            data["flow"] = loads(self.txns[split_id].get(f'gripper_flow_from_left_{idx}'.encode()))
            if load_image:
                data["rgb"] = decode_jpeg(loads(self.txns[split_id].get(f'left_rgb_{idx}'.encode())))
        else:
            data["flow"] = loads(self.txns[split_id].get(f'gripper_flow_from_right_{idx}'.encode()))
            if load_image:
                data["rgb"] = decode_jpeg(loads(self.txns[split_id].get(f'right_rgb_{idx}'.encode())))

        return data
    
    def rewrite_flow(self):
        if self.envs == []:
            self.open_lmdb(write=True)

        for idx in tqdm(range(self.dataset_len - 1), desc="Rewrite gripper flow from camera"):
            split_id = self.get_split_id(idx, self.max_steps)
            episode_id = loads(self.txns[split_id].get(f'episode_id_{idx}'.encode()))
            cartesian_pos = loads(self.txns[split_id].get(f'cartesian_pos_{idx}'.encode()))

            left_pose = loads(self.txns[split_id].get(f'left_extrinsic_{episode_id}'.encode()))
            gripper_flow_from_left = project(
                cartesian_pos[np.newaxis, :3],
                get_ext_matrix(left_pose),
                EXTERNAL_CAMERA_INTRINSIC,
            )[0]
            self.txns[split_id].replace(f'gripper_flow_from_left_{idx}'.encode(), dumps(gripper_flow_from_left))

            right_pose = loads(self.txns[split_id].get(f'right_extrinsic_{episode_id}'.encode()))
            gripper_flow_from_right = project(
                cartesian_pos[np.newaxis, :3],
                get_ext_matrix(right_pose),
                EXTERNAL_CAMERA_INTRINSIC,
            )[0]
            self.txns[split_id].replace(f'gripper_flow_from_right_{idx}'.encode(), dumps(gripper_flow_from_right))
        self.close_lmdb()
    
    def write_lang_token_id(self, tokenizer):
        if self.envs == []:
            self.open_lmdb(write=True)
        
        cartesian_pos = loads(self.txns[split_id].get(f'episode_id_{idx}'.encode()))

        last_episode_id = -1
        for idx in tqdm(range(self.dataset_len - 1), desc="Write lang token id"):
            split_id = self.get_split_id(idx, self.max_steps)
            episode_id = loads(self.txns[split_id].get(f'episode_id_{idx}'.encode()))
            if episode_id != last_episode_id:
                insts = loads(self.txns[split_id].get(f'inst_{episode_id}'.encode()))
                inst_tokens = []
                for inst in insts:
                    inst_token = tokenizer(
                        inst,
                        return_tensors="pt",
                        padding=False,
                        max_length=None,
                        truncation=None,
                        return_token_type_ids=False,
                    )['input_ids'][0].numpy()
                    inst_tokens.append(inst_token)
                self.txns[split_id].put(
                    f'inst_token_{episode_id}'.encode(),
                    dumps(inst_tokens),
                )
                last_episode_id = episode_id
            
        self.close_lmdb()
    
    def show_flow(self, episode_id):
        # locate the start sample idx of the episode in the split
        split_id = self.get_split_id(episode_id, self.max_episode_ids)
        start_idx = self.min_steps[split_id]
        while True:
            cur_episode_id = self.get(start_idx, load_image=False, left=True)["episode_id"]
            print(f"idx {start_idx}, cur_episode_id {cur_episode_id}")
            if cur_episode_id != episode_id:
                start_idx += 1
            else:
                break

        left_video = []
        right_video = []
        gripper_flow_from_left = []
        gripper_flow_from_right = []
        idx = start_idx
        while True:
            left_data = self.get(idx, load_image=True, left=True)
            right_data = self.get(idx, load_image=True, left=False)
            if left_data["episode_id"] != episode_id:
                break
            else:
                idx += 1
                print(idx)
            left_video.append(left_data["rgb"].permute(1, 2, 0).numpy())
            right_video.append(right_data["rgb"].permute(1, 2, 0).numpy())
            gripper_flow_from_left.append(left_data["flow"])
            gripper_flow_from_right.append(right_data["flow"])
        
        # Save videos
        left_video = torch.from_numpy(np.stack(left_video)).permute(0, 3, 1, 2)
        right_video = torch.from_numpy(np.stack(right_video)).permute(0, 3, 1, 2)
        gripper_flow_from_left = np.stack(gripper_flow_from_left)
        gripper_flow_from_right = np.stack(gripper_flow_from_right)
        from cotracker.utils.visualizer import Visualizer
        vis = Visualizer(save_dir=".", pad_value=0, linewidth=3, tracks_leave_trace=-1)
        vis.visualize(left_video[None], torch.from_numpy(gripper_flow_from_left).reshape(1, -1, 1, 2), filename=f"left_{episode_id}")
        vis.visualize(right_video[None], torch.from_numpy(gripper_flow_from_right).reshape(1, -1, 1, 2), filename=f"right_{episode_id}")
   
class DroidFlowDataset(Dataset):

    def __init__(self, dataset_path, obs_horizon, chunk_size, start_ratio, end_ratio):
        self.obs_horizon = obs_horizon
        self.chunk_size = chunk_size
        self.reader = DroidReader(dataset_path)
        self.dummy_rgb = torch.zeros((obs_horizon, 1, 3) + RES, dtype=torch.uint8) # (b t v c h w)
        self.dummy_inst_token = torch.ones(MAX_LAN_TOKEN_NUM, dtype=torch.int) 
        self.dummy_flow = torch.zeros((chunk_size, 2)) 
        self.dummy_mask = torch.zeros(chunk_size)
        self.start_step = int(self.reader.dataset_len * start_ratio)
        self.end_step = int(self.reader.dataset_len * end_ratio) - chunk_size - obs_horizon
    
    def __len__(self):
        return self.end_step - self.start_step
    
    """
    def compute_limit(self):
        max_flow = torch.zeros(2)
        min_flow = torch.zeros(2)
        for idx in tqdm(range(10000), desc='Computing flow limits'):
            for left in [True, False]:
                flow = self.dummy_flow.clone()
                for action_idx in range(self.chunk_size):
                    data = self.reader.get(idx + action_idx, load_image=False, left=left)
                    if action_idx == 0:
                        flow_start = data["flow"]
                        episode_id = data["episode_id"]
                    if data["episode_id"] == episode_id:
                        flow[action_idx] = torch.from_numpy(data["flow"] - flow_start)
                max_flow = torch.max(max_flow, flow.max(dim=0)[0])
                min_flow = torch.min(min_flow, flow.min(dim=0)[0])
    """

    def __getitem__(self, idx):
        idx = idx + self.start_step

        rgb = self.dummy_rgb.clone()
        inst_token = self.dummy_inst_token.clone()
        flow = self.dummy_flow.clone()
        mask = self.dummy_mask.clone()
        
        if np.random.rand() < 0.5:
            left = True
        else:
            left = False

        data = self.reader.get(idx, load_image=False, left=left)
        episode_id = data["episode_id"]
        inst_token[:len(data["inst_token"])] = torch.from_numpy(data["inst_token"])

        for obs_idx in range(self.obs_horizon):
            data = self.reader.get(idx + obs_idx, load_image=True, left=left)
            if data["episode_id"] == episode_id:
                rgb[obs_idx, 0] = data["rgb"]
        
        cur_timestamp = self.obs_horizon - 1
        for action_idx in range(self.chunk_size):
            data = self.reader.get(idx + cur_timestamp + action_idx, load_image=False, left=left)
            if data["episode_id"] == episode_id:
                if action_idx == 0:
                    flow_start = data["flow"]
                flow[action_idx] = torch.from_numpy(data["flow"] - flow_start)
                mask[action_idx] = 1
        
        return {
            "rgb": rgb,
            "inst_token": inst_token,
            "action": flow,
            "flow_start": flow_start, 
            "mask": mask,
        }

if __name__ == "__main__":
    reader = DroidReader("/root/autodl-tmp/lmdb_droid")
    from transformers import AutoProcessor
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    tokenizer = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).tokenizer
    # TODO: reader.write_lang_token_id(tokenizer)
    # TODO: reader.rewrite_flow()
    reader.show_flow(29999)