import torch
from torchvision.transforms.v2 import Resize, RandomCrop, CenterCrop, ColorJitter
import mimictest.Utils.RotationConversions as rot

def action_euler_to_6d(action):
    rot_euler = action[..., 3:6]
    rot_mat = rot.euler_angles_to_matrix(rot_euler, 'XYZ')
    rot_6d = rot.matrix_to_rotation_6d(rot_mat)
    new_action = torch.cat((action[..., :3], rot_6d, action[..., -1:]), dim=-1)
    return new_action

def action_axis_to_6d(action):
    rot_axis = action[..., 3:6]
    rot_mat = rot.axis_angle_to_matrix(rot_axis)
    rot_6d = rot.matrix_to_rotation_6d(rot_mat)
    new_action = torch.cat((action[..., :3], rot_6d, action[..., -1:]), dim=-1)
    return new_action

def action_6d_to_euler(action):
    rot_6d = action[..., 3:9]
    rot_mat = rot.rotation_6d_to_matrix(rot_6d)
    rot_euler = rot.matrix_to_euler_angles(rot_mat, 'XYZ')
    new_action = torch.cat((action[..., :3], rot_euler, action[..., -1:]), dim=-1)
    return new_action

def action_6d_to_axis(action):
    rot_6d = action[..., 3:9]
    rot_mat = rot.rotation_6d_to_matrix(rot_6d)
    rot_axis = rot.matrix_to_axis_angle(rot_mat)
    new_action = torch.cat((action[..., :3], rot_axis, action[..., -1:]), dim=-1)
    return new_action

class PreProcess(): 
    def __init__(
            self,
            desired_rgb_shape, 
            crop_shape,
            low_dim_max,
            low_dim_min,
            action_max, 
            action_min, 
            enable_6d_rot,
            abs_mode,
            device,
        ):
        self.train_transforms = torch.nn.Sequential(
            Resize([desired_rgb_shape, desired_rgb_shape], antialias=True),
            RandomCrop((crop_shape, crop_shape)),
        )
        self.eval_transforms = torch.nn.Sequential(
            Resize([desired_rgb_shape, desired_rgb_shape], antialias=True),
            CenterCrop((crop_shape, crop_shape)),
        )
        self.action_min = action_min.to(device)
        self.action_max = action_max.to(device)
        self.low_dim_min = low_dim_min.to(device)
        self.low_dim_max = low_dim_max.to(device)
        self.enable_6d_rot = enable_6d_rot
        self.abs_mode = abs_mode
    
    def rgb_process(self, rgb, train=True):
        '''
            Input:
                rgb:    (..., c, h, w)
                        dtype = uint8

            Output:
                rgb:    (..., c, h, w)
                        dtype = float32
        '''
        if train:
            rgb = self.train_transforms(rgb)
        else:
            rgb = self.eval_transforms(rgb)
        rgb = rgb.float()*(1/255.)
        return rgb

    def low_dim_normalize(self, low_dim):
        return (low_dim - self.low_dim_min) / (self.low_dim_max - self.low_dim_min)
    
    def action_normalize(self, action):
        if self.enable_6d_rot:
            if self.abs_mode:
                action = action_axis_to_6d(action)
            else:
                action = action_euler_to_6d(action)
        return (action - self.action_min) / (self.action_max - self.action_min)

    def action_back_normalize(self, action):
        action = action * (self.action_max - self.action_min) + self.action_min
        if self.enable_6d_rot:
            if self.abs_mode:
                action = action_6d_to_axis(action)
            else:
                action = action_6d_to_euler(action)
        return action