import torch
from torchvision.transforms.v2 import Resize, RandomCrop, CenterCrop, ColorJitter
import mimictest.Utils.RotationConversions as rot

def action_euler_to_6d(rot_euler):
    rot_mat = rot.euler_angles_to_matrix(rot_euler, 'XYZ')
    rot_6d = rot.matrix_to_rotation_6d(rot_mat)
    return rot_6d

def action_axis_to_6d(rot_axis):
    rot_mat = rot.axis_angle_to_matrix(rot_axis)
    rot_6d = rot.matrix_to_rotation_6d(rot_mat)
    return rot_6d

def action_6d_to_euler(rot_6d):
    rot_mat = rot.rotation_6d_to_matrix(rot_6d)
    rot_euler = rot.matrix_to_euler_angles(rot_mat, 'XYZ')
    return rot_euler

def action_6d_to_axis(rot_6d):
    rot_mat = rot.rotation_6d_to_matrix(rot_6d)
    rot_axis = rot.matrix_to_axis_angle(rot_mat)
    return rot_axis

class PreProcess(): 
    def __init__(
            self,
            process_configs,
            device,
        ):
        self.configs = process_configs
        for key in self.configs:
            if 'rgb_shape' in self.configs[key]:
                self.configs[key]['train_transforms'] = torch.nn.Sequential(
                    Resize(self.configs[key]['rgb_shape'], antialias=True),
                    RandomCrop(self.configs[key]['crop_shape']),
                )
                self.configs[key]['eval_transforms'] = torch.nn.Sequential(
                    Resize(self.configs[key]['rgb_shape'], antialias=True),
                    RandomCrop(self.configs[key]['crop_shape']),
                )
            if "max" in self.configs[key]:
                self.configs[key]['max'] = self.configs[key]['max'].to(device)
                self.configs[key]['min'] = self.configs[key]['min'].to(device)
    
    def process(self, batch, train=False):
        for key in batch:
            if 'rgb_shape' in self.configs[key]: # image data
                if train:
                    batch[key] = self.configs[key]['train_transforms'](batch[key])
                else:
                    batch[key] = self.configs[key]['eval_transforms'](batch[key])
                batch[key] = batch[key].float() / 255.
            if 'enable_6d_rot' in self.configs[key]:
                if self.configs[key]['abs_mode']:
                    rot_axis = batch[key][..., 3:6]
                    rot_6d = action_axis_to_6d(rot_axis)
                else:
                    rot_euler = batch[key][..., 3:6]
                    rot_6d = action_axis_to_6d(rot_euler)
                batch[key] = torch.cat((batch[key][..., :3], rot_6d, batch[key][..., 6:]), dim=-1)
            if "max" in self.configs[key]:
                batch[key] = (batch[key] - self.configs[key]['min']) / (self.configs[key]['max'] - self.configs[key]['min'])
                batch[key] = batch[key] * 2 - 1 # from (0, 1) to (-1, 1)
        return batch

    def back_process(self, batch):
        for key in batch:
            if "max" in self.configs[key]:
                batch[key] = (batch[key] + 1) * 0.5 # from (-1, 1) to (0, 1)
                batch[key] = batch[key] * (self.configs[key]['max'] - self.configs[key]['min']) + self.configs[key]['min']
            if 'rgb_shape' in self.configs[key]: # image data
                batch[key] = torch.clamp(batch[key], 0, 1)
                batch[key] = batch[key] * 255.
            if 'enable_6d_rot' in self.configs[key]:
                rot_6d = batch[key][..., 3:9]
                if self.configs[key]['abs_mode']:
                    rot_axis = action_6d_to_axis(rot_6d)
                    batch[key] = torch.cat((batch[key][..., :3], rot_axis, batch[key][..., 9:]), dim=-1)
                else:
                    rot_euler = action_6d_to_euler(rot_6d)
                    batch[key] = torch.cat((batch[key][..., :3], rot_euler, batch[key][..., 9:]), dim=-1)
            if 'binary' in self.configs[key]:
                batch[key] = torch.nn.Sigmoid()(batch[key])
                batch[key] = batch[key] > 0.5
                batch[key] = batch[key].int().float()
                batch[key] = batch[key] * 2.0 - 1.0

        if "arm_action" in batch and "gripper_action" in batch:
            batch['action'] = torch.cat((
                batch['arm_action'],
                batch['gripper_action'],
            ), dim=-1)
        elif "arm_action" in batch: # no gripper action
            batch['action'] = batch['arm_action']
        elif "action" in batch: # no gripper action
            batch['action'] = batch['action']
        return batch
