import os
import math
import numpy as np
import torch
from einops import rearrange

class Evaluation():
    def __init__(self, preprcs, prefetcher, save_path, device):
        self.preprcs = preprcs
        self.prefetcher = prefetcher
        self.save_path = save_path
        self.device = device
        return None

    def evaluate_on_env(self, acc, policy, batch_idx, num_eval_ep, max_test_ep_len, record_video=False):
        if policy.use_ema:
            policy.copy_ema_to_ema_net()
            policy.ema_net.eval()
        else:
            policy.net.eval()

        batch, _ = self.prefetcher.next_without_none()
        batch = self.preprcs.process(batch, train=False)
        with torch.no_grad():
            pred = policy.infer(batch)
        loss = torch.nn.functional.l1_loss(pred["action"], batch["action"])

        return loss 
