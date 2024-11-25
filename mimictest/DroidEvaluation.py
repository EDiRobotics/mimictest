import os
import math
import numpy as np
import torch
from einops import rearrange
import matplotlib.pyplot as plt

PLOT_IMG_NUM = 4

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
        rgb = rearrange(batch["rgb"][:PLOT_IMG_NUM], "b 1 1 c h w -> b h w c").to(dtype=torch.uint8, device="cpu").numpy()
        flow_start = batch["flow_start"][:PLOT_IMG_NUM, None]
        dataset_action = (batch["action"][:PLOT_IMG_NUM] + flow_start).cpu().numpy()
        batch = self.preprcs.process(batch, train=False)
        with torch.no_grad():
            pred = policy.infer(batch)
        loss = torch.nn.functional.l1_loss(pred["action"], batch["action"])
        pred = self.preprcs.back_process(pred)
        pred_action = (pred["action"][:PLOT_IMG_NUM] + flow_start).cpu().numpy()

        if record_video and acc.is_main_process:
            fig, axs = plt.subplots(nrows=2, ncols=PLOT_IMG_NUM, figsize=(9*PLOT_IMG_NUM, 9*2))
            for image_idx in range(PLOT_IMG_NUM):
                axs[0, image_idx].imshow(rgb[image_idx])
                axs[0, image_idx].plot(
                    dataset_action[image_idx, :, 0], 
                    dataset_action[image_idx, :, 1], 
                    color='yellow', linewidth=5,
                )
                axs[0, image_idx].axis('off')

                axs[1, image_idx].imshow(rgb[image_idx])
                axs[1, image_idx].plot(
                    pred_action[image_idx, :, 0], 
                    pred_action[image_idx, :, 1], 
                    color='yellow', linewidth=5,
                )
                axs[1, image_idx].axis('off')

            wandb_tracker = acc.get_tracker("wandb")
            wandb_tracker.log({f"vis": fig}, commit=False)
            fig.savefig(self.save_path/f"vis_{batch_idx}.png")
            plt.close(fig)

        return loss 
