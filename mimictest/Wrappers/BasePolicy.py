import os
import json
import torch
import wandb

class BasePolicy():
    def __init__(
            self,
            net,
            loss_func,
            do_compile,
        ):
        self.do_compile = do_compile
        if do_compile:
            self.net = torch.compile(net)
        else:
            self.net = net 
        self.use_ema = False
        self.loss_func = loss_func

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.net.parameters()))
        )
    
    def compile(self, cache_size_limit=128):
        torch._dynamo.config.cache_size_limit = cache_size_limit
        torch._dynamo.config.optimize_ddp = False  # https://github.com/pytorch/pytorch/issues/104674
        # TODO: https://github.com/pytorch/pytorch/issues/109774#issuecomment-2046633776
        self.net = torch.compile(self.net)
    
    def parameters(self):
        return filter(lambda p: p.requires_grad, self.net.parameters())

    def save_pretrained(self, acc, path, epoch_id):
        acc.wait_for_everyone()
        if hasattr(acc.unwrap_model(self.net), '_orig_mod'): # the model has been compiled
            ckpt = {"net": acc.unwrap_model(self.net)._orig_mod.state_dict()}
        else:
            ckpt = {"net": acc.unwrap_model(self.net).state_dict()}
        acc.save(ckpt, path / f'policy_{epoch_id}.pth')

    def load_pretrained(self, acc, path, load_epoch_id):
        if os.path.isfile(path / f'policy_{load_epoch_id}.pth'):
            ckpt = torch.load(path / f'policy_{load_epoch_id}.pth', map_location='cpu')
            if self.do_compile:
                missing_keys, unexpected_keys = self.net._orig_mod.load_state_dict(ckpt["net"], strict=False)
            else:
                missing_keys, unexpected_keys = self.net.load_state_dict(ckpt["net"], strict=False)
            acc.print('load ', path / f'policy_{load_epoch_id}.pth', '\nmissing ', missing_keys, '\nunexpected ', unexpected_keys)
        else: 
            acc.print(path / f'policy_{load_epoch_id}.pth', 'does not exist. Initialize new checkpoint')

    def load_wandb(self, acc, path, do_watch_parameters, save_interval):
        if os.path.isfile(path / "wandb_id.json"):
            run_id = json.load(open(path / "wandb_id.json", "r"))
            acc.init_trackers(
                project_name="mimictest", 
                init_kwargs={"wandb": {"id": run_id, "resume": "allow"}}
            )
            if acc.is_main_process:
                if do_watch_parameters:
                    wandb.watch(self.net, log="all", log_freq=save_interval)
        else: 
            acc.init_trackers(project_name="mimictest")
            if acc.is_main_process:
                tracker = acc.get_tracker("wandb")
                json.dump(tracker.run.id, open(path / "wandb_id.json", "w"))
                if do_watch_parameters:
                    wandb.watch(self.net, log="all", log_freq=save_interval)

    def compute_loss(self, rgb, low_dim, actions):
        pred = self.net(rgb, low_dim)
        loss = self.loss_func(pred, actions, reduction='none')
        return loss.sum(dim=(-1,-2)).mean()

    def infer(self, rgb, low_dim):
        pred_action = self.net(rgb, low_dim)
        return pred_action
