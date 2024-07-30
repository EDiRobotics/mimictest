import torch

class SimplePolicy():
    def __init__(
            self,
            net,
            loss_func = None,
        ):
        self.net = net 
        self.use_ema = False
        self.loss_func = loss_func

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.net.parameters()))
        )

    def save_pretrained(self, acc, path):
        acc.wait_for_everyone()
        ckpt = {
            "net": acc.unwrap_model(self.net).state_dict(),
        }
        acc.save(ckpt, path)

    def load_pretrained(self, acc, path):
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt["net"])
        acc.print('load ', path)

    def compute_loss(self, rgb, low_dim, actions):
        pred = self.net(rgb, low_dim)
        loss = self.loss_func(pred, actions, reduction='none')
        return loss.sum(dim=(-1,-2)).mean()

    def infer(self, rgb, low_dim):
        pred_action = self.net(rgb, low_dim)
        return pred_action
