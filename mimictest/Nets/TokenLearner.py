import torch
from torch import nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
            self,
            *,
            dim,
            ff_mult = 0.125,
            num_output_tokens = 8,
            ):
        super().__init__()
        inner_dim = int(dim * ff_mult)
        self.num_output_tokens = num_output_tokens
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU(),
                nn.Linear(inner_dim, num_output_tokens),
                ) 
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.transpose(1, 3).reshape(B, H*W, C) # (b c h w) -> (b (h w) c)
        x = self.norm(x)
        attn = self.net(x)
        attn = F.softmax(attn, dim=-2)
        B, HW, N = attn.shape
        attn = attn.view(B, HW, 1, N) # (b hw n) -> (b hw 1 n) dimension 1 can be broadcast
        x = x.unsqueeze(-1).expand(B, HW, C, self.num_output_tokens) # (b hw c) -> (b hw c n)
        x = (x * attn).mean(dim=1) # (b hw c n) -> (b c n) spatial pooling
        return x
