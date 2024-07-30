import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

class Attention(nn.Module):
    def __init__(self, ff_dim, head_dim, max_T, n_heads, drop_p, causal=False):
        # max_T: maximal sequence length
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_T = max_T
        self.causal = causal
        self.q_net = nn.Linear(ff_dim, head_dim * n_heads)
        self.k_net = nn.Linear(ff_dim, head_dim * n_heads)
        self.v_net = nn.Linear(ff_dim, head_dim * n_heads)
        self.proj_net = nn.Linear(head_dim * n_heads, ff_dim)
        self.drop_p = drop_p

    def forward(self, x):
        B, T, _ = x.shape # batch size, seq length, ff_dim
        E, D = self.n_heads, self.head_dim

        # Divide the tensors for multi head dot product
        q = self.q_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d
        k = self.k_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d
        v = self.v_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d

        inner = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_p, is_causal=self.causal)
        inner = inner.transpose(1, 2).contiguous().view(B, T, E * D) # b e t d -> b t (e d) Combine results from multi heads
        return self.proj_net(inner)

class Block(nn.Module):
    def __init__(self, ff_dim, head_dim, max_T, n_heads, drop_p, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(ff_dim)
        self.attn = Attention(ff_dim, head_dim, max_T, n_heads, drop_p, causal)
        self.ln2 = nn.LayerNorm(ff_dim)
        self.ff = nn.Sequential(
            nn.Linear(ff_dim, ff_dim * 4),
            nn.GELU(),
            nn.Linear(ff_dim * 4, ff_dim),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, ff_dim, head_dim, n_heads, n_blocks, max_T, drop_p, causal):
        super().__init__()
        self.blocks = nn.ModuleList([Block(ff_dim, head_dim, max_T, n_heads, drop_p, causal) for _ in range(n_blocks)])

        devisor = 1 / torch.sqrt(torch.tensor(ff_dim, dtype=torch.float32))
        self.pos_emb = nn.Parameter(torch.randn(1, max_T, ff_dim) * devisor)

        self.apply(self._init_module)

    def _init_module(self, module):
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            constant_(module.bias, 0)
            constant_(module.weight, 1.0)

    def forward(self, x):
        B, T, C = x.shape # B: batch size, T: sequence length, C: token dim
        x += self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        return x
