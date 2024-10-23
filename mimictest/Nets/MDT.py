# Code copied from https://github.com/intuitive-robots/mdt_policy/blob/main/mdt/models/networks/transformers/transformer_blocks.py

import math 
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Attention(nn.Module):
    def __init__(
            self, 
            n_embd: int,
            n_head: int,
            attn_pdrop: float,
            resid_pdrop: float,
            causal: bool = False,
            bias=True,
        ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal

    def forward(self, x, context=None, custom_attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # if the context is not None we do cross-attention othberwise self=attention
        # cross attention computes the query from x and the keys and values are from the context
        if context is not None:
            k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=custom_attn_mask, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):
    def __init__(
            self, 
            n_embd: int,
            bias: bool,
            dropout: float = 0
        ):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(
            self, 
            n_embd: int, 
            n_heads: int, 
            attn_pdrop: float, 
            resid_pdrop: float, 
            mlp_pdrop: float,
            causal: bool,
            use_cross_attention: bool = False,
            bias: bool = True, # True: bias in Linears
        ):
        super().__init__()
        self.norm_1 = LayerNorm(n_embd, bias)
        self.attn = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, causal, bias)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, causal, bias)
            self.norm_3 = LayerNorm(n_embd, bias)
        self.norm_2 = LayerNorm(n_embd, bias)
        self.mlp = MLP(n_embd, bias, mlp_pdrop)

    def forward(self, x, context=None, custom_attn_mask=None):
        x = x + self.attn(self.norm_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.norm_3(x), context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.norm_2(x))
        return x

class AdaLNZero(nn.Module):
    """
    AdaLN-Zero modulation for conditioning.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, c):
        return self.modulation(c).chunk(6, dim=-1)

def modulate(x, shift, scale):
    return shift + (x * (scale))

class ConditionedBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop, 
            causal, 
            film_cond_dim,
            use_cross_attention=False, 
            bias=False # and any other arguments from the Block class
        ):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, causal,
                         use_cross_attention=use_cross_attention, 
                         bias=bias)
        self.adaLN_zero = AdaLNZero(film_cond_dim)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(c)
        
        # Attention with modulation
        x_attn = self.norm_1(x)
        x_attn = modulate(x_attn, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_attn, custom_attn_mask=custom_attn_mask)
        
        # Cross attention if used
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.norm_3(x), context, custom_attn_mask=custom_attn_mask)
        
        # MLP with modulation
        x_mlp = self.norm_2(x)
        x_mlp = modulate(x_mlp, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)
        
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerFiLMDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            max_T: int,
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            bias: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConditionedBlock(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            causal=True, 
            use_cross_attention=use_cross_attention,
            bias=bias,
            film_cond_dim=embed_dim,
            ) 
            for _ in range(n_layers)]
        )
        self.time_emb = SinusoidalPosEmb(embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_T, embed_dim))
        self.norm = LayerNorm(embed_dim, bias)

    def forward(self, x, c, cond=None, custom_attn_mask=None):
        c = self.time_emb(c).unsqueeze(1)
        B, T, D = x.shape
        x = torch.cat((cond, x), dim=1)
        x += self.pos_emb[:, :x.shape[1]]
        for layer in self.blocks:
            x = layer(x, c, custom_attn_mask=custom_attn_mask)
        x = self.norm(x[:, -T:])
        return x
