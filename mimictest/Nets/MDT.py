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
            block_size: int,
            causal: bool = False,
            bias=False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            rotary_emb_dim = None,
            rotary_xpos_scale_base = 512,
            rotary_interpolation_factor = 1.,
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
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))
        self.use_rot_embed = use_rot_embed
        if self.use_rot_embed:
        # Update (12/2022): Rotary embedding has since been hugely successful, widely adopted in many large language models, including the largest in the world, PaLM. 
        # However, it has been uncovered in the ALiBi paper that rotary embeddings cannot length extrapolate well. 
        # This was recently addressed in <a href="https://arxiv.org/abs/2212.10554v1">a Microsoft research paper</a>. 
        # They propose a way to unobtrusively add the same decay as in ALiBi, and found that this resolves the extrapolation problem.
        # You can use it in this repository by setting `rotary_xpos = True`. Like ALiBi, it would enforce the attention to be local. You can set the receptive field with `rotary_xpos_scale_base` value, which defaults to `512`
            rotary_emb_dim = max(default(rotary_emb_dim, self.n_head // 2), 32)
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_emb_dim, 
                use_xpos = rotary_xpos, 
                xpos_scale_base = rotary_xpos_scale_base, 
                interpolate_factor = rotary_interpolation_factor, 
            ) 

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

        # apply rotary stuff here if needed:
        if self.use_rot_embed:
            q = self.rotary_pos_emb.rotate_queries_or_keys(q)
            k = self.rotary_pos_emb.rotate_queries_or_keys(k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=custom_attn_mask, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                if custom_attn_mask is not None:
                    att = att.masked_fill(custom_attn_mask == 0, float('-inf'))
                else:
                    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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
            block_size: int, 
            causal: bool,
            use_cross_attention: bool = False,
            use_rot_embed: bool=False,
            rotary_xpos: bool = False,
            bias: bool = False, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, use_rot_embed, rotary_xpos)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, use_rot_embed, rotary_xpos)
            self.ln3 = nn.LayerNorm(n_embd)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, mlp_pdrop)

    def forward(self, x, context=None, custom_attn_mask=None):
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
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
            block_size, 
            causal, 
            film_cond_dim,
            use_cross_attention=False, 
            use_rot_embed=False, 
            rotary_xpos=False, 
            bias=False # and any other arguments from the Block class
        ):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal,
                         use_cross_attention=use_cross_attention, 
                         use_rot_embed=use_rot_embed, 
                         rotary_xpos=rotary_xpos, 
                         bias=bias)
        self.adaLN_zero = AdaLNZero(film_cond_dim)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(c)
        
        # Attention with modulation
        x_attn = self.ln_1(x)
        x_attn = modulate(x_attn, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_attn, custom_attn_mask=custom_attn_mask)
        
        # Cross attention if used
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context, custom_attn_mask=custom_attn_mask)
        
        # MLP with modulation
        x_mlp = self.ln_2(x)
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
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
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
            block_size,
            causal=True, 
            use_cross_attention=use_cross_attention,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias,
            film_cond_dim=embed_dim,
            ) 
            for _ in range(n_layers)]
        )
        self.time_emb = SinusoidalPosEmb(embed_dim)
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, c, cond=None, custom_attn_mask=None):
        c = self.time_emb(c).unsqueeze(1)
        for layer in self.blocks:
            x = layer(x, c, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x
