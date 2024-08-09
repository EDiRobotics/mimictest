import torch 
from torch import nn
import mimictest.Nets.EfficientNetWithFiLM as EfficientNetWithFiLM
from mimictest.Nets.TokenLearner import TokenLearner
from mimictest.Nets.SimpleTransformer import Transformer

class RT1(nn.Module):
    def __init__(
            self,
            efficientnet_version,
            FiLM_cond_channel,
            lowdim_obs_num,
            num_actions,
            chunk_size,
            depth,
            vision_token_dim,
            ff_dim,
            n_heads,
            head_dim,
            max_T,
            token_learner_num_output_tokens,
            drop_prob,
            freeze_vision_tower,
            ):
        super().__init__()
        efficientnet_class = getattr(EfficientNetWithFiLM, efficientnet_version)
        self.vision_encoder = efficientnet_class(weights="DEFAULT", FiLM_cond_channel=FiLM_cond_channel)
        if freeze_vision_tower:
            for param in self.net.vision_encoder.parameters():
                param.requires_grad = False

        original_channels = self.vision_encoder.features[-1][1].weight.shape[0]
        self.vision_pre_proj = nn.Conv2d(original_channels, vision_token_dim, 1)
        self.token_learner = TokenLearner(
            dim = vision_token_dim,
            num_output_tokens = token_learner_num_output_tokens,
        )
        self.vision_after_proj = nn.Linear(vision_token_dim, ff_dim)
        self.lowdim_encoder = nn.Linear(lowdim_obs_num, ff_dim)
        self.lan_encoder = nn.Linear(FiLM_cond_channel, ff_dim)
        self.action_query = nn.Parameter(torch.zeros(chunk_size, ff_dim)) 
        self.BERT_dec = Transformer(
            ff_dim = ff_dim, 
            head_dim = head_dim,
            n_heads = n_heads, 
            n_blocks = depth, 
            max_T = max_T, 
            drop_p = drop_prob,
            causal = False,
        )
        self.to_actions = nn.Linear(ff_dim, num_actions)
        self.chunk_size = chunk_size
        self.num_actions = num_actions
        self.dummy_text_embeds = nn.Parameter(torch.zeros(1, FiLM_cond_channel), requires_grad=False)

    def forward(self, rgbs, low_dim, text_embeds=None):
        if text_embeds == None:
            text_embeds = self.dummy_text_embeds

        B, T, V, C, H, W = rgbs.shape
        rgbs = rgbs.view(B*T*V, C, H, W)
        rgb_tokens = self.vision_encoder(rgbs, text_embeds.expand(B*T*V, -1)) 
        rgb_tokens = self.vision_pre_proj(rgb_tokens)
        rgb_tokens = self.token_learner(rgb_tokens) 
        B_T_V, C, N = rgb_tokens.shape
        rgb_tokens = rgb_tokens.reshape(B, T, V, C, N).transpose(3, 4).reshape(B, T*V*N, C)
        rgb_tokens = self.vision_after_proj(rgb_tokens)

        low_dim_token = self.lowdim_encoder(low_dim) # (b t c)

        lan_token = self.lan_encoder(text_embeds.expand(B, -1))
        B, C = lan_token.shape
        lan_token = lan_token.view(B, 1, C)

        T, C = self.action_query.shape
        action_query = self.action_query.expand(B, T, C) # (t c) -> (b t c)

        input_tokens = torch.cat((rgb_tokens, low_dim_token, lan_token, action_query), dim=1)
        output_tokens = self.BERT_dec(input_tokens)
        actions = self.to_actions(output_tokens[:, -self.chunk_size:])
        return actions