import os
import math
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM

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

class FlorencePi0Net(nn.Module):
    def __init__(
            self,
            path,
            lowdim_obs_dim,
            num_actions,
            freeze_vision_tower,
        ):
        super().__init__()

        self.net = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        if freeze_vision_tower:
            for param in self.net.vision_tower.parameters():
                param.requires_grad = False

        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.tokenizer = AutoProcessor.from_pretrained(path, trust_remote_code=True).tokenizer
        prompt_token_ids = self.tokenizer(
            "<Action>",
            return_tensors="pt",
            padding=False,
            max_length=None,
            truncation=None,
            return_token_type_ids=False,
        )['input_ids']
        self.prompt_embeds = nn.Parameter(self.net.get_input_embeddings()(prompt_token_ids), requires_grad=False)

        token_dim = self.net.language_model.model.decoder.embed_tokens.embedding_dim
        self.low_dim_encoder = nn.Linear(lowdim_obs_dim, token_dim)
        self.action_encoder = nn.Linear(num_actions, token_dim)
        self.action_timestep_mixer = nn.Sequential(
            nn.Linear(2*token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(token_dim, num_actions),
        )
        self.time_emb = SinusoidalPosEmb(token_dim)

    def forward(self, batch):
        if batch['obs_features'] is None:
            B, T, V, C, H, W = batch['rgb'].shape
            rgb = batch['rgb'].view(B*T*V, C, H, W)
            rgb_features = self.net._encode_image(rgb)
            B_T_V, N, D = rgb_features.shape
            rgb_features = rgb_features.view(B, T*V*N, D)
            
            text_embeds = self.prompt_embeds.repeat(B, 1, 1) # (b n d)
            inputs_embeds, attention_mask = self.net._merge_input_ids_with_image_features(rgb_features, text_embeds)

            obs_features = inputs_embeds
        else:
            inputs_embeds = batch['obs_features']
            obs_features = batch['obs_features']

        low_dim = self.low_dim_encoder(batch['low_dim']) # (b, t, d)

        noisy_actions = self.action_encoder(batch['noisy_inputs']['action'])
        B, T, D = noisy_actions.shape
        time_emb = self.time_emb(batch["timesteps"]).unsqueeze(1).repeat(1, T, 1) # (b t d)
        noisy_actions = torch.cat((noisy_actions, time_emb), dim=-1) # (b t 2d)
        noisy_actions = self.action_timestep_mixer(noisy_actions) # (b t d)

        decoder_inputs_embeds = torch.cat((low_dim, noisy_actions), dim=1)
        decoder_outputs_embeds = self.net.language_model(
            inputs_embeds = inputs_embeds,
            decoder_inputs_embeds = decoder_inputs_embeds,
            output_hidden_states = True,
        )['decoder_hidden_states'][-1]

        # predict the noise residual
        pred_noise = {}
        pred_noise['action'] = self.action_decoder(decoder_outputs_embeds[:, -T:])
        return pred_noise, obs_features