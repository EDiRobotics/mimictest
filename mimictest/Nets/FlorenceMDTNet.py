import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
from mimictest.Nets.MDT import TransformerFiLMDecoder

class FlorenceMDTNet(nn.Module):
    def __init__(
            self,
            path,
            freeze_vision_tower,
            freeze_florence,
            num_actions,
            num_action_query,
            lowdim_obs_dim,
            n_heads, 
            attn_pdrop, 
            resid_pdrop,
            n_layers, 
            mlp_pdrop,
        ):
        super().__init__()

        self.net = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        if freeze_vision_tower:
            for param in self.net.vision_tower.parameters():
                param.requires_grad = False
        if freeze_florence:
            for param in self.net.parameters():
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
        self.action_query = nn.Parameter(torch.zeros(1, num_action_query, token_dim))
        self.action_encoder = nn.Linear(num_actions, token_dim)
        self.action_decoder = nn.Linear(token_dim, num_actions)
        self.low_dim_encoder = nn.Linear(lowdim_obs_dim, token_dim)
        self.noise_pred_net = TransformerFiLMDecoder(
            embed_dim=token_dim, 
            n_heads=n_heads, 
            attn_pdrop=attn_pdrop, 
            resid_pdrop=resid_pdrop,
            n_layers=n_layers, 
            mlp_pdrop=mlp_pdrop,
        ) 

    def forward(self, rgb, low_dim, noisy_actions, timesteps, obs_features=None):
        if obs_features is None:
            B, T, V, C, H, W = rgb.shape
            rgb = rgb.view(B*T*V, C, H, W)
            rgb_features = self.net._encode_image(rgb)
            B_T_V, N, D = rgb_features.shape
            rgb_features = rgb_features.view(B, T*V*N, D)
            
            text_embeds = self.prompt_embeds.repeat(B, 1, 1) # (b n d)
            inputs_embeds, attention_mask = self.net._merge_input_ids_with_image_features(rgb_features, text_embeds)

            low_dim = self.low_dim_encoder(low_dim) # (b, t, d)
            action_query = self.action_query.repeat(B, 1, 1) # (b, num_action_query, d)
            num_action_query = action_query.shape[1]

            decoder_inputs_embeds = torch.cat((low_dim, action_query), dim = 1)
            decoder_outputs_embeds = self.net.language_model(
                inputs_embeds = inputs_embeds,
                decoder_inputs_embeds = decoder_inputs_embeds,
                output_hidden_states = True,
            )['decoder_hidden_states'][-1]
            obs_features = decoder_outputs_embeds[:, -num_action_query:] # (b, num_action_query, d)

        # predict the noise residual
        noisy_actions = self.action_encoder(noisy_actions)
        pred = self.noise_pred_net(
            noisy_actions, timesteps, cond=obs_features)
        pred = self.action_decoder(pred)

        return pred, obs_features 
