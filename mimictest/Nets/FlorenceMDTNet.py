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
            max_T,
            ff_dim,
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
        
        # not update language embedding for training speed, may harm performance
        for param in self.net.language_model.model.shared.parameters():
            param.requires_grad = False
        for param in self.net.language_model.model.encoder.embed_positions.parameters():
            param.requires_grad = False
        for param in self.net.language_model.model.decoder.embed_positions.parameters():
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
        self.LLM_output_projector = nn.Linear(token_dim, ff_dim)
        self.action_query = nn.Parameter(torch.zeros(1, num_action_query, token_dim))
        self.action_encoder = nn.Linear(num_actions, ff_dim)
        self.action_decoder = nn.Sequential(
            nn.Linear(ff_dim, num_actions),
        )
        if lowdim_obs_dim > 0:
            self.low_dim_encoder = nn.Linear(lowdim_obs_dim, ff_dim)
        self.noise_pred_net = TransformerFiLMDecoder(
            embed_dim=ff_dim, 
            max_T=max_T,
            n_heads=n_heads, 
            attn_pdrop=attn_pdrop, 
            resid_pdrop=resid_pdrop,
            n_layers=n_layers, 
            mlp_pdrop=mlp_pdrop,
        ) 

    def forward(self, batch):
        if batch['obs_features'] is None:
            B, T, V, C, H, W = batch['rgb'].shape
            rgb = batch['rgb'].view(B*T*V, C, H, W)
            rgb_features = self.net._encode_image(rgb)
            B_T_V, N, D = rgb_features.shape
            rgb_features = rgb_features.view(B, T*V*N, D)
            
            text_embeds = self.prompt_embeds.repeat(B, 1, 1) # (b n d)
            if "inst_token" in batch:
                inst_embeds = self.net.get_input_embeddings()(batch["inst_token"]) 
                text_embeds = torch.cat((text_embeds, inst_embeds), dim=1)
            inputs_embeds, attention_mask = self.net._merge_input_ids_with_image_features(rgb_features, text_embeds)

            action_query = self.action_query.repeat(B, 1, 1) # (b, num_action_query, d)

            decoder_outputs_embeds = self.net.language_model(
                inputs_embeds = inputs_embeds,
                decoder_inputs_embeds = action_query,
                output_hidden_states = True,
            )['decoder_hidden_states'][-1]
            obs_features = self.LLM_output_projector(decoder_outputs_embeds)

            if "low_dim" in batch:
                low_dim = self.low_dim_encoder(batch['low_dim']) # (b, t, d)
                obs_features = torch.cat((obs_features, low_dim), dim=1)
        else:
            obs_features = batch['obs_features']

        # predict the noise residual
        pred_noise = {}
        noisy_input = batch['noisy_inputs']['action']
        noisy_input = self.action_encoder(noisy_input)
        out = self.noise_pred_net(
            noisy_input, batch['timesteps'], cond=obs_features)
        pred_noise['action'] = self.action_decoder(out)
        return pred_noise, obs_features