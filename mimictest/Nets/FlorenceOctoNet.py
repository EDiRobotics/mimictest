import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
from mimictest.Nets.Chi_Transformer import TransformerForDiffusion

class FlorenceOctoNet(nn.Module):
    def __init__(
            self,
            path,
            freeze_vision_tower,
            lowdim_obs_dim,
            num_actions,
            max_T,
            n_layer,
            n_head,
            n_emb,
            p_drop_emb,
            p_drop_attn,
            causal_attn,
            time_as_cond,
            obs_as_cond,
            n_cond_layers,
        ):
        super().__init__()
        self.freeze_vision_tower = freeze_vision_tower

        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.tokenizer = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True).tokenizer
        self.net = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, local_files_only=True)
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
        self.action_query = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.noise_pred_net = TransformerForDiffusion(
            input_dim=num_actions,
            output_dim=num_actions,
            max_T=max_T, 
            n_obs_steps=1,
            cond_dim=token_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
        )

    def forward(self, rgb, low_dim, noisy_actions, timesteps, obs_features=None, text=""):
        if obs_features is None:
            B, T, V, C, H, W = rgb.shape
            rgb = rgb.view(B*T*V, C, H, W)
            if self.freeze_vision_tower:
                with torch.no_grad():
                    rgb_features = self.net._encode_image(rgb)
            else:
                rgb_features = self.net._encode_image(rgb)
            B_T_V, N, D = rgb_features.shape
            rgb_features = rgb_features.view(B, T*V*N, D)

            text_token_ids = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                max_length=None,
                truncation=None,
                return_token_type_ids=False,
            )['input_ids'].to(rgb.device)
            text_embeds = self.net.get_input_embeddings()(text_token_ids)
            text_embeds = torch.cat((self.prompt_embeds, text_embeds), dim=1).repeat(B, 1, 1) # (b n d)
            inputs_embeds, attention_mask = self.net._merge_input_ids_with_image_features(rgb_features, text_embeds)

            low_dim = self.low_dim_encoder(low_dim) # (b, t, d)
            action_query = self.action_query.repeat(B, 1, 1) # (b, 1, d)
            decoder_inputs_embeds = torch.cat((low_dim, action_query), dim = 1)

            decoder_outputs_embeds = self.net(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                decoder_inputs_embeds = decoder_inputs_embeds,
                output_hidden_states = True,
            )['decoder_hidden_states'][-1]
            obs_features = decoder_outputs_embeds[:, -1:] # (b, 1, d)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, cond=obs_features)

        return noise_pred, obs_features 
