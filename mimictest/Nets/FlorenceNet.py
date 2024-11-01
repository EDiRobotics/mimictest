import os
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM

class FlorenceNet(nn.Module):
    def __init__(
            self,
            path,
            lowdim_obs_dim,
            num_actions,
            chunk_size,
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
        self.to_action = nn.Linear(token_dim, num_actions)
        self.action_query = nn.Parameter(torch.zeros(1, chunk_size, token_dim))
        self.chunk_size = chunk_size

    def forward(self, batch):
        B, T, V, C, H, W = batch['rgb'].shape
        rgb = batch['rgb'].reshape(B*T*V, C, H, W)
        rgb_features = self.net._encode_image(rgb)
        B_T_V, N, D = rgb_features.shape
        rgb_features = rgb_features.reshape(B, T*V*N, D)

        text_embeds = self.prompt_embeds.repeat(B, 1, 1) # (b n d)
        inputs_embeds, attention_mask = self.net._merge_input_ids_with_image_features(rgb_features, text_embeds)

        low_dim = self.low_dim_encoder(batch['low_dim']) # (b, t, d)
        action_query = self.action_query.repeat(B, 1, 1) # (b, chunk size, d)

        decoder_inputs_embeds = torch.cat((low_dim, action_query), dim = 1)
        decoder_outputs_embeds = self.net.language_model(
            inputs_embeds = inputs_embeds,
            decoder_inputs_embeds = decoder_inputs_embeds,
            output_hidden_states = True,
        )['decoder_hidden_states'][-1]
        pred_actions = self.to_action(decoder_outputs_embeds[:, -self.chunk_size:])
        pred = {'action': pred_actions}
        return pred
