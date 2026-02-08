import torch
import torch.nn as nn

class PrefixTuning(nn.Module):
    def __init__(self, model, prefix_length = 10):
        super().__init__()
        self.model = model

        self.prefix_keys = nn.ParameterList([
            nn.Parameter(torch.randn(prefix_length, model.config.hidden_size))
            for _ in range(model.config.num_hidden_layers)
        ])
        self.prefix_values = nn.ParameterList([
            nn.Parameter(torch.randn(prefix_length, model.config.hidden_size))
            for _ in range(model.config.num_hidden_layers)
        ])

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]

        def custom_attention(layer_idx, query, key, value):
            batch_prefix_k = self.prefix_keys[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
            batch_prefix_v = self.prefix_values[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)

            key_with_prefix = torch.cat([batch_prefix_k, key], dim = 1)
            value_with_prefix = torch.cat([batch_prefix_v, value], dim = 1)

            attn_weights = torch.matmul(query, key_with_prefix.transpose(-2, -1))
            attn_weights = torch.softmax(attn_weights, dim= -1)
            output = torch.matmul(attn_weights, value_with_prefix)

            return output