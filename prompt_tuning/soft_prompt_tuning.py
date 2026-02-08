import torch
import torch.nn as nn

class SoftPromptTuning:
    def __init__(self, model, num_prompts=20):
        self.model = model
        self.num_prompts = num_prompts

        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_prompts, model.config.hidden_size)
        )
    def forward(self, input_ids):
        text_embeds = self.model.get_input_embeddings()(input_ids)

        inputs = torch.cat([
            self.prompt_embeddings.unsqueeze(0).expand(input_ids.shape[0], -1, -1),
            text_embeds
        ], dim = 1)

        return self.model(inputs_embeds = inputs)

