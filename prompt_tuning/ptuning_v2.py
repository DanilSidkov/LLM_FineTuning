import torch
import torch.nn as nn
from typing import Optional, Tuple

class PTuningV2(nn.Module):
    def __init__(self, model, num_prompts=20, prefix_len=10):
        """
        P-Tuning v2: Deep Prompt Tuning с адаптерами для каждого слоя
        
        Args:
            model: базовая модель (замороженная)
            num_prompts: количество базовых промпт-токенов
            prefix_len: длина префикса для каждого слоя
        """
        super().__init__()
        self.model = model
        self.num_prompts = num_prompts
        self.prefix_len = prefix_len
        self.hidden_size = model.config.hidden_size
        
        self.prompt_tokens = nn.Parameter(
            torch.randn(num_prompts, self.hidden_size)
        )
        
        self.prefix_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size * 2, 2 * self.hidden_size * self.prefix_len)
            )
            for _ in range(model.config.num_hidden_layers)
        ])
        
        self.hooks = []
        self.register_hooks()
        
        self.prefix_cache = {}
    
    def register_hooks(self):
        """Регистрируем forward hooks на attention слои"""
        
        def make_hook(layer_idx):
            def attention_hook(module, input_args, output):
                hidden_states = input_args[0]
                batch_size = hidden_states.shape[0]
                
                if layer_idx not in self.prefix_cache:
                    prefix_params = self.prefix_encoders[layer_idx](self.prompt_tokens.mean(dim=0, keepdim=True))
                    prefix_params = prefix_params.view(-1, self.prefix_len, 2 * self.hidden_size)
                    
                    prefix_key = prefix_params[..., :self.hidden_size]
                    prefix_value = prefix_params[..., self.hidden_size:]
                    
                    self.prefix_cache[layer_idx] = (prefix_key, prefix_value)
                
                prefix_key, prefix_value = self.prefix_cache[layer_idx]
                
                prefix_key = prefix_key.expand(batch_size, -1, -1)
                prefix_value = prefix_value.expand(batch_size, -1, -1)
                
                return self.modified_attention_forward(module, hidden_states, prefix_key, prefix_value)
            
            return attention_hook
        
        for i, layer in enumerate(self.model.transformer.h):
            hook = layer.attn.register_forward_hook(make_hook(i))
            self.hooks.append(hook)
    
    def modified_attention_forward(self, attention_module, hidden_states, prefix_key, prefix_value):
        """
        Модифицированный forward pass для attention с префиксами
        (упрощенная версия, на практике нужно адаптировать под конкретную архитектуру)
        """
        query = attention_module.q_proj(hidden_states)
        key = attention_module.k_proj(hidden_states)
        value = attention_module.v_proj(hidden_states)
        
        key_with_prefix = torch.cat([prefix_key, key], dim=1)
        value_with_prefix = torch.cat([prefix_value, value], dim=1)
        
        attention_scores = torch.matmul(query, key_with_prefix.transpose(-2, -1))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value_with_prefix)
        
        output = attention_module.out_proj(attention_output)
        
        return output, attention_probs
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass с P-Tuning v2
        """
        text_embeds = self.model.transformer.wte(input_ids)
        batch_size = input_ids.shape[0]
        prompt_embeds = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        inputs_embeds = torch.cat([prompt_embeds, text_embeds], dim=1)
        
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_prompts, 
                                    device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        return outputs
    
    def remove_hooks(self):
        """Удаляем все хуки (важно для освобождения памяти)"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def train(self, mode=True):
        """Переключаем режим train/eval"""
        super().train(mode)
        self.model.train(mode)
    
    def eval(self):
        """Переключаем в режим eval"""
        super().eval()
        self.model.eval()