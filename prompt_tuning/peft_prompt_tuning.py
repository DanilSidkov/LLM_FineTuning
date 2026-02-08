from peft import get_peft_model, PromptTuningConfig, TaskType, PrefixTuningConfig, PromptEncoderConfig


peft_config = PromptTuningConfig( #soft tuning
    task_type = TaskType.CAUSAL_LM,
    prompt_tuning_init = "RANDOM"
    num_virtual_tokens = 20,
    tokenizer_name_or_path = "gpt2",
)

peft_config = PrefixTuningConfig(
    task_type = TaskType.CAUSAL_LM,
    num_virtual_tokens = 30,
    prefix_projection = True,
)

peft_config = PromptEncoderConfig(
    task_type = TaskType.CAUSAL_LM,
    num_virtual_tokens = 20,
    encoder_hidden_size = 128,
    encoder_num_layers = 2,
    encoder_dropout = 0.1,
)

model = "какая-то модель"
model = get_peft_model(model, peft_config)
