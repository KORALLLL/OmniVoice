"""LoRA configuration validation and target discovery helpers."""

from peft import LoraConfig, PeftModel, get_peft_model


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "audio_embeddings",
    "audio_heads",
)


def find_lora_target_modules(model, suffixes):
    """Return matching module names for each requested target suffix."""
    matches = {suffix: [] for suffix in suffixes}
    for name, module in model.named_modules():
        for suffix in suffixes:
            if name == suffix or name.endswith(f".{suffix}"):
                matches[suffix].append(name)
    missing = [suffix for suffix, names in matches.items() if not names]
    if missing:
        raise ValueError(f"LoRA target modules not found: {', '.join(missing)}")
    return matches


def validate_lora_config(config):
    """Validate adapter-only LoRA settings before mutating a model."""
    if not config.lora_enabled:
        return
    if not config.init_from_checkpoint:
        raise ValueError("LoRA requires init_from_checkpoint")
    if config.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if config.lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")
    if not 0.0 <= config.lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    if config.lora_bias != "none":
        raise ValueError("lora_bias must be 'none' for adapter-only training")
    if not config.lora_target_modules:
        raise ValueError("lora_target_modules must not be empty")


def trainable_parameter_counts(model):
    """Return the number of trainable and total model parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def apply_lora(model, config):
    """Insert LoRA adapters after validating every requested target module."""
    validate_lora_config(config)
    matches = find_lora_target_modules(model, config.lora_target_modules)
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=list(config.lora_target_modules),
        task_type=None,
    )
    model = get_peft_model(model, peft_config)
    trainable, _ = trainable_parameter_counts(model)
    if trainable == 0:
        raise RuntimeError("LoRA produced no trainable parameters")
    return model, matches


def load_lora_adapter(model, checkpoint_path, is_trainable):
    """Load a saved LoRA adapter into a base model."""
    return PeftModel.from_pretrained(
        model, checkpoint_path, is_trainable=is_trainable
    )


def is_lora_model(model):
    """Return whether a model is wrapped in PEFT's LoRA model type."""
    return isinstance(model, PeftModel)
