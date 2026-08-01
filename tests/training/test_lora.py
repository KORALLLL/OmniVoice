import json
import re

import pytest

from omnivoice.training.config import TrainingConfig


def test_training_config_loads_broad_lora_defaults(tmp_path):
    from omnivoice.training.lora import DEFAULT_LORA_TARGET_MODULES

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "init_from_checkpoint": "k2-fsa/OmniVoice",
                "lora_enabled": True,
            }
        )
    )
    config = TrainingConfig.from_json(path)
    assert config.lora_enabled is True
    assert config.lora_rank == 32
    assert config.lora_alpha == 64
    assert config.lora_dropout == 0.05
    assert config.lora_bias == "none"
    assert config.lora_target_modules == list(DEFAULT_LORA_TARGET_MODULES)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lora_rank", 0, "lora_rank must be positive"),
        ("lora_alpha", 0, "lora_alpha must be positive"),
        ("lora_dropout", 1.0, "lora_dropout must be in [0, 1)"),
        ("lora_bias", "all", "lora_bias must be 'none' for adapter-only training"),
    ],
)
def test_validate_lora_config_rejects_invalid_values(field, value, message):
    from omnivoice.training.lora import validate_lora_config

    config = TrainingConfig(lora_enabled=True, init_from_checkpoint="base")
    setattr(config, field, value)
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_lora_config(config)


def test_apply_lora_wraps_every_broad_target_and_freezes_base(toy_omnivoice):
    from omnivoice.training.lora import DEFAULT_LORA_TARGET_MODULES, apply_lora

    config = TrainingConfig(
        init_from_checkpoint="base",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
    )
    model, matches = apply_lora(toy_omnivoice, config)

    assert set(matches) == set(DEFAULT_LORA_TARGET_MODULES)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all("lora_" in name for name in trainable)
    assert any("audio_embeddings" in name for name in trainable)
    assert any("audio_heads" in name for name in trainable)


def test_apply_lora_rejects_missing_requested_target(toy_omnivoice):
    from omnivoice.training.lora import apply_lora

    config = TrainingConfig(
        init_from_checkpoint="base",
        lora_enabled=True,
        lora_target_modules=["q_proj", "does_not_exist"],
    )
    with pytest.raises(ValueError, match="does_not_exist"):
        apply_lora(toy_omnivoice, config)


def test_find_lora_target_modules_returns_full_module_names(toy_omnivoice):
    from omnivoice.training.lora import find_lora_target_modules

    matches = find_lora_target_modules(toy_omnivoice, ["q_proj", "audio_heads"])

    assert matches == {"q_proj": ["q_proj"], "audio_heads": ["audio_heads"]}


def test_trainable_parameter_counts_reports_trainable_and_total(toy_omnivoice):
    from omnivoice.training.lora import trainable_parameter_counts

    trainable, total = trainable_parameter_counts(toy_omnivoice)

    assert trainable == total
    assert total == sum(parameter.numel() for parameter in toy_omnivoice.parameters())


def test_load_lora_adapter_restores_non_trainable_adapter(tmp_path, toy_omnivoice):
    from omnivoice.training.lora import (
        apply_lora,
        is_lora_model,
        load_lora_adapter,
    )

    config = TrainingConfig(init_from_checkpoint="base", lora_enabled=True)
    adapter, _ = apply_lora(toy_omnivoice, config)
    adapter.save_pretrained(tmp_path)

    restored = load_lora_adapter(type(toy_omnivoice)(), tmp_path, is_trainable=False)

    assert is_lora_model(adapter)
    assert is_lora_model(restored)
    assert not any(parameter.requires_grad for parameter in restored.parameters())
