from accelerate import Accelerator
from conftest import ToyOmniVoice

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.training.config import TrainingConfig
from omnivoice.training.lora import apply_lora, is_lora_model, save_lora_adapter


def test_from_lora_pretrained_uses_recorded_base_and_loads_adapter(
    tmp_path, monkeypatch, toy_omnivoice
):
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
    )
    trained, _ = apply_lora(toy_omnivoice, config)
    checkpoint = tmp_path / "checkpoint-7"
    save_lora_adapter(
        trained,
        checkpoint,
        config=config,
        step=7,
        accelerator=Accelerator(cpu=True),
    )
    loaded_bases = []
    monkeypatch.setattr(
        OmniVoice,
        "from_pretrained",
        classmethod(
            lambda cls, path, **kwargs: loaded_bases.append(path) or ToyOmniVoice()
        ),
    )

    model = OmniVoice.from_lora_pretrained(checkpoint, device_map="cpu")

    assert loaded_bases == ["k2-fsa/OmniVoice"]
    assert is_lora_model(model)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
