import json

import pytest
import torch
from accelerate import Accelerator
from conftest import ToyOmniVoice
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.training import builder
from omnivoice.training.config import TrainingConfig
from omnivoice.training.lora import apply_lora, is_lora_model, save_lora_adapter


def _make_tokenizer(vocab_size):
    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        **{f"token-{index}": index for index in range(4, vocab_size)},
    }
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def _save_grown_vocab_adapter(tmp_path):
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        seed=1234,
    )
    base_model = ToyOmniVoice()
    base_state = {
        key: value.detach().clone() for key, value in base_model.state_dict().items()
    }
    tokenizer = _make_tokenizer(32)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<grown>"]})
    trained = builder._finalize_training_model(base_model, tokenizer, config)
    expected_embedding = (
        trained.get_base_model()
        .get_input_embeddings()
        .base_layer.weight.detach()
        .clone()
    )
    checkpoint = tmp_path / "checkpoint-7"
    save_lora_adapter(
        trained,
        checkpoint,
        config=config,
        step=7,
        accelerator=Accelerator(cpu=True),
    )
    tokenizer.save_pretrained(checkpoint)
    return checkpoint, base_state, expected_embedding


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


def test_from_lora_pretrained_can_load_an_exact_resolved_base_override(
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
    snapshot = tmp_path / "snapshots" / ("d" * 40)
    snapshot.mkdir(parents=True)
    loaded_bases = []
    monkeypatch.setattr(
        OmniVoice,
        "from_pretrained",
        classmethod(
            lambda cls, path, **kwargs: loaded_bases.append(path) or ToyOmniVoice()
        ),
    )

    model = OmniVoice.from_lora_pretrained(
        checkpoint,
        base_model_override=str(snapshot.resolve()),
        device_map="cpu",
    )

    assert loaded_bases == [str(snapshot.resolve())]
    assert is_lora_model(model)


@pytest.mark.parametrize("load_from_adapter_dir", [False, True])
def test_from_lora_pretrained_restores_grown_vocab_before_adapter_attachment(
    tmp_path, monkeypatch, load_from_adapter_dir
):
    checkpoint, base_state, expected_embedding = _save_grown_vocab_adapter(tmp_path)
    fresh_base = ToyOmniVoice()
    fresh_base.load_state_dict(base_state)
    fresh_base.text_tokenizer = _make_tokenizer(32)
    retained_audio_tokenizer = object()
    fresh_base.audio_tokenizer = retained_audio_tokenizer
    loaded_bases = []
    monkeypatch.setattr(
        OmniVoice,
        "from_pretrained",
        classmethod(
            lambda cls, path, **kwargs: loaded_bases.append(path) or fresh_base
        ),
    )

    load_path = checkpoint / "adapter" if load_from_adapter_dir else checkpoint
    model = OmniVoice.from_lora_pretrained(load_path, device_map="cpu")

    restored_base = model.get_base_model()
    restored_embedding = restored_base.get_input_embeddings().base_layer.weight
    assert loaded_bases == ["k2-fsa/OmniVoice"]
    assert len(restored_base.text_tokenizer) == 33
    assert restored_base.text_tokenizer.convert_tokens_to_ids("<grown>") == 32
    assert restored_base.config.vocab_size == 33
    assert restored_base.audio_tokenizer is retained_audio_tokenizer
    assert restored_embedding.shape == (33, 8)
    torch.testing.assert_close(restored_embedding, expected_embedding, rtol=0, atol=0)
    assert is_lora_model(model)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert not (checkpoint / "model.safetensors").exists()
    assert not (checkpoint / "pytorch_model.bin").exists()


def test_from_lora_pretrained_rejects_growth_without_resize_seed(tmp_path, monkeypatch):
    checkpoint, base_state, _ = _save_grown_vocab_adapter(tmp_path)
    metadata_path = checkpoint / "adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("embedding_resize_seed", None)
    metadata_path.write_text(json.dumps(metadata))
    fresh_base = ToyOmniVoice()
    fresh_base.load_state_dict(base_state)
    monkeypatch.setattr(
        OmniVoice,
        "from_pretrained",
        classmethod(lambda cls, path, **kwargs: fresh_base),
    )

    with pytest.raises(ValueError, match="embedding resize seed"):
        OmniVoice.from_lora_pretrained(checkpoint, device_map="cpu")
