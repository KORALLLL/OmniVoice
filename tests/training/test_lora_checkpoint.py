import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from conftest import DummyTokenizer, ToyOmniVoice
from peft import get_peft_model_state_dict
from safetensors import safe_open
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from omnivoice.training import builder
from omnivoice.training import checkpoint as checkpoint_module
from omnivoice.training import lora as lora_module
from omnivoice.training.checkpoint import load_checkpoint, save_checkpoint
from omnivoice.training.config import TrainingConfig
from omnivoice.training.lora import (
    apply_lora,
    load_lora_adapter,
    read_lora_metadata,
    register_lora_state_hooks,
    resolve_adapter_dir,
)
from omnivoice.training.trainer import OmniTrainer


def test_lora_checkpoint_omits_frozen_model_and_contains_adapter(
    tmp_path, toy_omnivoice
):
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        output_dir=str(tmp_path),
        lora_enabled=True,
        steps=2,
    )
    model, _ = apply_lora(toy_omnivoice, config)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    register_lora_state_hooks(accelerator)

    save_checkpoint(
        accelerator,
        model,
        DummyTokenizer(),
        config,
        str(tmp_path),
        step=7,
        keep_last_n=-1,
    )

    checkpoint = tmp_path / "checkpoint-7"
    assert (checkpoint / "adapter" / "adapter_config.json").is_file()
    assert (checkpoint / "adapter" / "adapter_model.safetensors").is_file()
    assert (checkpoint / "adapter_metadata.json").is_file()
    assert not (checkpoint / "model.safetensors").exists()
    assert not (checkpoint / "pytorch_model.bin").exists()
    assert any(checkpoint.glob("optimizer*"))
    assert any(checkpoint.glob("scheduler*"))
    with safe_open(
        checkpoint / "adapter" / "adapter_model.safetensors", framework="pt"
    ) as adapter_file:
        assert all("base_layer" not in key for key in adapter_file.keys())  # noqa: SIM118

    assert read_lora_metadata(checkpoint) == {
        "format_version": 1,
        "base_model_name_or_path": "k2-fsa/OmniVoice",
        "step": 7,
        "text_vocab_size": 32,
        "embedding_resize_seed": 42,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "lora_target_modules": [
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
        ],
    }
    assert resolve_adapter_dir(checkpoint) == (checkpoint, checkpoint / "adapter")
    assert resolve_adapter_dir(checkpoint / "adapter") == (
        checkpoint,
        checkpoint / "adapter",
    )


def save_checkpoint_after_one_step(tmp_path, base_model):
    base_state = copy.deepcopy(base_model.state_dict())
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        output_dir=str(tmp_path),
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        steps=2,
    )
    model, _ = apply_lora(base_model, config)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    register_lora_state_hooks(accelerator)
    outputs = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        audio_ids=torch.tensor([[3, 2, 1]]),
        labels=torch.tensor([[1, 2, 3]]),
    )
    accelerator.backward(outputs.loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    expected_adapter = {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(
            accelerator.unwrap_model(model)
        ).items()
    }
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    save_checkpoint(
        accelerator,
        model,
        DummyTokenizer(),
        config,
        str(tmp_path),
        step=7,
        keep_last_n=-1,
    )
    expected_next_random = torch.rand(4)
    return (
        tmp_path / "checkpoint-7",
        expected_adapter,
        expected_optimizer,
        expected_scheduler,
        expected_next_random,
        base_state,
    )


def save_checkpoint_with_resolved_hub_base(tmp_path, base_model):
    snapshot_path = (
        tmp_path
        / "hub"
        / "models--k2-fsa--OmniVoice"
        / "snapshots"
        / "0123456789abcdef"
    )
    base_model.config._name_or_path = str(snapshot_path)
    base_model.__dict__["name_or_path"] = str(snapshot_path)
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        output_dir=str(tmp_path),
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        steps=2,
    )
    model, _ = apply_lora(base_model, config)
    assert model.peft_config[model.active_adapter].base_model_name_or_path == str(
        snapshot_path
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    register_lora_state_hooks(accelerator)
    save_checkpoint(
        accelerator,
        model,
        DummyTokenizer(),
        config,
        str(tmp_path),
        step=7,
        keep_last_n=-1,
    )
    return tmp_path / "checkpoint-7"


def assert_optimizer_state_equal(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id in actual["state"]:
        assert (
            actual["state"][parameter_id].keys()
            == expected["state"][parameter_id].keys()
        )
        for key, actual_value in actual["state"][parameter_id].items():
            expected_value = expected["state"][parameter_id][key]
            if isinstance(actual_value, torch.Tensor):
                torch.testing.assert_close(actual_value.cpu(), expected_value.cpu())
            else:
                assert actual_value == expected_value


def test_lora_checkpoint_resume_restores_adapter_and_optimizer(
    tmp_path, toy_omnivoice
):
    (
        checkpoint,
        expected_adapter,
        expected_optimizer,
        expected_scheduler,
        expected_next_random,
        base_state,
    ) = save_checkpoint_after_one_step(tmp_path, toy_omnivoice)
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        resume_from_checkpoint=str(checkpoint),
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        steps=2,
    )
    resumed_base = ToyOmniVoice()
    resumed_base.load_state_dict(base_state)
    model = load_lora_adapter(
        resumed_base, checkpoint, config=config, is_trainable=True
    )
    actual_adapter = {
        key: value.detach().cpu()
        for key, value in get_peft_model_state_dict(model).items()
    }
    assert actual_adapter.keys() == expected_adapter.keys()
    assert all(
        torch.equal(actual_adapter[key], expected_adapter[key])
        for key in actual_adapter
    )

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    register_lora_state_hooks(accelerator)

    torch.manual_seed(123456)
    assert load_checkpoint(accelerator, str(checkpoint)) == 7
    assert_optimizer_state_equal(optimizer.state_dict(), expected_optimizer)
    assert scheduler.state_dict() == expected_scheduler
    torch.testing.assert_close(torch.rand(4), expected_next_random)


def test_hub_base_accepts_its_resolved_snapshot_identity(tmp_path, toy_omnivoice):
    checkpoint = save_checkpoint_with_resolved_hub_base(tmp_path, toy_omnivoice)
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        steps=2,
    )

    restored = load_lora_adapter(
        ToyOmniVoice(), checkpoint, config=config, is_trainable=True
    )

    assert restored.peft_config[restored.active_adapter].r == 4
    assert read_lora_metadata(checkpoint)["base_model_name_or_path"] == (
        "k2-fsa/OmniVoice"
    )
    adapter_config = json.loads(
        (checkpoint / "adapter" / "adapter_config.json").read_text()
    )
    assert adapter_config["base_model_name_or_path"] == "k2-fsa/OmniVoice"


def test_hub_snapshot_identity_still_rejects_a_different_base(
    tmp_path, toy_omnivoice
):
    checkpoint = save_checkpoint_with_resolved_hub_base(tmp_path, toy_omnivoice)
    config = TrainingConfig(
        init_from_checkpoint="other/OmniVoice",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        steps=2,
    )

    with pytest.raises(ValueError, match="base_model_name_or_path"):
        load_lora_adapter(
            ToyOmniVoice(), checkpoint, config=config, is_trainable=True
        )


def test_full_checkpoint_still_writes_model_weights(tmp_path, toy_omnivoice):
    config = TrainingConfig(
        output_dir=str(tmp_path),
        lora_enabled=False,
        steps=2,
    )
    optimizer = AdamW(toy_omnivoice.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(
        toy_omnivoice, optimizer, scheduler
    )

    save_checkpoint(
        accelerator,
        model,
        DummyTokenizer(),
        config,
        str(tmp_path),
        step=7,
        keep_last_n=-1,
    )

    checkpoint = tmp_path / "checkpoint-7"
    assert (checkpoint / "model.safetensors").is_file()
    assert not (checkpoint / "adapter").exists()


def test_lora_checkpoint_can_replace_the_same_step(tmp_path, toy_omnivoice):
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        output_dir=str(tmp_path),
        lora_enabled=True,
        steps=2,
    )
    model, _ = apply_lora(toy_omnivoice, config)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    register_lora_state_hooks(accelerator)

    for _ in range(2):
        save_checkpoint(
            accelerator,
            model,
            DummyTokenizer(),
            config,
            str(tmp_path),
            step=7,
            keep_last_n=-1,
        )

    checkpoint = tmp_path / "checkpoint-7"
    assert (checkpoint / "adapter" / "adapter_model.safetensors").is_file()
    assert not list(checkpoint.glob(".adapter-*"))


def test_lora_checkpoint_publish_failure_restores_previous_checkpoint(
    tmp_path, toy_omnivoice, monkeypatch
):
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        output_dir=str(tmp_path),
        lora_enabled=True,
        steps=2,
    )
    model, _ = apply_lora(toy_omnivoice, config)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    register_lora_state_hooks(accelerator)
    save_checkpoint(
        accelerator,
        model,
        DummyTokenizer(),
        config,
        str(tmp_path),
        step=7,
        keep_last_n=-1,
    )
    checkpoint = tmp_path / "checkpoint-7"
    adapter_path = checkpoint / "adapter" / "adapter_model.safetensors"
    original_adapter = adapter_path.read_bytes()

    with torch.no_grad():
        next(
            parameter for parameter in model.parameters() if parameter.requires_grad
        ).add_(1)
    real_replace = checkpoint_module.os.replace

    def fail_new_checkpoint_publish(source, destination):
        if Path(source).name == ".checkpoint-7.tmp" and Path(
            destination
        ) == checkpoint:
            raise OSError("injected checkpoint publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_new_checkpoint_publish)

    with pytest.raises(OSError, match="injected checkpoint publish failure"):
        save_checkpoint(
            accelerator,
            model,
            DummyTokenizer(),
            config,
            str(tmp_path),
            step=7,
            keep_last_n=-1,
        )

    assert adapter_path.read_bytes() == original_adapter
    assert not (tmp_path / ".checkpoint-7.tmp").exists()
    assert not (tmp_path / ".checkpoint-7.old").exists()


def test_lora_trainer_registers_compact_state_hooks(
    tmp_path, toy_omnivoice, monkeypatch
):
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        output_dir=str(tmp_path),
        lora_enabled=True,
        steps=2,
    )
    model, _ = apply_lora(toy_omnivoice, config)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    monkeypatch.setattr(
        OmniTrainer, "_init_accelerator", lambda trainer: accelerator
    )
    trainer = OmniTrainer(
        model,
        config,
        train_dataloader=[],
        tokenizer=DummyTokenizer(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    trainer.save_checkpoint(7)

    checkpoint = tmp_path / "checkpoint-7"
    assert (checkpoint / "adapter" / "adapter_model.safetensors").is_file()
    assert not (checkpoint / "model.safetensors").exists()


def test_builder_resume_metadata_mismatch_rejected_before_trainer(
    tmp_path, toy_omnivoice
):
    checkpoint, *_ = save_checkpoint_after_one_step(tmp_path, toy_omnivoice)
    config = TrainingConfig(
        init_from_checkpoint="other/base",
        resume_from_checkpoint=str(checkpoint),
        lora_enabled=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.25,
        lora_bias="all",
        lora_target_modules=["q_proj"],
        steps=2,
    )

    with pytest.raises(ValueError) as exc_info:
        builder._finalize_training_model(ToyOmniVoice(), DummyTokenizer(), config)

    message = str(exc_info.value)
    for field in (
        "base_model_name_or_path",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "lora_target_modules",
    ):
        assert field in message


def test_read_lora_metadata_rejects_unknown_format_version(tmp_path):
    (tmp_path / "adapter_metadata.json").write_text(
        json.dumps({"format_version": 2})
    )

    with pytest.raises(ValueError, match="format_version"):
        read_lora_metadata(tmp_path)


def test_save_rejects_training_config_that_differs_from_live_adapter(
    tmp_path, toy_omnivoice
):
    model_config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        steps=2,
    )
    save_config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        lora_enabled=True,
        lora_rank=16,
        lora_alpha=32,
        steps=2,
    )
    model, _ = apply_lora(toy_omnivoice, model_config)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    register_lora_state_hooks(accelerator)

    with pytest.raises(ValueError) as exc_info:
        save_checkpoint(
            accelerator,
            model,
            DummyTokenizer(),
            save_config,
            str(tmp_path),
            step=7,
            keep_last_n=-1,
        )

    assert "lora_rank" in str(exc_info.value)
    assert "lora_alpha" in str(exc_info.value)


def test_load_rejects_adapter_config_that_differs_from_metadata(
    tmp_path, toy_omnivoice
):
    checkpoint, *_ = save_checkpoint_after_one_step(tmp_path, toy_omnivoice)
    adapter_config_path = checkpoint / "adapter" / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text())
    adapter_config["r"] = 16
    adapter_config["lora_alpha"] = 32
    adapter_config_path.write_text(json.dumps(adapter_config))
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        steps=2,
    )

    with pytest.raises(ValueError) as exc_info:
        load_lora_adapter(ToyOmniVoice(), checkpoint, config=config, is_trainable=True)

    assert "adapter_config.r" in str(exc_info.value)
    assert "adapter_config.lora_alpha" in str(exc_info.value)


@pytest.mark.parametrize(
    "distributed_type",
    [
        DistributedType.DEEPSPEED,
        DistributedType.FSDP,
        DistributedType.MEGATRON_LM,
    ],
)
def test_lora_state_hooks_reject_backends_that_save_before_hooks(distributed_type):
    accelerator = SimpleNamespace(
        distributed_type=distributed_type,
        register_save_state_pre_hook=lambda hook: None,
        register_load_state_pre_hook=lambda hook: None,
    )

    with pytest.raises(ValueError, match=distributed_type.value):
        register_lora_state_hooks(accelerator)


def test_lora_resize_initializes_frozen_rows_deterministically():
    class LargerTokenizer(DummyTokenizer):
        def __len__(self):
            return 33

    original = ToyOmniVoice()
    original_state = copy.deepcopy(original.state_dict())
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        seed=42,
        steps=2,
    )

    torch.manual_seed(100)
    first_base = ToyOmniVoice()
    first_base.load_state_dict(original_state)
    first = builder._finalize_training_model(first_base, LargerTokenizer(), config)
    torch.manual_seed(200)
    second_base = ToyOmniVoice()
    second_base.load_state_dict(original_state)
    second = builder._finalize_training_model(second_base, LargerTokenizer(), config)

    first_embedding = first.base_model.model.embed_tokens.base_layer.weight
    second_embedding = second.base_model.model.embed_tokens.base_layer.weight
    assert torch.equal(first_embedding, second_embedding)


def test_main_process_io_broadcasts_error_before_raising(monkeypatch):
    broadcasts = []
    monkeypatch.setattr(
        lora_module,
        "broadcast_object_list",
        lambda payload: broadcasts.append(list(payload)) or payload,
    )
    accelerator = SimpleNamespace(is_main_process=True)

    with pytest.raises(OSError, match="write failed"):
        lora_module._run_main_process_io(
            accelerator, lambda: (_ for _ in ()).throw(OSError("write failed"))
        )

    assert broadcasts == [["OSError: write failed"]]


def test_non_main_process_receives_main_process_io_error(monkeypatch):
    def receive_error(payload):
        payload[0] = "OSError: write failed"
        return payload

    monkeypatch.setattr(lora_module, "broadcast_object_list", receive_error)
    accelerator = SimpleNamespace(is_main_process=False)

    with pytest.raises(RuntimeError, match="OSError: write failed"):
        lora_module._run_main_process_io(
            accelerator,
            lambda: pytest.fail("non-main process must not perform checkpoint I/O"),
        )


def test_save_state_rank_zero_failure_is_gathered_before_raising(
    tmp_path, monkeypatch
):
    gathered_errors = []

    def gather_errors(payload):
        gathered_errors.append(list(payload))
        return payload

    monkeypatch.setattr(
        checkpoint_module, "gather_object", gather_errors, raising=False
    )
    accelerator = SimpleNamespace(
        is_main_process=True,
        process_index=0,
        save_state=lambda path: (_ for _ in ()).throw(
            OSError("rank zero state write failed")
        ),
    )

    with pytest.raises(RuntimeError, match="rank zero state write failed"):
        save_checkpoint(
            accelerator,
            model=None,
            tokenizer=None,
            config=TrainingConfig(lora_enabled=False),
            output_dir=str(tmp_path),
            step=7,
            keep_last_n=-1,
        )

    assert gathered_errors == [["process 0 OSError: rank zero state write failed"]]


def test_save_state_peer_receives_rank_zero_failure(tmp_path, monkeypatch):
    def gather_rank_zero_error(payload):
        return ["process 0 OSError: rank zero state write failed", *payload]

    monkeypatch.setattr(
        checkpoint_module, "gather_object", gather_rank_zero_error, raising=False
    )
    accelerator = SimpleNamespace(
        is_main_process=False,
        process_index=1,
        save_state=lambda path: None,
    )

    with pytest.raises(RuntimeError, match="rank zero state write failed"):
        save_checkpoint(
            accelerator,
            model=None,
            tokenizer=None,
            config=TrainingConfig(lora_enabled=False),
            output_dir=str(tmp_path),
            step=7,
            keep_last_n=-1,
        )


def test_xla_full_checkpoint_does_not_use_unsupported_object_gather(
    tmp_path, monkeypatch
):
    class XlaFullModel:
        def save_pretrained(self, path, **kwargs):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.safetensors").write_bytes(b"xla model")

    def unsupported_gather(payload):
        raise NotImplementedError("gather objects in TPU is not supported")

    monkeypatch.setattr(checkpoint_module, "gather_object", unsupported_gather)
    model = XlaFullModel()
    accelerator = SimpleNamespace(
        is_main_process=True,
        process_index=0,
        distributed_type=DistributedType.XLA,
        save_state=lambda path: Path(path).mkdir(parents=True, exist_ok=True),
        unwrap_model=lambda wrapped_model: wrapped_model,
        save=torch.save,
    )

    save_checkpoint(
        accelerator,
        model,
        DummyTokenizer(),
        TrainingConfig(lora_enabled=False),
        str(tmp_path),
        step=7,
        keep_last_n=-1,
    )

    checkpoint = tmp_path / "checkpoint-7"
    assert (checkpoint / "model.safetensors").read_bytes() == b"xla model"
    assert (checkpoint / "tokenizer_config.json").is_file()
