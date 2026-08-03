"""LoRA configuration, checkpoint, and target discovery helpers."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from accelerate.utils import DistributedType, broadcast_object_list
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer

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

LORA_METADATA_FORMAT_VERSION = 1


def _run_main_process_io(accelerator, operation):
    """Run main-rank I/O and propagate failures before any rank raises."""
    error = [None]
    result = None
    caught_error = None
    if accelerator.is_main_process:
        try:
            result = operation()
        except BaseException as exc:  # noqa: BLE001
            caught_error = exc
            error[0] = f"{type(exc).__name__}: {exc}"
    broadcast_object_list(error)
    if error[0] is not None:
        if caught_error is not None:
            raise caught_error
        raise RuntimeError(f"Main process checkpoint I/O failed: {error[0]}")
    return result


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


def text_embedding_vocab_size(model):
    """Return the unwrapped text embedding vocabulary size."""
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    embeddings = model.get_input_embeddings()
    embeddings = getattr(embeddings, "base_layer", embeddings)
    return int(embeddings.weight.shape[0])


def resize_lora_token_embeddings(model, vocab_size, seed):
    """Resize text embeddings deterministically before attaching LoRA."""
    current_vocab_size = text_embedding_vocab_size(model)
    llm_config = getattr(model.config, "llm_config", model.config)
    if current_vocab_size == vocab_size:
        llm_config.vocab_size = vocab_size
        return

    resize_model = getattr(model, "llm", model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        resize_model.resize_token_embeddings(vocab_size)
    llm_config.vocab_size = vocab_size


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


def resolve_adapter_dir(checkpoint_path):
    """Return ``(checkpoint_root, adapter_dir)`` for a checkpoint or adapter path."""
    path = Path(checkpoint_path)
    nested_adapter = path / "adapter"
    if nested_adapter.is_dir():
        return path, nested_adapter
    if (path / "adapter_config.json").is_file():
        return path.parent, path
    raise FileNotFoundError(f"LoRA adapter not found at {path}")


def read_lora_metadata(checkpoint_path):
    """Read and validate schema-v1 metadata from a checkpoint root."""
    checkpoint_path = Path(checkpoint_path)
    if (checkpoint_path / "adapter_config.json").is_file():
        checkpoint_path = checkpoint_path.parent
    metadata_path = checkpoint_path / "adapter_metadata.json"
    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("format_version") != LORA_METADATA_FORMAT_VERSION:
        raise ValueError(
            "Unsupported LoRA metadata format_version: "
            f"{metadata.get('format_version')!r}"
        )
    return metadata


def _active_lora_config(model):
    active_adapter = model.active_adapter
    if not isinstance(active_adapter, str):
        raise TypeError("LoRA checkpoint saving requires exactly one active adapter")
    return model.peft_config[active_adapter]


def _lora_metadata(model, config, step):
    peft_config = _active_lora_config(model)
    live_values = {
        "lora_rank": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_dropout": peft_config.lora_dropout,
        "lora_bias": peft_config.bias,
    }
    differences = []
    for field, actual_value in live_values.items():
        expected_value = getattr(config, field)
        if actual_value != expected_value:
            differences.append(
                f"{field}: adapter={actual_value!r}, config={expected_value!r}"
            )
    live_targets = set(peft_config.target_modules)
    configured_targets = list(config.lora_target_modules)
    if live_targets != set(configured_targets):
        differences.append(
            "lora_target_modules: "
            f"adapter={sorted(live_targets)!r}, config={configured_targets!r}"
        )
    configured_base = _normalize_base_identifier(config.init_from_checkpoint)
    live_base = peft_config.base_model_name_or_path
    if live_base and _normalize_base_identifier(live_base) != configured_base:
        differences.append(
            "base_model_name_or_path: "
            f"adapter={live_base!r}, config={config.init_from_checkpoint!r}"
        )
    if differences:
        raise ValueError(
            "Live LoRA adapter differs from training config:\n- "
            + "\n- ".join(differences)
        )
    metadata = {
        "format_version": LORA_METADATA_FORMAT_VERSION,
        "base_model_name_or_path": configured_base,
        "step": step,
        "text_vocab_size": text_embedding_vocab_size(model),
        "embedding_resize_seed": config.seed,
        **live_values,
        "lora_target_modules": configured_targets,
    }
    if config.base_model_revision is not None:
        metadata["base_model_revision"] = config.base_model_revision
    return metadata


def _normalize_base_identifier(value):
    if value is None:
        return None
    raw_value = str(value).rstrip("/\\")
    path_parts = Path(raw_value).parts
    for index, part in enumerate(path_parts):
        if not part.startswith("models--"):
            continue
        if index + 2 >= len(path_parts) or path_parts[index + 1] != "snapshots":
            continue
        encoded_repo = part.removeprefix("models--")
        if "--" in encoded_repo:
            owner, repository = encoded_repo.split("--", 1)
            return f"{owner}/{repository}"
    return os.path.normpath(raw_value)


def validate_resume_metadata(config, metadata):
    """Reject every incompatible adapter setting in one diagnostic."""
    expected = {
        "base_model_name_or_path": config.init_from_checkpoint,
        "base_model_revision": config.base_model_revision,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_bias": config.lora_bias,
        "lora_target_modules": list(config.lora_target_modules),
    }
    differences = []
    for field, expected_value in expected.items():
        actual_value = metadata.get(field)
        if field == "base_model_name_or_path":
            values_match = _normalize_base_identifier(
                actual_value
            ) == _normalize_base_identifier(expected_value)
        else:
            values_match = actual_value == expected_value
        if not values_match:
            differences.append(
                f"{field}: checkpoint={actual_value!r}, config={expected_value!r}"
            )
    if differences:
        raise ValueError(
            "LoRA checkpoint metadata is incompatible:\n- "
            + "\n- ".join(differences)
        )


def _validate_adapter_config(adapter_dir, metadata):
    adapter_config_path = Path(adapter_dir) / "adapter_config.json"
    with adapter_config_path.open() as adapter_config_file:
        adapter_config = json.load(adapter_config_file)
    field_mapping = {
        "r": "lora_rank",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "bias": "lora_bias",
    }
    differences = []
    for adapter_field, metadata_field in field_mapping.items():
        actual_value = adapter_config.get(adapter_field)
        expected_value = metadata.get(metadata_field)
        if actual_value != expected_value:
            differences.append(
                f"adapter_config.{adapter_field}: "
                f"adapter={actual_value!r}, metadata={expected_value!r}"
            )
    adapter_targets = set(adapter_config.get("target_modules") or [])
    metadata_targets = metadata.get("lora_target_modules") or []
    if adapter_targets != set(metadata_targets):
        differences.append(
            "adapter_config.target_modules: "
            f"adapter={sorted(adapter_targets)!r}, metadata={metadata_targets!r}"
        )
    adapter_base = adapter_config.get("base_model_name_or_path")
    metadata_base = metadata.get("base_model_name_or_path")
    if adapter_base and _normalize_base_identifier(
        adapter_base
    ) != _normalize_base_identifier(metadata_base):
        differences.append(
            "adapter_config.base_model_name_or_path: "
            f"adapter={adapter_base!r}, metadata={metadata_base!r}"
        )
    if differences:
        raise ValueError(
            "PEFT adapter config differs from checkpoint metadata:\n- "
            + "\n- ".join(differences)
        )


def save_lora_adapter(model, checkpoint_path, config, step, accelerator):
    """Stage and atomically publish an adapter plus its checkpoint metadata."""
    checkpoint_root = Path(checkpoint_path)
    metadata = _lora_metadata(model, config, step)

    def save_files():
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=".adapter-", dir=str(checkpoint_root))
        )
        metadata_path = staging_dir / "adapter_metadata.json"
        try:
            _active_lora_config(model).base_model_name_or_path = metadata[
                "base_model_name_or_path"
            ]
            model.save_pretrained(
                staging_dir,
                is_main_process=True,
                save_function=accelerator.save,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            with metadata_path.open("w") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)
                metadata_file.write("\n")

            final_adapter_dir = checkpoint_root / "adapter"
            final_metadata_path = checkpoint_root / "adapter_metadata.json"
            os.replace(metadata_path, final_metadata_path)
            if final_adapter_dir.exists():
                shutil.rmtree(final_adapter_dir)
            os.replace(staging_dir, final_adapter_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    _run_main_process_io(accelerator, save_files)


def register_lora_state_hooks(accelerator):
    """Suppress full-model state while retaining other Accelerate state."""
    unsupported_backends = {
        DistributedType.DEEPSPEED,
        DistributedType.FSDP,
        DistributedType.MEGATRON_LM,
    }
    if accelerator.distributed_type in unsupported_backends:
        raise ValueError(
            "Adapter-only LoRA checkpoints do not support Accelerate backend "
            f"{accelerator.distributed_type.value}"
        )

    def save_hook(models, weights, output_dir):
        weights.clear()

    def load_hook(models, input_dir):
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def load_lora_adapter(model, checkpoint_path, config=None, is_trainable=False):
    """Load a saved LoRA adapter into a base model."""
    checkpoint_root, adapter_dir = resolve_adapter_dir(checkpoint_path)
    is_checkpoint_adapter = adapter_dir == checkpoint_root / "adapter"
    metadata_path = checkpoint_root / "adapter_metadata.json"
    if is_checkpoint_adapter or metadata_path.is_file() or config is not None:
        metadata = read_lora_metadata(checkpoint_root)
        _validate_adapter_config(adapter_dir, metadata)
    if config is not None:
        validate_resume_metadata(config, metadata)
    return PeftModel.from_pretrained(
        model, adapter_dir, is_trainable=is_trainable
    )


def _load_checkpoint_text_tokenizer(checkpoint_root):
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
        "vocab.txt",
    )
    if not any((checkpoint_root / name).is_file() for name in tokenizer_files):
        return None
    try:
        return AutoTokenizer.from_pretrained(checkpoint_root, local_files_only=True)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load the LoRA checkpoint text tokenizer from {checkpoint_root}"
        ) from exc


def _legacy_resize_seed(checkpoint_root):
    train_config_path = checkpoint_root / "train_config.json"
    if not train_config_path.is_file():
        return None
    try:
        with train_config_path.open() as config_file:
            train_config = json.load(config_file)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read legacy LoRA resize metadata from {train_config_path}"
        ) from exc
    return train_config.get("seed")


def _prepare_lora_base_for_inference(model, checkpoint_path):
    checkpoint_root, _ = resolve_adapter_dir(checkpoint_path)
    metadata = read_lora_metadata(checkpoint_root)
    checkpoint_tokenizer = _load_checkpoint_text_tokenizer(checkpoint_root)
    metadata_vocab_size = metadata.get("text_vocab_size")
    if metadata_vocab_size is not None and (
        isinstance(metadata_vocab_size, bool)
        or not isinstance(metadata_vocab_size, int)
        or metadata_vocab_size <= 0
    ):
        raise ValueError(
            "LoRA checkpoint text_vocab_size must be a positive integer"
        )

    if checkpoint_tokenizer is not None:
        tokenizer_vocab_size = len(checkpoint_tokenizer)
        if (
            metadata_vocab_size is not None
            and tokenizer_vocab_size != metadata_vocab_size
        ):
            raise ValueError(
                "LoRA checkpoint tokenizer vocabulary differs from metadata: "
                f"tokenizer={tokenizer_vocab_size}, "
                f"text_vocab_size={metadata_vocab_size}"
            )
        target_vocab_size = tokenizer_vocab_size
    else:
        target_vocab_size = metadata_vocab_size

    current_vocab_size = text_embedding_vocab_size(model)
    if target_vocab_size is not None and target_vocab_size != current_vocab_size:
        if checkpoint_tokenizer is None:
            raise ValueError(
                "LoRA checkpoint text vocabulary differs from the base model, but "
                "the checkpoint tokenizer is missing; load the complete checkpoint "
                "root or re-save the adapter with tokenizer artifacts"
            )
        resize_seed = metadata.get("embedding_resize_seed")
        if resize_seed is None:
            resize_seed = _legacy_resize_seed(checkpoint_root)
        if isinstance(resize_seed, bool) or not isinstance(resize_seed, int):
            raise ValueError(
                "LoRA checkpoint requires an embedding resize seed; expected "
                "adapter_metadata.json field 'embedding_resize_seed' or a legacy "
                "train_config.json integer 'seed'"
            )
        resize_lora_token_embeddings(model, target_vocab_size, resize_seed)

    if checkpoint_tokenizer is not None:
        model.text_tokenizer = checkpoint_tokenizer
        model.config.pad_token_id = checkpoint_tokenizer.pad_token_id
        model.config.bos_token_id = checkpoint_tokenizer.bos_token_id
        model.config.eos_token_id = checkpoint_tokenizer.eos_token_id
    return model


def load_lora_for_inference(model, checkpoint_path):
    """Load a checkpoint adapter as a frozen model ready for inference."""
    model = _prepare_lora_base_for_inference(model, checkpoint_path)
    try:
        model = load_lora_adapter(model, checkpoint_path, is_trainable=False)
    except RuntimeError as exc:
        message = str(exc)
        if "lora_embedding_A" in message and "size mismatch" in message:
            raise ValueError(
                "LoRA adapter embedding vocabulary differs from the base model and "
                "this legacy checkpoint lacks enough tokenizer/resize metadata to "
                "reconstruct it; load the complete checkpoint root or re-save the "
                "adapter"
            ) from exc
        raise
    model.requires_grad_(False)
    model.eval()
    return model


def is_lora_model(model):
    """Return whether a model is wrapped in PEFT's LoRA model type."""
    return isinstance(model, PeftModel)
