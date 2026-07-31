# OmniVoice Broad LoRA Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add adapter-only broad LoRA fine-tuning, resumable compact checkpoints, inference loading, an eight-GPU example, and a deterministic four-utterance memorization check to OmniVoice.

**Architecture:** PEFT wraps the complete `OmniVoice` model after tokenizer resizing so one adapter can target Qwen projections, text embeddings, audio embeddings, and the audio output head. Accelerate save/load hooks suppress frozen base-model serialization while the existing trainer continues to own optimizer, scheduler, RNG, DDP, evaluation, and checkpoint rotation. A small integration CLI evaluates the native masked-token cross-entropy of a reloaded adapter.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, PEFT, Accelerate, WebDataset, pytest, Bash.

## Global Constraints

- Existing non-LoRA training configurations and checkpoints must retain their current behavior.
- The default base model is exactly `k2-fsa/OmniVoice`.
- The default adapter is rank 32, alpha 64, dropout 0.05, and bias `none`.
- Default targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `embed_tokens`, `audio_embeddings`, and `audio_heads`.
- All original model weights remain frozen; only LoRA parameters are optimized and serialized.
- LoRA checkpoints must not contain a frozen full-model weight file.
- The four-sample validation threshold defaults to masked-token cross-entropy `0.01`.
- Use `rtk` for every shell command and `apply_patch` for all file edits.

---

### Task 1: LoRA configuration, target validation, and adapter insertion

**Files:**
- Modify: `pyproject.toml`
- Modify: `omnivoice/training/config.py`
- Create: `omnivoice/training/lora.py`
- Create: `tests/conftest.py`
- Create: `tests/training/test_lora.py`

**Interfaces:**
- Consumes: `TrainingConfig`, an initialized and tokenizer-resized `OmniVoice` model.
- Produces: `DEFAULT_LORA_TARGET_MODULES`, `validate_lora_config(config)`, `find_lora_target_modules(model, suffixes)`, `apply_lora(model, config)`, `load_lora_adapter(model, checkpoint_path, is_trainable)`, `is_lora_model(model)`, and `trainable_parameter_counts(model)`.

- [ ] **Step 1: Install test/runtime dependencies in the isolated environment**

Run:

```bash
rtk python -m pip install -e . peft pytest
```

Expected: editable OmniVoice installation succeeds and `python -c 'import peft, pytest'` exits 0.

- [ ] **Step 2: Add shared network-free model fixtures**

Create a tiny Transformers model whose real module names exercise all broad
targets:

```python
class ToyOmniConfig(PretrainedConfig):
    model_type = "toy_omnivoice"

    def __init__(self, vocab_size=32, hidden_size=8, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class ToyOmniVoice(PreTrainedModel):
    config_class = ToyOmniConfig

    def __init__(self, config=None):
        config = config or ToyOmniConfig()
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.audio_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(config.hidden_size, config.hidden_size, bias=False))
        self.audio_heads = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, audio_ids=None, labels=None, **kwargs):
        hidden = self.embed_tokens(input_ids)
        if audio_ids is not None:
            hidden = hidden + self.audio_embeddings(audio_ids)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            hidden = hidden + 0.01 * getattr(self, name)(hidden)
        logits = self.audio_heads(hidden)
        loss = logits.float().square().mean() if labels is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        return SimpleNamespace(loss=loss, logits=logits)


class DummyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self):
        return 32

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}")


@pytest.fixture
def toy_omnivoice():
    return ToyOmniVoice()
```

Import new LoRA helpers inside individual fixtures/tests so the initial RED
run fails on the missing production API rather than during test collection.

- [ ] **Step 3: Write failing configuration tests**

Add tests that express the new JSON API before production fields exist:

```python
def test_training_config_loads_broad_lora_defaults(tmp_path):
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
    config = TrainingConfig(lora_enabled=True, init_from_checkpoint="base")
    setattr(config, field, value)
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_lora_config(config)
```

- [ ] **Step 4: Run the configuration tests and verify RED**

Run:

```bash
rtk pytest -q tests/training/test_lora.py
```

Expected: FAIL because the LoRA module and `TrainingConfig` fields do not exist.

- [ ] **Step 5: Add dependency and explicit configuration fields**

Add `peft` to project dependencies and add these dataclass fields:

```python
lora_enabled: bool = False
lora_rank: int = 32
lora_alpha: int = 64
lora_dropout: float = 0.05
lora_bias: str = "none"
lora_target_modules: List[str] = field(
    default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
)
```

Keep the constant in `training/lora.py`; import it into `config.py` so the default list has one source of truth.

- [ ] **Step 6: Implement validation and target discovery**

Implement exact suffix validation before PEFT mutates the model:

```python
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
```

- [ ] **Step 7: Write the failing adapter-insertion tests**

Create a tiny `PreTrainedModel` fixture containing all ten target suffixes, then assert real PEFT behavior:

```python
def test_apply_lora_wraps_every_broad_target_and_freezes_base(toy_omnivoice):
    config = TrainingConfig(
        init_from_checkpoint="base",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
    )
    model, matches = apply_lora(toy_omnivoice, config)

    assert set(matches) == set(DEFAULT_LORA_TARGET_MODULES)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)
    assert any("audio_embeddings" in name for name in trainable)
    assert any("audio_heads" in name for name in trainable)


def test_apply_lora_rejects_missing_requested_target(toy_omnivoice):
    config = TrainingConfig(
        init_from_checkpoint="base",
        lora_enabled=True,
        lora_target_modules=["q_proj", "does_not_exist"],
    )
    with pytest.raises(ValueError, match="does_not_exist"):
        apply_lora(toy_omnivoice, config)
```

- [ ] **Step 8: Run adapter-insertion tests and verify RED**

Run:

```bash
rtk pytest -q tests/training/test_lora.py
```

Expected: configuration tests pass; adapter tests FAIL because `apply_lora` is absent.

- [ ] **Step 9: Implement adapter insertion and parameter counting**

Use the generic PEFT wrapper so custom OmniVoice modules participate:

```python
def apply_lora(model, config):
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
    trainable, total = trainable_parameter_counts(model)
    if trainable == 0:
        raise RuntimeError("LoRA produced no trainable parameters")
    return model, matches
```

`trainable_parameter_counts` returns `(trainable_count, total_count)` using `numel()`.

- [ ] **Step 10: Run Task 1 tests and verify GREEN**

Run:

```bash
rtk pytest -q tests/training/test_lora.py
```

Expected: all tests pass with no warnings from the project code.

- [ ] **Step 11: Commit Task 1**

```bash
rtk git add pyproject.toml omnivoice/training/config.py omnivoice/training/lora.py tests/conftest.py tests/training/test_lora.py
rtk git commit -m "feat: add broad LoRA model configuration"
```

---

### Task 2: Builder integration and trainable-only optimizer

**Files:**
- Modify: `omnivoice/training/builder.py`
- Modify: `omnivoice/training/trainer.py`
- Modify: `tests/training/test_lora.py`

**Interfaces:**
- Consumes: `apply_lora`, `load_lora_adapter`, `TrainingConfig.resume_from_checkpoint`.
- Produces: `_finalize_training_model(model, tokenizer, config)` applying resize, token IDs, and LoRA in order; `build_model_and_tokenizer` returning either normal `OmniVoice` or a PEFT-wrapped model; `OmniTrainer.create_optimizer_and_scheduler` filtering frozen parameters.

- [ ] **Step 1: Write failing builder-order and optimizer tests**

Exercise a focused finalization seam and record call order:

```python
def test_finalize_training_model_resizes_before_applying_lora(monkeypatch):
    calls = []
    model = ToyOmniVoice()
    original_resize = model.resize_token_embeddings
    monkeypatch.setattr(
        model,
        "resize_token_embeddings",
        lambda size: calls.append("resize") or original_resize(size),
    )
    monkeypatch.setattr(
        builder,
        "apply_lora",
        lambda value, config: (calls.append("lora") or value, {}),
    )
    config = TrainingConfig(init_from_checkpoint="base", lora_enabled=True)
    tokenizer = DummyTokenizer()
    tokenizer.__class__.__len__ = lambda self: 33
    finalized = builder._finalize_training_model(model, tokenizer, config)
    assert finalized is model
    assert calls == ["resize", "lora"]


def test_optimizer_contains_only_trainable_parameters(toy_omnivoice):
    config = TrainingConfig(init_from_checkpoint="base", lora_enabled=True, steps=2)
    model, _ = apply_lora(toy_omnivoice, config)
    trainer = object.__new__(OmniTrainer)
    trainer.model = model
    trainer.config = config
    optimizer, _ = trainer.create_optimizer_and_scheduler()
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert optimizer_ids == expected_ids
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
rtk pytest -q tests/training/test_lora.py -k 'builder or optimizer'
```

Expected: FAIL because builder does not apply LoRA and optimizer includes frozen parameters.

- [ ] **Step 3: Apply LoRA after resize and IDs are configured**

Move the existing resize and token-ID assignments into
`_finalize_training_model`, then apply LoRA at the end of that helper:

```python
if config.lora_enabled:
    if config.resume_from_checkpoint:
        model = load_lora_adapter(
            model,
            config.resume_from_checkpoint,
            config=config,
            is_trainable=True,
        )
    else:
        model, matches = apply_lora(model, config)
        logger.info("Matched LoRA modules: %s", matches)
    trainable, total = trainable_parameter_counts(model)
    logger.info(
        "LoRA trainable parameters: %d / %d (%.4f%%)",
        trainable,
        total,
        100.0 * trainable / total,
    )
```

Extract only the loading seams needed by the tests; do not change non-LoRA model semantics.

- [ ] **Step 4: Filter optimizer parameters**

Change optimizer construction to:

```python
trainable_parameters = [
    parameter for parameter in self.model.parameters() if parameter.requires_grad
]
if not trainable_parameters:
    raise RuntimeError("Model has no trainable parameters")
optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=self.config.learning_rate,
    weight_decay=self.config.weight_decay,
)
```

- [ ] **Step 5: Run Task 2 tests and the Task 1 regression suite**

Run:

```bash
rtk pytest -q tests/training/test_lora.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
rtk git add omnivoice/training/builder.py omnivoice/training/trainer.py tests/training/test_lora.py
rtk git commit -m "feat: integrate LoRA with training builder"
```

---

### Task 3: Adapter-only checkpoint saving and exact resume

**Files:**
- Modify: `omnivoice/training/lora.py`
- Modify: `omnivoice/training/checkpoint.py`
- Modify: `omnivoice/training/trainer.py`
- Create: `tests/training/test_lora_checkpoint.py`

**Interfaces:**
- Consumes: PEFT-wrapped model, `Accelerator`, `TrainingConfig`, tokenizer, `checkpoint-<step>` path.
- Produces: `register_lora_state_hooks(accelerator)`, `save_lora_adapter(...)`, `read_lora_metadata(path)`, `resolve_adapter_dir(path)`, and resumable adapter-only checkpoints.

- [ ] **Step 1: Write failing checkpoint-content test**

Use a CPU `Accelerator`, toy PEFT model, AdamW, and constant scheduler:

```python
def test_lora_checkpoint_omits_frozen_model_and_contains_adapter(tmp_path, toy_omnivoice):
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        output_dir=str(tmp_path),
        lora_enabled=True,
        steps=2,
    )
    model, _ = apply_lora(toy_omnivoice, config)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
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
```

- [ ] **Step 2: Run checkpoint test and verify RED**

Run:

```bash
rtk pytest -q tests/training/test_lora_checkpoint.py::test_lora_checkpoint_omits_frozen_model_and_contains_adapter
```

Expected: FAIL because the existing saver writes the full wrapped model and has no adapter metadata.

- [ ] **Step 3: Implement adapter metadata and checkpoint path resolution**

Use this stable metadata schema:

```json
{
  "format_version": 1,
  "base_model_name_or_path": "k2-fsa/OmniVoice",
  "step": 7,
  "lora_rank": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "lora_bias": "none",
  "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "audio_embeddings", "audio_heads"]
}
```

`resolve_adapter_dir` accepts either a checkpoint root containing `adapter/` or the adapter directory itself. `read_lora_metadata` requires the checkpoint root and validates `format_version == 1`.

- [ ] **Step 4: Register save/load suppression hooks**

Register hooks once per LoRA trainer:

```python
def register_lora_state_hooks(accelerator):
    def save_hook(models, weights, output_dir):
        weights.clear()

    def load_hook(models, input_dir):
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
```

The builder has already loaded adapter weights before resume, so the load hook only prevents Accelerate from seeking a suppressed full-model payload. Optimizer, scheduler, scaler, and RNG state continue through `Accelerator.load_state`.

- [ ] **Step 5: Make checkpoint saving branch explicitly on LoRA**

Change the saver signature to accept `config`. After `accelerator.save_state`:

```python
unwrap_model = accelerator.unwrap_model(model)
if is_lora_model(unwrap_model):
    save_lora_adapter(
        unwrap_model,
        checkpoint_dir,
        config=config,
        step=step,
        accelerator=accelerator,
    )
else:
    unwrap_model.save_pretrained(
        checkpoint_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
```

Write the adapter to a sibling temporary directory, write metadata there, and
publish it with `os.replace` only after both files are complete. Save tokenizer
and training config on the main process as before.

- [ ] **Step 6: Write failing round-trip resume test**

```python
def test_lora_checkpoint_resume_restores_adapter_and_optimizer(tmp_path, toy_omnivoice):
    checkpoint, expected_adapter, expected_optimizer = save_checkpoint_after_one_step(
        tmp_path, toy_omnivoice
    )
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        resume_from_checkpoint=str(checkpoint),
        lora_enabled=True,
        steps=2,
    )
    model = load_lora_adapter(
        ToyOmniVoice(), checkpoint, config=config, is_trainable=True
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

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    register_lora_state_hooks(accelerator)
    assert load_checkpoint(accelerator, str(checkpoint)) == 7
    assert_optimizer_state_equal(optimizer.state_dict(), expected_optimizer)
```

Define the test-only setup and comparison helpers immediately above the test:

```python
def save_checkpoint_after_one_step(tmp_path, base_model):
    config = TrainingConfig(
        init_from_checkpoint="k2-fsa/OmniVoice",
        output_dir=str(tmp_path),
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        steps=2,
    )
    model, _ = apply_lora(base_model, config)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
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
    save_checkpoint(
        accelerator,
        model,
        DummyTokenizer(),
        config,
        str(tmp_path),
        step=7,
        keep_last_n=-1,
    )
    return tmp_path / "checkpoint-7", expected_adapter, expected_optimizer


def assert_optimizer_state_equal(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id in actual["state"]:
        assert actual["state"][parameter_id].keys() == expected["state"][parameter_id].keys()
        for key, actual_value in actual["state"][parameter_id].items():
            expected_value = expected["state"][parameter_id][key]
            if isinstance(actual_value, torch.Tensor):
                torch.testing.assert_close(actual_value.cpu(), expected_value.cpu())
            else:
                assert actual_value == expected_value
```

Also add `test_full_checkpoint_still_writes_model_weights` using an unwrapped
`ToyOmniVoice` and `TrainingConfig(lora_enabled=False)`; assert its checkpoint
contains `model.safetensors` and no `adapter/` directory.

Also test that base-model or target-list metadata mismatch raises before Accelerator initialization.

- [ ] **Step 7: Run round-trip test and verify RED**

Run:

```bash
rtk pytest -q tests/training/test_lora_checkpoint.py -k 'resume or mismatch'
```

Expected: FAIL because adapter loading and metadata compatibility checks are not implemented.

- [ ] **Step 8: Implement trainable adapter loading and compatibility validation**

Use PEFT's loader over the already initialized base model:

```python
def load_lora_adapter(model, checkpoint_path, config=None, is_trainable=False):
    checkpoint_root, adapter_dir = resolve_adapter_dir(checkpoint_path)
    metadata = read_lora_metadata(checkpoint_root)
    if config is not None:
        validate_resume_metadata(config, metadata)
    return PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=is_trainable,
    )
```

Compare normalized base identifiers, rank, alpha, dropout, bias, and ordered target suffixes. Diagnostics name every differing field.

- [ ] **Step 9: Run checkpoint tests and all LoRA tests**

Run:

```bash
rtk pytest -q tests/training/test_lora.py tests/training/test_lora_checkpoint.py
```

Expected: all tests pass and checkpoints contain no frozen model file.

- [ ] **Step 10: Commit Task 3**

```bash
rtk git add omnivoice/training/lora.py omnivoice/training/checkpoint.py omnivoice/training/trainer.py tests/training/test_lora_checkpoint.py
rtk git commit -m "feat: save resumable LoRA-only checkpoints"
```

---

### Task 4: Adapter-aware inference loader

**Files:**
- Modify: `omnivoice/models/omnivoice.py`
- Modify: `omnivoice/training/lora.py`
- Create: `tests/models/test_lora_loading.py`

**Interfaces:**
- Consumes: checkpoint root containing adapter metadata and `adapter/`.
- Produces: `OmniVoice.from_lora_pretrained(adapter_checkpoint, **kwargs)` returning an inference-mode PEFT model over the recorded base model.

- [ ] **Step 1: Write failing inference round-trip test**

Avoid downloading a real checkpoint by patching the normal base loader and exercising a real PEFT adapter:

```python
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
    save_lora_adapter(trained, checkpoint, config=config, step=7)
    loaded_bases = []
    monkeypatch.setattr(
        OmniVoice,
        "from_pretrained",
        classmethod(lambda cls, path, **kwargs: loaded_bases.append(path) or ToyOmniVoice()),
    )
    model = OmniVoice.from_lora_pretrained(checkpoint, device_map="cpu")
    assert loaded_bases == ["k2-fsa/OmniVoice"]
    assert is_lora_model(model)
    assert all(not parameter.requires_grad for parameter in model.parameters())
```

- [ ] **Step 2: Run inference test and verify RED**

Run:

```bash
rtk pytest -q tests/models/test_lora_loading.py
```

Expected: FAIL because `from_lora_pretrained` does not exist.

- [ ] **Step 3: Implement the classmethod with a lazy PEFT import**

Add:

```python
@classmethod
def from_lora_pretrained(cls, adapter_checkpoint, *args, **kwargs):
    from omnivoice.training.lora import load_lora_for_inference, read_lora_metadata

    metadata = read_lora_metadata(adapter_checkpoint)
    base_model = cls.from_pretrained(
        metadata["base_model_name_or_path"],
        *args,
        **kwargs,
    )
    return load_lora_for_inference(base_model, adapter_checkpoint)
```

Keep normal `OmniVoice.from_pretrained` free of a hard PEFT import so inference without adapters retains current behavior when PEFT is unavailable in an older environment.

- [ ] **Step 4: Run loader and regression tests**

Run:

```bash
rtk pytest -q tests/models/test_lora_loading.py tests/training/test_lora.py tests/training/test_lora_checkpoint.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
rtk git add omnivoice/models/omnivoice.py omnivoice/training/lora.py tests/models/test_lora_loading.py
rtk git commit -m "feat: load OmniVoice LoRA adapters for inference"
```

---

### Task 5: Eight-GPU LoRA example and documentation

**Files:**
- Create: `examples/config/train_config_finetune_lora.json`
- Create: `examples/run_finetune_lora.sh`
- Modify: `examples/README.md`
- Modify: `docs/training.md`
- Modify: `README.md`
- Create: `tests/test_lora_examples.py`

**Interfaces:**
- Consumes: existing raw JSONL tokenization and `omnivoice.cli.train`.
- Produces: a runnable eight-GPU LoRA workflow and documented adapter inference/resume commands.

- [ ] **Step 1: Write failing example-validation tests**

```python
def test_lora_example_config_is_broad_and_adapter_only():
    config = json.loads(Path("examples/config/train_config_finetune_lora.json").read_text())
    assert config["lora_enabled"] is True
    assert config["lora_rank"] == 32
    assert set(config["lora_target_modules"]) == set(DEFAULT_LORA_TARGET_MODULES)
    assert config["init_from_checkpoint"] == "k2-fsa/OmniVoice"


def test_lora_script_defaults_to_eight_gpus():
    script = Path("examples/run_finetune_lora.sh").read_text()
    assert 'GPU_IDS="0,1,2,3,4,5,6,7"' in script
    assert "NUM_GPUS=8" in script
    assert "train_config_finetune_lora.json" in script
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
rtk pytest -q tests/test_lora_examples.py
```

Expected: FAIL because the LoRA example files do not exist.

- [ ] **Step 3: Add the broad LoRA JSON example**

Copy the existing fine-tuning hyperparameters, then add:

```json
"lora_enabled": true,
"lora_rank": 32,
"lora_alpha": 64,
"lora_dropout": 0.05,
"lora_bias": "none",
"lora_target_modules": [
  "q_proj", "k_proj", "v_proj", "o_proj",
  "gate_proj", "up_proj", "down_proj",
  "embed_tokens", "audio_embeddings", "audio_heads"
]
```

Use `learning_rate: 0.0001` as the initial large-dataset LoRA learning rate and retain bf16/flex attention defaults.

- [ ] **Step 4: Add the eight-GPU shell workflow**

Follow `run_finetune.sh` stages exactly, with these defaults:

```bash
GPU_IDS="0,1,2,3,4,5,6,7"
NUM_GPUS=8
TRAIN_CONFIG="config/train_config_finetune_lora.json"
OUTPUT_DIR="exp/omnivoice_finetune_lora"
```

Quote every path and integer expansion. Keep raw-data tokenization restartable through `stage` and `stop_stage`.

- [ ] **Step 5: Document training, resume, and inference**

Document:

```python
model = OmniVoice.from_lora_pretrained(
    "exp/omnivoice_finetune_lora/checkpoint-5000",
    device_map="cuda:0",
    dtype=torch.float16,
)
```

Explain global token batch as `batch_tokens × num_processes × gradient_accumulation_steps`, adapter checkpoint contents, target customization, and why the frozen base must stay accessible.

- [ ] **Step 6: Run example tests and shell syntax validation**

Run:

```bash
rtk pytest -q tests/test_lora_examples.py
rtk bash -n examples/run_finetune_lora.sh
```

Expected: all tests pass and Bash exits 0.

- [ ] **Step 7: Commit Task 5**

```bash
rtk git add examples/config/train_config_finetune_lora.json examples/run_finetune_lora.sh examples/README.md docs/training.md README.md tests/test_lora_examples.py
rtk git commit -m "docs: add eight-GPU LoRA fine-tuning workflow"
```

---

### Task 6: Deterministic four-utterance memorization workflow

**Files:**
- Create: `omnivoice/scripts/select_memorization_samples.py`
- Create: `omnivoice/cli/eval_memorization.py`
- Create: `examples/config/train_config_lora_memorization.json`
- Create: `examples/config/data_config_lora_memorization.json`
- Create: `examples/run_lora_memorization.sh`
- Create: `tests/scripts/test_select_memorization_samples.py`
- Create: `tests/cli/test_eval_memorization.py`

**Interfaces:**
- Consumes: a raw JSONL with `id`, `audio_path`, and `text`; a completed LoRA checkpoint.
- Produces: `select_records(input_path, count, seed)`, a four-row manifest, a deterministic evaluation CLI, and a single-GPU overfit script.

- [ ] **Step 1: Write failing deterministic-selection tests**

```python
def test_select_records_returns_four_unique_valid_records(tmp_path):
    manifest = write_manifest_with_valid_invalid_and_duplicate_rows(tmp_path)
    first = select_records(manifest, count=4, seed=42)
    second = select_records(manifest, count=4, seed=42)
    assert first == second
    assert len(first) == 4
    assert len({row["id"] for row in first}) == 4
    assert all(Path(row["audio_path"]).is_file() for row in first)


def test_select_records_reports_available_count(tmp_path):
    manifest = write_manifest(tmp_path, valid_count=3)
    with pytest.raises(ValueError, match="found 3 valid unique records; need 4"):
        select_records(manifest, count=4, seed=42)
```

- [ ] **Step 2: Run selection tests and verify RED**

Run:

```bash
rtk pytest -q tests/scripts/test_select_memorization_samples.py
```

Expected: FAIL because the selection module does not exist.

- [ ] **Step 3: Implement selection CLI**

Validation rules are exact: row is a JSON object; `id`, `audio_path`, and non-empty `text` are present; ID is unique; audio path resolves to an existing regular file. Shuffle valid rows with `random.Random(seed).shuffle`, select exactly `count`, and write UTF-8 JSONL with `ensure_ascii=False`.

CLI:

```text
python -m omnivoice.scripts.select_memorization_samples \
  --input-jsonl SOURCE \
  --output-jsonl data/lora_memorization/four.jsonl \
  --count 4 \
  --seed 42
```

- [ ] **Step 4: Run selection tests and verify GREEN**

Run:

```bash
rtk pytest -q tests/scripts/test_select_memorization_samples.py
```

Expected: all tests pass.

- [ ] **Step 5: Write failing deterministic evaluator tests**

Test the threshold function without loading the large model:

```python
def test_check_memorization_accepts_loss_at_threshold():
    assert check_memorization(loss=0.01, threshold=0.01) == 0


def test_check_memorization_rejects_loss_above_threshold():
    assert check_memorization(loss=0.0101, threshold=0.01) == 1


def test_mean_eval_loss_is_weighted_by_batch_count():
    class FakeModel(nn.Module):
        def forward(self, loss, **kwargs):
            return SimpleNamespace(loss=loss)

    batches = [
        {"loss": torch.tensor(0.2)},
        {"loss": torch.tensor(0.3)},
    ]
    loss, num_batches = mean_eval_loss(
        FakeModel(), batches, device="cpu", dtype=torch.float32
    )
    assert loss == pytest.approx(0.25)
    assert num_batches == 2
```

- [ ] **Step 6: Run evaluator tests and verify RED**

Run:

```bash
rtk pytest -q tests/cli/test_eval_memorization.py
```

Expected: FAIL because the evaluator functions do not exist.

- [ ] **Step 7: Implement deterministic adapter evaluation**

The CLI accepts:

```text
--adapter-checkpoint PATH
--train-config PATH
--data-config PATH
--threshold FLOAT
--device cuda:0
```

It loads the train config, sets `resume_from_checkpoint` to the adapter checkpoint, builds model/tokenizer/dataloaders, evaluates the dev loader under `torch.inference_mode()` and the configured autocast dtype, prints a JSON object containing `loss`, `threshold`, `passed`, and `num_batches`, and exits 1 when the threshold is missed. Reject empty dev data and non-positive thresholds.

- [ ] **Step 8: Add fixed memorization configs and orchestration script**

Use a single GPU to avoid four WebDataset shards being exhausted unevenly across DDP ranks. Training config differences from broad LoRA defaults:

```json
"lora_rank": 64,
"lora_alpha": 128,
"lora_dropout": 0.0,
"drop_cond_ratio": 0.0,
"prompt_ratio_range": [0.3, 0.3],
"mask_ratio_range": [1.0, 1.0],
"language_ratio": 1.0,
"instruct_ratio": 0.0,
"learning_rate": 0.001,
"steps": 2000,
"attn_implementation": "sdpa",
"batch_tokens": 4096,
"max_batch_size": 4,
"eval_steps": 100,
"save_steps": 500
```

The data config references the same generated `data.lst` for train and dev. The shell script selects four records, tokenizes them once, trains, then evaluates `checkpoint-2000` with threshold `0.01`. It requires `SOURCE_JSONL` and fails before creating outputs if the variable is missing or invalid.

- [ ] **Step 9: Run memorization utility tests and shell syntax check**

Run:

```bash
rtk pytest -q tests/scripts/test_select_memorization_samples.py tests/cli/test_eval_memorization.py
rtk bash -n examples/run_lora_memorization.sh
```

Expected: all tests pass and Bash exits 0.

- [ ] **Step 10: Commit Task 6**

```bash
rtk git add omnivoice/scripts/select_memorization_samples.py omnivoice/cli/eval_memorization.py examples/config/train_config_lora_memorization.json examples/config/data_config_lora_memorization.json examples/run_lora_memorization.sh tests/scripts/test_select_memorization_samples.py tests/cli/test_eval_memorization.py
rtk git commit -m "test: add four-sample LoRA memorization workflow"
```

---

### Task 7: Full regression and available integration validation

**Files:**
- Modify only if a failing test exposes a documented defect; add a regression test before each correction.

**Interfaces:**
- Consumes: all implementation tasks and an optional user-supplied `SOURCE_JSONL`.
- Produces: fresh verification evidence and an explicit record of whether empirical memorization was run.

- [ ] **Step 1: Run the complete test suite**

Run:

```bash
rtk pytest -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run static and syntax checks**

Run:

```bash
rtk python -m compileall -q omnivoice tests
rtk ruff check omnivoice tests
rtk bash -n examples/run_finetune_lora.sh
rtk bash -n examples/run_lora_memorization.sh
rtk git diff --check
```

Expected: every command exits 0.

- [ ] **Step 3: Run a CPU adapter round-trip smoke check**

Run:

```bash
rtk pytest -q tests/training/test_lora_checkpoint.py tests/models/test_lora_loading.py
```

Expected: adapter save, resume, and inference reload tests pass without a full-model payload.

- [ ] **Step 4: Run the four-sample GPU memorization check when data is available**

Run:

```bash
SOURCE_JSONL=/absolute/path/to/source.jsonl rtk bash examples/run_lora_memorization.sh
```

Expected: the final evaluator prints JSON with `"passed": true`, loss at most `0.01`, and exits 0. If no source manifest is available, record this check as blocked by missing test data without claiming empirical memorization success.

- [ ] **Step 5: Inspect final repository state and requirements**

Run:

```bash
rtk git status --short
rtk git diff HEAD~6 --stat
rtk git log --oneline -7
```

Expected: only planned files are changed, every acceptance criterion maps to passing evidence, and commit history is task-focused.

- [ ] **Step 6: Commit any test-driven corrections**

When Step 1–4 required a correction, stage only its production file and regression test, then commit:

```bash
rtk git commit -m "fix: correct LoRA integration validation"
```

Skip this step when the worktree is clean.
