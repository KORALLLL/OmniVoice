#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builders for constructing training components.

Provides factory functions to assemble the model, tokenizer, and data loaders
from a ``TrainingConfig``. Called by ``omnivoice.cli.train`` to set up training.

Key functions:
- ``build_model_and_tokenizer()``: Loads the model and text tokenizer.
- ``build_dataloaders()``: Builds train/eval data loaders from a data config JSON.
  The batching strategy is chosen based on ``TrainingConfig.attn_implementation``:

  - ``"flex_attention"``: sequence packing via ``PackingIterableDataset`` +
    ``PackingDataCollator``. Batch shape is ``[1, C, batch_tokens]``.
  - other (e.g. ``"sdpa"``): length-grouped padding via
    ``StreamLengthGroupDataset`` + ``PaddingDataCollator``. Batch shape
    is ``[B, C, max_len]`` where B ≥ 1 and max_len ≤ batch_tokens.
"""

import logging
import os
import re
from functools import partial
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import logging as hf_logging
from transformers.trainer_utils import seed_worker

from omnivoice.data.batching import PackingIterableDataset, StreamLengthGroupDataset
from omnivoice.data.collator import PackingDataCollator, PaddingDataCollator
from omnivoice.data.dataset import WebDatasetReader, prepare_data_manifests_from_json
from omnivoice.data.processor import OmniVoiceSampleProcessor
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig, _resolve_model_path
from omnivoice.training.config import TrainingConfig

logger = logging.getLogger(__name__)


def build_model_and_tokenizer(
    config: TrainingConfig,
) -> Tuple[OmniVoice, AutoTokenizer]:
    """Load Tokenizer and Model, handle resizing and special tokens."""
    logger.info("Initializing Model & Tokenizer...")

    # 1. Tokenizer
    tokenizer_path = (
        config.init_from_checkpoint
        if config.init_from_checkpoint
        else config.llm_name_or_path
    )
    tokenizer_path = _resolve_model_path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    new_tokens = [
        "<|denoise|>",
        "<|lang_start|>",
        "<|lang_end|>",
        "<|instruct_start|>",
        "<|instruct_end|>",
        "<|text_start|>",
        "<|text_end|>",
    ]

    tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    if config.init_from_checkpoint:
        logger.info(f"Loading weights from {config.init_from_checkpoint}")
        model = OmniVoice.from_pretrained(
            config.init_from_checkpoint,
            attn_implementation=config.attn_implementation,
            dtype=torch.float32,
            train=True,
        )
    else:
        resolved_llm = _resolve_model_path(config.llm_name_or_path)
        llm_config = AutoConfig.from_pretrained(resolved_llm)

        ov_config = OmniVoiceConfig(
            audio_vocab_size=config.audio_vocab_size,
            audio_mask_id=config.audio_mask_id,
            num_audio_codebook=config.num_audio_codebook,
            audio_codebook_weights=config.audio_codebook_weights,
            llm_config=llm_config,
        )

        original_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # suppress expected lm_head.weight warnings

        llm = AutoModel.from_pretrained(
            resolved_llm,
            attn_implementation=config.attn_implementation,
            dtype=torch.float32,
        )

        hf_logging.set_verbosity(original_level)
        model = OmniVoice(config=ov_config, llm=llm)

    # 3. Resize Embeddings
    if len(tokenizer) != model.config.llm_config.vocab_size:
        model.llm.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)

    if config.lora_rank > 0:
        from peft import LoraConfig, PeftModel, get_peft_model
        from omnivoice.training.checkpoint import (
            load_lora_audio_modules,
            load_lora_trainable_modules,
        )

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
        )
        adapter_dir = None
        # Lightweight LoRA checkpoints intentionally omit a duplicate frozen
        # base model.  If their adapter sidecar exists, initialize from it;
        # a full Accelerate-state restore can still overwrite it afterwards.
        if config.resume_from_checkpoint:
            candidate = os.path.join(config.resume_from_checkpoint, "lora_adapter")
            if os.path.isdir(candidate):
                adapter_dir = candidate
        model.llm = (
            PeftModel.from_pretrained(model.llm, adapter_dir, is_trainable=True)
            if adapter_dir else get_peft_model(model.llm, lora_config)
        )
        if adapter_dir:
            load_lora_audio_modules(model, adapter_dir)
        # Audio modules are outside the PEFT wrapper and default to trainable.
        # Explicitly respect the LoRA-only setting.
        for module in (model.audio_embeddings, model.audio_heads):
            for parameter in module.parameters():
                parameter.requires_grad_(bool(config.lora_train_audio_modules))
        last_n_layers = int(getattr(config, "lora_train_last_n_layers", 0) or 0)
        if last_n_layers > 0:
            layer_matches = []
            for name, parameter in model.llm.named_parameters():
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if match:
                    layer_matches.append((int(match.group(1)), parameter))
            if not layer_matches:
                raise RuntimeError("Could not locate transformer layers for LoRA suffix tuning")
            first_trainable_layer = max(index for index, _ in layer_matches) - last_n_layers + 1
            suffix_parameters = 0
            for index, parameter in layer_matches:
                if index >= first_trainable_layer:
                    parameter.requires_grad_(True)
                    suffix_parameters += parameter.numel()
            logger.info(
                "Enabled full training for final %d transformer layers (%d parameters)",
                last_n_layers,
                suffix_parameters,
            )
        if adapter_dir:
            load_lora_trainable_modules(model, adapter_dir)

    # 4. Config IDs
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def build_dataloaders(
    config: TrainingConfig, tokenizer: AutoTokenizer
) -> Tuple[DataLoader, DataLoader]:
    """Setup Data Pipeline: Manifests -> WDS -> Batching -> Loaders.

    Batching strategy depends on ``config.attn_implementation``:
    - ``"flex_attention"``: sequence packing (PackingIterableDataset +
      PackingDataCollator). All samples are concatenated into one long sequence.
    - other (e.g. ``"sdpa"``): length-grouped padding
      (LengthGroupedIterableDataset + PaddingDataCollator). Samples with
      similar token lengths are batched together and padded to the same length.
    """
    logger.info("Initializing Data Readers...")

    processor = OmniVoiceSampleProcessor(
        text_tokenizer=tokenizer,
        num_channels=config.num_audio_codebook,
        audio_mask_id=config.audio_mask_id,
        prompt_ratio_range=config.prompt_ratio_range,
        mask_ratio_range=config.mask_ratio_range,
        drop_cond_ratio=config.drop_cond_ratio,
        language_ratio=config.language_ratio,
        use_pinyin_ratio=config.use_pinyin_ratio,
        instruct_ratio=config.instruct_ratio,
        only_instruct_ratio=config.only_instruct_ratio,
    )

    train_manifests, dev_manifests = prepare_data_manifests_from_json(
        config.data_config
    )
    raw_train_ds = WebDatasetReader(
        manifests=train_manifests,
        evaluation=False,
        label_text_regex=config.train_label_text_regex,
        label_text_keep_ratio=getattr(config, "train_label_text_keep_ratio", 0.0),
    )
    if getattr(config, "train_epoch_sample_count", None):
        raw_train_ds.num_items = int(config.train_epoch_sample_count)
        logger.info(
            "Using effective filtered epoch sample count: %d",
            raw_train_ds.num_items,
        )

    use_packing = config.attn_implementation == "flex_attention"

    if use_packing:
        train_dataset = PackingIterableDataset(
            raw_train_ds, processor, config.batch_tokens
        )
        collate_fn = PackingDataCollator(processor, config.batch_tokens)
    else:
        train_dataset = StreamLengthGroupDataset(
            raw_train_ds,
            batch_duration=config.batch_tokens,
            min_length=config.min_sample_tokens,
            max_length=config.max_sample_tokens,
            max_sample=config.max_batch_size,
            processor=processor,
            length_fn=lambda s: s["length"],
        )
        collate_fn = PaddingDataCollator(processor, config.batch_tokens)

    logger.info(
        "Using %s (attn_implementation=%s)",
        "sequence packing" if use_packing else "length-grouped padding",
        config.attn_implementation,
    )

    init_fn = partial(
        seed_worker,
        num_workers=config.num_workers,
        rank=(
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
        pin_memory=config.pin_memory,
        prefetch_factor=4,
    )

    eval_loader = None
    if dev_manifests:
        raw_dev_ds = WebDatasetReader(manifests=dev_manifests, evaluation=True)
        if use_packing:
            dev_dataset = PackingIterableDataset(
                raw_dev_ds, processor, config.batch_tokens
            )
        else:
            dev_dataset = StreamLengthGroupDataset(
                raw_dev_ds,
                batch_duration=config.batch_tokens,
                min_length=config.min_sample_tokens,
                max_length=config.max_sample_tokens,
                max_sample=config.max_batch_size,
                processor=processor,
                length_fn=lambda s: s["length"],
            )
        eval_loader = DataLoader(
            dev_dataset,
            batch_size=None,  # Each item is already a collated batch
            num_workers=1,
            collate_fn=collate_fn,
            pin_memory=config.pin_memory,
            prefetch_factor=2,
        )

    return train_loader, eval_loader
