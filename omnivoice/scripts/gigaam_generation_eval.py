#!/usr/bin/env python3
"""Generate a fixed held-out Russian probe and score it with GigaAM ASR."""

import argparse
import json
import re
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel

import gigaam
from omnivoice.data.dataset import prepare_data_manifests_from_json
from omnivoice.models.omnivoice import OmniVoice
from omnivoice.training.checkpoint import load_lora_audio_modules, load_lora_trainable_modules


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, 1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def normalise(text: str) -> str:
    # Balalaika marks lexical stress with a plus sign; it is not a word break.
    text = text.replace("+", "")
    text = gigaam.normalize_raw_text(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def load_probe_texts(data_config: str, limit: int) -> list[str]:
    _, dev_manifests = prepare_data_manifests_from_json(data_config)
    texts: list[str] = []
    seen: set[str] = set()
    for _, label_path, _, _ in dev_manifests:
        with open(label_path, encoding="utf-8") as labels:
            for line in labels:
                record = json.loads(line)
                text = record.get("text", "")
                clean = normalise(text)
                if 2 <= len(clean.split()) <= 10 and len(clean) <= 90 and clean not in seen:
                    texts.append(text)
                    seen.add(clean)
                if len(texts) == limit:
                    return texts
    if len(texts) < limit:
        raise RuntimeError(f"Only found {len(texts)} short deterministic probe texts")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--gigaam-model", default="v3_e2e_rnnt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = Path(args.checkpoint) / "lora_adapter"
    train_config_path = Path(args.checkpoint) / "train_config.json"
    if train_config_path.is_file():
        base_checkpoint = json.loads(train_config_path.read_text())["init_from_checkpoint"]
    else:
        with open(adapter_dir / "omnivoice_lora.json") as manifest_file:
            base_checkpoint = json.load(manifest_file)["base_omnivoice_checkpoint"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = OmniVoice.from_pretrained(base_checkpoint, dtype=dtype)
    model.llm = PeftModel.from_pretrained(model.llm, adapter_dir)
    load_lora_audio_modules(model, str(adapter_dir))
    load_lora_trainable_modules(model, str(adapter_dir))
    model.to(device)
    model.audio_tokenizer.to(device)
    asr = gigaam.load_model(args.gigaam_model, device=device, fp16_encoder=True)

    references = load_probe_texts(args.data_config, args.num_samples)
    sample_rows = []
    total_word_errors = total_words = total_char_errors = total_chars = 0
    for index, reference in enumerate(references):
        audio = model.generate(
            text=reference,
            language="ru",
            duration=4.0,
            num_step=32,
            class_temperature=0.0,
        )[0]
        audio_path = output_dir / f"sample-{index:02d}.wav"
        sf.write(audio_path, audio, model.sampling_rate)
        hypothesis = asr.transcribe(str(audio_path)).text
        ref_norm, hyp_norm = normalise(reference), normalise(hypothesis)
        word_errors = edit_distance(ref_norm.split(), hyp_norm.split())
        char_errors = edit_distance(list(ref_norm.replace(" ", "")), list(hyp_norm.replace(" ", "")))
        total_word_errors += word_errors
        total_words += max(1, len(ref_norm.split()))
        total_char_errors += char_errors
        total_chars += max(1, len(ref_norm.replace(" ", "")))
        sample_rows.append(
            {
                "audio_path": str(audio_path),
                "reference": ref_norm,
                "hypothesis": hyp_norm,
                "word_errors": word_errors,
                "char_errors": char_errors,
            }
        )

    metrics = {
        "wer": total_word_errors / total_words,
        "cer": total_char_errors / total_chars,
        "num_utt": len(sample_rows),
        "num_word_errors": total_word_errors,
        "num_words": total_words,
        "num_char_errors": total_char_errors,
        "num_chars": total_chars,
        "samples": sample_rows,
    }
    with open(output_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
