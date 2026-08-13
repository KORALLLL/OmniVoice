#!/usr/bin/env python3
"""Full hard-number validation: fixed voices × every benchmark text."""
import argparse
import io
import importlib.util
import json
import os
import random
import re
import tarfile
import tempfile
from pathlib import Path

import gigaam
import soundfile as sf
import torch
import numpy as np
from peft import PeftModel
from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt
from omnivoice.training.checkpoint import load_lora_audio_modules, load_lora_trainable_modules


def metric_lib():
    relative = Path("ml-intern-runs/omnivoice-lora-gigaam-4epoch/hard_number_eval/scripts/eval_tts.py")
    candidates = [
        Path(os.environ["OMNIVOICE_HARD_EVAL_METRICS"])
        if os.environ.get("OMNIVOICE_HARD_EVAL_METRICS") else None,
        Path(__file__).parents[2] / relative,
        Path("/workspace/OmniVoice") / relative,
    ]
    path = next((candidate for candidate in candidates if candidate and candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("hard-number metric helper eval_tts.py was not found")
    spec = importlib.util.spec_from_file_location("number_metrics", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def score_hard_number_row(row, hypothesis, metrics):
    """Score ASR output in a representation compatible with the benchmark.

    GigaAM normally writes spoken Russian numbers as decimal digits.  Comparing
    that directly with ``normalized_gold`` (which deliberately spells numbers
    out for TTS) makes a correct reading score as a complete substitution.  If
    the ASR emitted digits, score the number region against the benchmark's
    original digit value and replace the reference's spelled-out number span by
    that same digit value for whole-utterance metrics.  Keep the established
    word-form scorer when the ASR itself returns word-form numbers.
    """
    if not re.search(r"\d", hypothesis):
        score = metrics.score_row(row, hypothesis)
        score["metric_protocol"] = "word_form"
        return score

    expected_digits = "".join(re.findall(r"\d", str(row["hard_number"])))
    observed_digits = "".join(re.findall(r"\d", hypothesis))
    if not expected_digits:
        raise RuntimeError(f"Benchmark row {row['id']} has no numeric reference")

    ref_number_words, start, end = metrics.number_span(row["input"], row["gold"])
    gold_tokens = metrics.toks(row["gold"])
    if start is None:
        canonical_gold = row["gold"]
    else:
        canonical_gold = " ".join(gold_tokens[:start] + [expected_digits] + gold_tokens[end + 1 :])

    # ASR may insert spaces into a single value (e.g. ``408 178 103``).
    # Collapse adjacent digit groups only; ordinary surrounding text remains
    # untouched for whole-utterance WER/CER.
    canonical_hyp = re.sub(
        r"\d+(?:\s+\d+)+",
        lambda match: "".join(re.findall(r"\d+", match.group(0))),
        hypothesis,
    )
    u_wer, u_ref_n = metrics.wer(canonical_gold, canonical_hyp)
    u_cer, _ = metrics.cer(canonical_gold, canonical_hyp)
    digit_ref = list(expected_digits)
    digit_hyp = list(observed_digits)
    n_sdi = metrics.sdi(digit_ref, digit_hyp)
    n_err = n_sdi["S"] + n_sdi["D"] + n_sdi["I"]
    n_rate = n_err / n_sdi["N"] if n_sdi["N"] else 0.0
    return {
        "u_wer": u_wer,
        "u_cer": u_cer,
        "n_wer": n_rate,
        "n_cer": n_rate,
        "ref_num": expected_digits,
        "hyp_num": observed_digits,
        "u_ref_n": u_ref_n,
        "n_ref_n": n_sdi["N"],
        "nw_sdi": n_sdi,
        "nc_sdi": n_sdi,
        "metric_protocol": "digit_canonical",
    }


def select_voices(manifest, selection, count, seed, excluded_speaker_keys=None):
    if selection.exists():
        voices = json.loads(selection.read_text(encoding="utf-8"))["voices"]
        if len(voices) == count: return voices
        raise RuntimeError("invalid persisted voice selection")
    pairs = [(Path(line.split()[0]), Path(line.split()[1])) for line in manifest.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(pairs)
    excluded_speaker_keys = excluded_speaker_keys or set()
    # A manifest is a list of shards, not a list of utterances.  Inspect every
    # sidecar row to obtain a representative candidate for each speaker; using
    # only the first row silently capped a six-shard dev split at six voices.
    candidates = []
    seen_candidates = set()
    id_re = re.compile(r'"id"\s*:\s*("(?:\\\\.|[^"\\\\])*")')
    podcast_re = re.compile(r'"podcast_id"\s*:\s*([^,}]+)')
    speaker_re = re.compile(r'"speaker_id"\s*:\s*([^,}]+)')
    ref_text_re = re.compile(r'"ref_text"\s*:\s*("(?:\\\\.|[^"\\\\])*")')
    for archive, label in pairs:
        with label.open(encoding="utf-8") as handle:
            for line in handle:
                # Sidecars also carry very large tokenizer payloads which are
                # irrelevant for selecting a reference.  Extract only the four
                # small fields needed here, avoiding a multi-gigabyte transient
                # allocation while scanning the held-out shards.
                sample_match = id_re.search(line)
                podcast_match = podcast_re.search(line)
                speaker_match = speaker_re.search(line)
                ref_text_match = ref_text_re.search(line)
                if not (sample_match and podcast_match and speaker_match and ref_text_match):
                    continue
                sample_id = json.loads(sample_match.group(1))
                podcast, speaker = podcast_match.group(1), speaker_match.group(1)
                ref_text = json.loads(ref_text_match.group(1))
                key = f"{podcast}:{speaker}"
                if (not ref_text or key in excluded_speaker_keys
                        or key in seen_candidates):
                    continue
                candidates.append((archive, sample_id, key, ref_text))
                seen_candidates.add(key)
    random.Random(seed).shuffle(candidates)
    voices = []
    for archive, sample_id, key, ref_text in candidates:
        try:
            with tarfile.open(archive) as tar:
                member = tar.extractfile(f"{sample_id}.npy")
                audio_codes = np.load(io.BytesIO(member.read())).tolist()
        except Exception:
            continue
        if (len(audio_codes) != 8 or not audio_codes
                or not all(channel for channel in audio_codes)
                or any(token < 0 or token >= 1024 for channel in audio_codes for token in channel)):
            continue
        voices.append({"speaker_key": key, "ref_text": ref_text, "audio_codes": audio_codes})
        if len(voices) == count: break
    if len(voices) != count: raise RuntimeError(f"found {len(voices)} voices, expected {count}")
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text(json.dumps({"seed": seed, "voices": voices}, ensure_ascii=False), encoding="utf-8")
    return voices


def aggregate(rows, metric, length):
    numerator = sum(row[metric] * row[length] for row in rows if row.get(metric) is not None)
    denominator = sum(row[length] for row in rows if row.get(metric) is not None)
    return numerator / denominator if denominator else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--dataset-path", required=True)
    p.add_argument("--voice-manifest-path", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--selection-path", required=True); p.add_argument("--num-voices", type=int, default=40)
    p.add_argument("--selection-seed", type=int, default=42); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--excluded-speaker-keys-path", default=None)
    p.add_argument("--gigaam-model", default="v3_e2e_rnnt"); p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--samples-per-voice", type=int, default=0)
    # generate() defaults denoise=True, which prepends <|denoise|> whenever a
    # reference audio is present - i.e. on every utterance this script scores.
    # Training only emits that token for labels carrying clean_start_token_idx,
    # a key absent from the whole corpus, so the fine-tune has never seen it.
    # --no-denoise removes the mismatch.
    p.add_argument("--denoise", dest="denoise", action="store_true", default=True)
    p.add_argument("--no-denoise", dest="denoise", action="store_false")
    p.add_argument("--generation-seed", type=int, default=42)
    p.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate the unadapted init_from_checkpoint model.  This is used "
        "only for an apples-to-apples held-out baseline; no adapter or audio "
        "sidecar is loaded.",
    )
    args = p.parse_args(); metrics = metric_lib()
    rows = []
    for line in Path(args.dataset_path).read_text(encoding="utf-8").splitlines():
        source = json.loads(line); rows.append({"id": source["id"], "input": source["text"], "gold": source["normalized_gold"], "stressed": source["stressed"], "hard_number": source["hard_number"]})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    excluded_speakers = set()
    if args.excluded_speaker_keys_path:
        excluded_speakers = set(json.loads(Path(args.excluded_speaker_keys_path).read_text(encoding="utf-8")))
    voices = select_voices(Path(args.voice_manifest_path), Path(args.selection_path), args.num_voices, args.selection_seed, excluded_speakers)
    config = json.loads((Path(args.checkpoint) / "train_config.json").read_text())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = OmniVoice.from_pretrained(config["init_from_checkpoint"], dtype=torch.bfloat16 if device == "cuda:0" else torch.float32)
    if not args.base_only:
        adapter_dir = Path(args.checkpoint) / "lora_adapter"
        if adapter_dir.is_dir():
            model.llm = PeftModel.from_pretrained(model.llm, adapter_dir)
            load_lora_audio_modules(model, str(adapter_dir))
            load_lora_trainable_modules(model, str(adapter_dir))
        else:
            # Full fine-tuning (lora_rank=0) writes an Accelerate checkpoint
            # with the whole model in model.safetensors and no adapter sidecar.
            # Without this branch the eval dies on a missing adapter_config.json
            # and takes every training rank down with it.
            weights = Path(args.checkpoint) / "model.safetensors"
            if not weights.is_file():
                raise SystemExit(
                    f"{args.checkpoint} has neither lora_adapter/ nor model.safetensors"
                )
            from safetensors.torch import load_file as _load_safetensors

            state = _load_safetensors(str(weights), device="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"full-checkpoint load: {len(missing)} missing, "
                      f"{len(unexpected)} unexpected keys", flush=True)
    model.to(device); model.audio_tokenizer.to(device)
    asr = gigaam.load_model(args.gigaam_model, device=device, fp16_encoder=device == "cuda:0")
    results_path = out / "per_utt.jsonl"
    results = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    # A process may be interrupted while writing the final line.
                    break
    completed = {(r["voice_index"], r["benchmark_id"]) for r in results}
    examples = out / "audio_examples"; examples.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hard-number-") as tmp:
        shuffled_rows = list(rows); random.Random(args.selection_seed).shuffle(shuffled_rows)
        for vi, voice in enumerate(voices):
            ref_codes = torch.tensor(voice["audio_codes"], dtype=torch.long)
            prompt = VoiceClonePrompt(ref_audio_tokens=ref_codes, ref_text=voice["ref_text"], ref_rms=0.1)
            voice_rows = (shuffled_rows[vi * args.samples_per_voice:(vi + 1) * args.samples_per_voice]
                          if args.samples_per_voice else rows)
            for start in range(0, len(voice_rows), args.batch_size):
                batch = [r for r in voice_rows[start:start + args.batch_size]
                         if (vi, r["id"]) not in completed]
                if args.max_pairs: batch = batch[:max(0, args.max_pairs - len(results))]
                # Resume may leave completed and incomplete batches interleaved
                # (for example after an interruption while a two-item batch is
                # being written).  Skip this batch only; breaking here silently
                # omits all later benchmark rows for the same voice.
                if not batch: continue
                torch.manual_seed(args.generation_seed + vi * 100000 + start)
                # Let the model estimate an appropriate duration per utterance.
                # A fixed eight-second target speeds up long number expressions
                # (and can truncate them), which turns this into a duration test
                # rather than a text-fidelity validation.
                audio_batch = model.generate(text=[r["stressed"] for r in batch], language=["ru"] * len(batch), voice_clone_prompt=[prompt] * len(batch), num_step=32, class_temperature=0.0, denoise=args.denoise)
                for row, audio in zip(batch, audio_batch):
                    fallback = ""
                    if len(audio) == 0:
                        torch.manual_seed(args.generation_seed + vi * 100000 + row["id"])
                        retry = model.generate(text=[row["stressed"]], language=["ru"], voice_clone_prompt=[prompt], duration=[12.0], num_step=32, class_temperature=0.0, denoise=args.denoise)
                        audio = retry[0]
                    if len(audio) == 0:
                        # The decoder emitted frames but silence trimming removed all
                        # samples. Preserve the raw decoded waveform: it is a valid
                        # (and appropriately poor, if silent) evaluation output.
                        torch.manual_seed(args.generation_seed + vi * 100000 + row["id"] + 1)
                        retry = model.generate(text=[row["stressed"]], language=["ru"], voice_clone_prompt=[prompt], duration=[12.0], num_step=32, class_temperature=0.0, denoise=args.denoise, postprocess_output=False)
                        audio = retry[0]; fallback = "raw_untrimmed"
                    if len(audio) == 0:
                        raise RuntimeError(f"No decoder output for benchmark row {row['id']}")
                    # Keep only the current waveform while it is being decoded.
                    # A full validation has 2,000 generations; retaining every
                    # temporary WAV fills the relatively small root filesystem
                    # and aborts an otherwise healthy evaluation.
                    wav = Path(tmp) / f"{vi}-{row['id']}.wav"
                    chunk_wavs = []
                    sf.write(wav, audio, model.sampling_rate)
                    try:
                        try:
                            transcript = asr.transcribe(str(wav)).text
                        except ValueError as error:
                            if "Too long wav file" not in str(error):
                                raise
                            # Keep validation self-contained: GigaAM's optional
                            # ``transcribe_longform`` path requires pyannote, which
                            # is not part of this training environment.  Decode
                            # contiguous safe-length chunks instead.  It avoids an
                            # OOM-prone batch and makes no external VAD dependency
                            # a prerequisite for a full checkpoint validation.
                            samples, sample_rate = sf.read(wav, dtype="float32")
                            # GigaAM accepts strictly less than 25 seconds.
                            # 24 seconds minimizes split boundaries while keeping
                            # a one-second safety margin.
                            chunk_size = int(sample_rate * 24)
                            chunks = []
                            for chunk_index, start in enumerate(range(0, len(samples), chunk_size)):
                                chunk_wav = Path(tmp) / f"{vi}-{row['id']}-asr-{chunk_index}.wav"
                                chunk_wavs.append(chunk_wav)
                                sf.write(chunk_wav, samples[start : start + chunk_size], sample_rate)
                                chunks.append(asr.transcribe(str(chunk_wav)).text)
                            transcript = " ".join(chunks)
                    finally:
                        # The transcript is the sole input needed after this
                        # point, so removing files here bounds disk use to a
                        # single generated utterance and its ASR chunks.
                        wav.unlink(missing_ok=True)
                        for chunk_wav in chunk_wavs:
                            chunk_wav.unlink(missing_ok=True)
                    score = score_hard_number_row(row, transcript, metrics); score.update(benchmark_id=row["id"], voice_index=vi, speaker_key=voice["speaker_key"], generation_fallback=fallback)
                    if len(results) < 8: sf.write(examples / f"voice-{vi:02d}-id-{row['id']}.wav", audio, model.sampling_rate)
                    results.append(score); completed.add((vi, row["id"]))
                    with results_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(score, ensure_ascii=False) + "\n")
                if args.max_pairs and len(results) >= args.max_pairs: break
            if args.max_pairs and len(results) >= args.max_pairs: break
    expected_pairs = (len(voices) * args.samples_per_voice if args.samples_per_voice else len(rows) * len(voices))
    out_metrics = {"metric_protocol": "digit_canonical_when_asr_emits_digits", "num_utt": len(results), "num_texts": len(rows), "num_voices": len(voices), "samples_per_voice": args.samples_per_voice or len(rows), "expected_pairs": expected_pairs, "is_full_validation": len(results) == expected_pairs, "utt_wer": aggregate(results, "u_wer", "u_ref_n"), "utt_cer": aggregate(results, "u_cer", "u_ref_n"), "num_wer": aggregate(results, "n_wer", "n_ref_n"), "num_cer": aggregate(results, "n_cer", "n_ref_n")}
    (out / "metrics.json").write_text(json.dumps(out_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out_metrics, ensure_ascii=False))


if __name__ == "__main__": main()
