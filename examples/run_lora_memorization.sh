#!/bin/bash

# Deterministically overfit four valid utterances with a single-GPU LoRA run.

set -euo pipefail

if [ -z "${SOURCE_JSONL:-}" ]; then
    echo "SOURCE_JSONL must name a raw JSONL manifest" >&2
    exit 2
fi
if [ ! -f "${SOURCE_JSONL}" ]; then
    echo "SOURCE_JSONL is not a regular file: ${SOURCE_JSONL}" >&2
    exit 2
fi
SOURCE_JSONL="$(cd "$(dirname "${SOURCE_JSONL}")" && pwd)/$(basename "${SOURCE_JSONL}")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

SELECTED_JSONL="${REPO_ROOT}/data/lora_memorization/four.jsonl"
TOKEN_DIR="${REPO_ROOT}/data/lora_memorization/tokens"
TRAIN_CONFIG="${SCRIPT_DIR}/config/train_config_lora_memorization.json"
DATA_CONFIG="${SCRIPT_DIR}/config/data_config_lora_memorization.json"
OUTPUT_DIR="${REPO_ROOT}/exp/omnivoice_lora_memorization"
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

python -m omnivoice.scripts.select_memorization_samples \
    --input-jsonl "${SOURCE_JSONL}" \
    --output-jsonl "${SELECTED_JSONL}" \
    --count 4 \
    --seed 42

CUDA_VISIBLE_DEVICES=0 python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl "${SELECTED_JSONL}" \
    --tar_output_pattern "${TOKEN_DIR}/audios/shard-%06d.tar" \
    --jsonl_output_pattern "${TOKEN_DIR}/txts/shard-%06d.jsonl" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --samples_per_shard 1 \
    --min_num_shards 4 \
    --nj_per_gpu 1 \
    --loader_workers 1 \
    --shuffle False \
    --shuffle-seed 42

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --gpu_ids 0 \
    --num_processes 1 \
    -m omnivoice.cli.train \
    --train_config "${TRAIN_CONFIG}" \
    --data_config "${DATA_CONFIG}" \
    --output_dir "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0 python -m omnivoice.cli.eval_memorization \
    --adapter-checkpoint "${OUTPUT_DIR}/checkpoint-2000" \
    --train-config "${TRAIN_CONFIG}" \
    --data-config "${DATA_CONFIG}" \
    --threshold 0.01 \
    --device cuda:0
