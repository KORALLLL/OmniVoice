#!/bin/bash

# Alternate eight-GPU LoRA training with checkpoint-isolated validation.

set -euo pipefail

if [ -z "${SELECTED_MANIFEST:-}" ]; then
    echo "SELECTED_MANIFEST must name the deterministic 24-clip manifest" >&2
    exit 2
fi
if [ -z "${DEADLINE_STATE:-}" ]; then
    echo "DEADLINE_STATE must name the memorization result.json" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

resolve_config() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "${SCRIPT_DIR}" "$1" ;;
    esac
}

TRAIN_CONFIG="$(resolve_config "${TRAIN_CONFIG:-config/train_config_finetune_lora.json}")"
DATA_CONFIG="$(resolve_config "${DATA_CONFIG:-config/data_config_finetune.json}")"
VALIDATION_CONFIG="$(resolve_config "${VALIDATION_CONFIG:-config/hard_number_validation.json}")"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/exp/omnivoice_finetune_lora}"
VALIDATION_OUTPUT_ROOT="${VALIDATION_OUTPUT_ROOT:-${REPO_ROOT}/exp/omnivoice_validation}"

python -m omnivoice.cli.run_lora_validation \
    --train-config "${TRAIN_CONFIG}" \
    --data-config "${DATA_CONFIG}" \
    --validation-config "${VALIDATION_CONFIG}" \
    --selected-manifest "${SELECTED_MANIFEST}" \
    --deadline-state "${DEADLINE_STATE}" \
    --output-dir "${OUTPUT_DIR}" \
    --validation-output-root "${VALIDATION_OUTPUT_ROOT}"
