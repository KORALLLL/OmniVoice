#!/bin/bash

# Run the bounded four-utterance memorization orchestrator on one GPU.

set -euo pipefail

if [ -z "${SELECTED_MANIFEST:-}" ]; then
    echo "SELECTED_MANIFEST must name the deterministic selected manifest" >&2
    exit 2
fi
if [ ! -f "${SELECTED_MANIFEST}" ]; then
    echo "SELECTED_MANIFEST is not a regular file: ${SELECTED_MANIFEST}" >&2
    exit 2
fi
SELECTED_MANIFEST="$(cd "$(dirname "${SELECTED_MANIFEST}")" && pwd)/$(basename "${SELECTED_MANIFEST}")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

TRAIN_CONFIG="${TRAIN_CONFIG:-${SCRIPT_DIR}/config/train_config_lora_memorization.json}"
DATA_CONFIG="${DATA_CONFIG:-${SCRIPT_DIR}/config/data_config_lora_memorization.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/exp/omnivoice_lora_memorization}"

CUDA_VISIBLE_DEVICES=0 python -m omnivoice.cli.run_memorization \
    --selected-manifest "${SELECTED_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --train-config "${TRAIN_CONFIG}" \
    --data-config "${DATA_CONFIG}" \
    --max-wall-clock-seconds 1200 \
    --experiment-wall-clock-seconds 3600
