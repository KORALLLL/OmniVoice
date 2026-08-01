import os
import shlex
import subprocess
from pathlib import Path

from omnivoice.training.config import TrainingConfig
from omnivoice.training.lora import DEFAULT_LORA_TARGET_MODULES


ROOT = Path(__file__).resolve().parents[1]


def test_lora_example_config_loads_broad_adapter_training_defaults():
    config = TrainingConfig.from_json(
        ROOT / "examples/config/train_config_finetune_lora.json"
    )

    assert config.lora_enabled is True
    assert config.lora_rank == 32
    assert config.lora_alpha == 64
    assert config.lora_dropout == 0.05
    assert config.lora_bias == "none"
    assert config.lora_target_modules == list(DEFAULT_LORA_TARGET_MODULES)
    assert config.init_from_checkpoint == "k2-fsa/OmniVoice"
    assert config.learning_rate == 0.0001
    assert config.mixed_precision == "bf16"
    assert config.attn_implementation == "flex_attention"


def test_lora_example_launcher_runs_default_eight_gpu_training(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "commands.log"

    for command in ("python", "accelerate"):
        stub = bin_dir / command
        stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n')
        stub.chmod(0o755)

    environment = os.environ | {
        "COMMAND_LOG": str(command_log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "examples/run_finetune_lora.sh")],
        cwd=ROOT / "examples",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in command_log.read_text().splitlines()]
    assert commands[-1] == [
        "launch",
        "--gpu_ids",
        "0,1,2,3,4,5,6,7",
        "--num_processes",
        "8",
        "-m",
        "omnivoice.cli.train",
        "--train_config",
        "config/train_config_finetune_lora.json",
        "--data_config",
        "config/data_config_finetune.json",
        "--output_dir",
        "exp/omnivoice_finetune_lora",
    ]
