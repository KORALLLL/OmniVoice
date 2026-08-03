from pathlib import Path

import tomllib


def test_validation_extra_contains_runtime_dependencies():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    extra = project["project"]["optional-dependencies"]["validation"]
    assert any(item.startswith("wandb") for item in extra)
    assert "onnx-asr==0.12.0" in extra
    assert any(item.startswith("onnxruntime-gpu") for item in extra)


def test_default_dev_group_retains_static_gate_tools():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    dev = project["dependency-groups"]["dev"]
    assert any(item.startswith("pytest") for item in dev)
    assert any(item.startswith("ruff") for item in dev)
