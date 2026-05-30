"""Tests for the training script CLI."""

import subprocess
import sys

import pytest


def test_train_script_burgers_smoke():
    """Runs the training script for 2 epochs as a smoke test."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--problem", "burgers",
            "--n_samples", "20",
            "--n_x", "32",
            "--modes", "4",
            "--width", "8",
            "--n_layers", "2",
            "--n_epochs", "2",
            "--batch_size", "4",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_train_script_navier_stokes_smoke():
    """Runs the Navier-Stokes training script for 2 epochs as a smoke test."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--problem", "navier_stokes",
            "--n_samples", "8",
            "--h", "16",
            "--w", "16",
            "--T_in", "4",
            "--modes", "4",
            "--width", "4",
            "--n_layers", "2",
            "--n_epochs", "2",
            "--batch_size", "2",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_train_script_config_file(tmp_path):
    """Runs the training script using a YAML config file."""
    import yaml

    config = {
        "problem": "burgers",
        "n_samples": 20,
        "n_x": 32,
        "modes": 4,
        "width": 8,
        "n_layers": 2,
        "n_epochs": 2,
        "batch_size": 4,
    }
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(config))

    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_file)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
