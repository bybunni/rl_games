"""Integration test for shared PPO-LSTM ONNX export parity on CartPole masked velocity."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_lstm_onnx.py"
CHECKPOINT_DIR = ROOT / "runs" / "lstm_variants" / "cartpole_lstm_after_mlp" / "nn"


def _has_module(name):
    return importlib.util.find_spec(name) is not None


def _find_checkpoint():
    if not CHECKPOINT_DIR.is_dir():
        return None
    ckpts = sorted(CHECKPOINT_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def test_export_lstm_cartpole_variant_to_onnx_and_validate(tmp_path):
    if not _has_module("onnx"):
        pytest.skip("onnx is not installed in this environment")
    if not _has_module("onnxruntime"):
        pytest.skip("onnxruntime is not installed in this environment")

    checkpoint = _find_checkpoint()
    if checkpoint is None:
        pytest.skip(f"checkpoint not found under: {CHECKPOINT_DIR}")

    onnx_path = tmp_path / "cartpole_lstm_after_mlp.onnx"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--onnx-out",
        str(onnx_path),
        "--variant",
        "after_mlp",
        "--steps",
        "16",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--tolerance",
        "1e-4",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(
            "ONNX export/validation script failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    assert onnx_path.is_file(), f"Expected ONNX file to exist: {onnx_path}"

