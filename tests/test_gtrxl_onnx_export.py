"""Integration test for GTrXL ONNX export parity on LunarLander."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_gtrxl_onnx.py"
CHECKPOINT = (
    ROOT
    / "runs"
    / "lunarlander_gtrxl_18-02-30-08"
    / "nn"
    / "lunarlander_gtrxl_ep_306_rew_200.92949.pth"
)


def _has_module(name):
    return importlib.util.find_spec(name) is not None


def test_export_lunarlander_gtrxl_to_onnx_and_validate(tmp_path):
    if not _has_module("onnx"):
        pytest.skip("onnx is not installed in this environment")
    if not _has_module("onnxruntime"):
        pytest.skip("onnxruntime is not installed in this environment")
    if not CHECKPOINT.is_file():
        pytest.skip(f"checkpoint not found: {CHECKPOINT}")

    onnx_path = tmp_path / "lunarlander_gtrxl.onnx"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--checkpoint",
        str(CHECKPOINT),
        "--onnx-out",
        str(onnx_path),
        "--steps",
        "64",
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
