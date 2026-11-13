# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional MXFP8 dtypes must not block ordinary BF16 training imports."""

import importlib.util
import subprocess
import sys

import pytest
import torch

from megatron.core.inference.quantization import mxfp8_tensor


@pytest.fixture
def without_e8m0(monkeypatch):
    monkeypatch.delattr(torch, "float8_e8m0fnu", raising=False)
    name = "megatron.core.inference.quantization._mxfp8_import_test"
    spec = importlib.util.spec_from_file_location(name, mxfp8_tensor.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_bf16_training_modules_import_without_optional_e8m0_dtype():
    # A fresh interpreter catches eager imports that an already-loaded test module hides.
    script = """
import torch
torch.compile(lambda x: x)  # Initialize this Torch version's own compiler dtype tables first.
if hasattr(torch, 'float8_e8m0fnu'):
    del torch.float8_e8m0fnu
from megatron.core.transformer.moe.experts import SequentialMLP
from experiments.xcx.models.gpt import XCXModelConfig
import pretrain_gpt
assert not hasattr(torch, 'float8_e8m0fnu')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_flashinfer_byte_scale_validation_does_not_require_e8m0(without_e8m0):
    tensor = without_e8m0.MXFP8Tensor(
        data=torch.empty(1, 32, dtype=torch.float8_e4m3fn),
        scale=torch.zeros(1, 1, dtype=torch.uint8),
        backend="flashinfer",
    )
    without_e8m0.validate_mxfp8_tensor(tensor)


@pytest.mark.parametrize("operation", ["convert", "validate", "quantize", "update"])
def test_triton_mxfp8_reports_missing_dtype_before_quantization(without_e8m0, operation):
    tensor = without_e8m0.MXFP8Tensor(
        data=torch.empty(1, 32, dtype=torch.float8_e4m3fn),
        scale=torch.zeros(512, dtype=torch.uint8),
        backend="triton",
    )
    source = torch.zeros(1, 32, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=r"MXFP8.*torch.float8_e8m0fnu"):
        if operation == "convert":
            without_e8m0.ensure_mxfp8_scale_dtype(tensor.scale)
        elif operation == "validate":
            without_e8m0.validate_mxfp8_tensor(tensor)
        elif operation == "quantize":
            without_e8m0.MXFP8Tensor.from_bf16(source, backend="triton")
        else:
            tensor.quantize_(source)


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="E8M0 dtype unavailable")
def test_supported_e8m0_conversion_preserves_storage_and_values():
    raw = torch.tensor([0, 1, 126, 127, 128, 254, 255], dtype=torch.uint8)
    converted = mxfp8_tensor.ensure_mxfp8_scale_dtype(raw)
    assert converted.dtype == torch.float8_e8m0fnu
    assert converted.data_ptr() == raw.data_ptr()
    assert torch.equal(converted.view(torch.uint8), raw)
    assert mxfp8_tensor.ensure_mxfp8_scale_dtype(converted) is converted
