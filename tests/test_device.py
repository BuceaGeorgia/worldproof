"""Tests for device autodetection (CLAUDE.md hardware rules)."""

from __future__ import annotations

import pytest
import torch

from worldproof.device import available_devices, get_device


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "mps", "cuda")


def test_autodetect_is_an_available_device():
    assert get_device().type in available_devices()


def test_cpu_is_always_available():
    assert "cpu" in available_devices()
    assert get_device("cpu").type == "cpu"


def test_prefer_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown device"):
        get_device("tpu")


def test_prefer_unavailable_backend_raises():
    # This machine (Apple Silicon) has no CUDA; the request must fail loudly
    # rather than silently falling back.
    if "cuda" not in available_devices():
        with pytest.raises(RuntimeError, match="not available"):
            get_device("cuda")
