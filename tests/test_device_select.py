"""Unit tests for the shared torch device-selection helper.

These cover every branch of ``select_device`` (cuda > mps > cpu, plus the
defensive "torch has no backends attribute" path) using lightweight torch
stubs, so the helper keeps 100% line coverage without needing a real GPU/MPS
machine in CI.
"""
import types

from src.inference._device import select_device


def _fake_torch(*, cuda: bool, mps=None):
    """Build a minimal torch stand-in.

    ``cuda`` toggles ``torch.cuda.is_available()``. ``mps`` may be True/False
    to expose ``torch.backends.mps.is_available()``, or None to omit the
    ``backends`` attribute entirely (mimicking a torch build/stub without it).
    ``torch.device(name)`` simply echoes the device name back so assertions
    can compare against the string.
    """
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        device=lambda name: name,
    )
    if mps is not None:
        torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        )
    return torch


def test_select_device_prefers_cuda_when_available():
    """cuda wins over everything else when present."""
    assert select_device(_fake_torch(cuda=True, mps=True)) == "cuda"


def test_select_device_uses_mps_when_no_cuda():
    """With no cuda but mps available, pick mps (Apple Silicon path)."""
    assert select_device(_fake_torch(cuda=False, mps=True)) == "mps"


def test_select_device_falls_back_to_cpu_when_mps_unavailable():
    """No cuda and mps present-but-unavailable degrades to cpu."""
    assert select_device(_fake_torch(cuda=False, mps=False)) == "cpu"


def test_select_device_handles_torch_without_backends():
    """A torch lacking a `backends` attribute must not raise — it's cpu."""
    assert select_device(_fake_torch(cuda=False, mps=None)) == "cpu"
