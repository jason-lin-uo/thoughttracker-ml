"""Shared torch device selection.

Keeping the cuda/mps/cpu preference in ONE place means the stance loader,
the topic-relevance loader, and the offline evaluation script all agree on
which accelerator to use — a single, independently-tested code path instead
of three copies of the same ternary drifting apart over time.
"""
from __future__ import annotations

from typing import Any


def select_device(torch: Any) -> Any:
    """Return the best available ``torch.device``: cuda > mps > cpu.

    The preference order mirrors where this code actually runs: ``cuda`` on
    the GPU training boxes, ``mps`` on Apple Silicon (e.g. running the demo
    or an interview locally on a MacBook), and ``cpu`` everywhere else.

    ``torch`` is passed in rather than imported here so callers that import
    torch lazily keep that cost local, and so tests can inject a stub. The
    capability probes are deliberately defensive: a partially-populated torch
    — a stub without a ``backends`` attribute, or a torch built without MPS
    support — is treated as "that accelerator is unavailable" and selection
    degrades to ``cpu`` rather than raising ``AttributeError``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # getattr chain tolerates a torch that lacks `backends` (older builds or
    # test doubles) — those resolve to None and fall through to cpu.
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")
