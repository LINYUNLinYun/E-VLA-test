"""Small, opt-in NVTX helpers for CUDA profiling.

Set ``OPENPI_NVTX=1`` when launching the profiled process. Keeping the ranges
opt-in avoids perturbing normal latency measurements with host-side markers.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import torch


_T = TypeVar("_T")


def _enabled() -> bool:
    return os.environ.get("OPENPI_NVTX", "0").strip().lower() not in {"", "0", "false", "no", "off"}


@contextlib.contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    """Emit an NVTX range when profiling is enabled and CUDA is available."""
    active = bool(_enabled() and torch.cuda.is_available())
    if active:
        torch.cuda.nvtx.range_push(str(message))
    try:
        yield
    finally:
        if active:
            torch.cuda.nvtx.range_pop()


def nvtx_call(message: str, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Call ``fn`` inside an opt-in NVTX range."""
    with nvtx_range(message):
        return fn(*args, **kwargs)
