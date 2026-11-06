"""Experiment interface for the in-model deterministic A/B SigLIP compression.

The implementation lives next to the JAX model
(``openpi/models/vision_compression.py``) and is wired into
``Encoder1DBlock.__call__``; it is inert unless ``configure`` is called before
model construction.  This module only exposes the static switch and the
mapping/merge primitives for tests and reporting.
"""

from __future__ import annotations

from openpi.models import vision_compression as _impl

N_TOKENS = _impl.N_TOKENS
N_DST = _impl.N_DST
N_SRC = _impl.N_SRC
DST_INDEX = _impl.DST_INDEX
SRC_INDEX = _impl.SRC_INDEX
CONFIG = _impl.CONFIG

make_mapping = _impl.make_mapping
merge = _impl.merge
unmerge = _impl.unmerge


def install(mode: str = "off", keep_tokens: int = N_TOKENS) -> None:
    """Set the static compression configuration before model creation."""
    _impl.configure(mode, keep_tokens)


def configure(mode: str, keep_tokens: int) -> None:
    _impl.configure(mode, keep_tokens)


def restore() -> None:
    _impl.configure("off", N_TOKENS)


def is_active() -> bool:
    return _impl.is_active()
