"""Small, dependency-light helpers shared by the profiling entry points."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


def _command(*args: str, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_state(repo: Path) -> dict[str, Any]:
    status = _command("git", "status", "--short", cwd=repo)
    return {
        "path": str(repo.resolve()),
        "commit": _command("git", "rev-parse", "HEAD", cwd=repo),
        "remote": _command("git", "remote", "get-url", "origin", cwd=repo),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def system_manifest() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": os.path.realpath(os.sys.executable),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": _command(
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        "nsys_version": _command("nsys", "--version"),
    }


def python_packages() -> dict[str, str]:
    """Return the resolved Python environment without invoking pip."""
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def timing_summary(samples_ms: list[float]) -> dict[str, Any]:
    if not samples_ms:
        return {"samples_ms": [], "count": 0}
    ordered = sorted(samples_ms)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "samples_ms": samples_ms,
        "count": len(samples_ms),
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "min_ms": ordered[0],
        "p50_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "max_ms": ordered[-1],
    }


class StageAccounting:
    """Accumulate per-NVTX-range wall-clock samples.

    NVTX ranges are only visible after the fact, in the nsys report. Recording the
    same boundaries in-process lets the manifest show whether a parent range and the
    sum of its children actually reconcile, which is the check that the earlier
    captures failed.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    def record(self, name: str, elapsed_ms: float) -> None:
        self._samples.setdefault(name, []).append(elapsed_ms)

    def reset(self) -> None:
        """Drop warmup samples so only the capture window is summarised."""
        self._samples.clear()

    def samples(self, name: str) -> list[float]:
        return list(self._samples.get(name, ()))

    def summary(self) -> dict[str, Any]:
        return {name: timing_summary(values) for name, values in sorted(self._samples.items())}

    def reconcile(self, parent: str, children: Sequence[str]) -> dict[str, Any]:
        """Compare one parent range against the sum of its declared children."""
        parent_samples = self._samples.get(parent, [])
        parent_total = sum(parent_samples)
        child_totals = {name: sum(self._samples.get(name, ())) for name in children}
        children_total = sum(child_totals.values())
        return {
            "parent": parent,
            "parent_total_ms": parent_total,
            "children": child_totals,
            "children_total_ms": children_total,
            "unaccounted_ms": parent_total - children_total,
            "unaccounted_fraction": (
                (parent_total - children_total) / parent_total if parent_total else None
            ),
        }


@contextlib.contextmanager
def _raw_nvtx_range(name: str) -> Iterator[None]:
    """Emit an NVTX range through nvtx, or through PyTorch as a fallback."""
    try:
        import nvtx  # type: ignore[import-not-found]
    except ImportError:
        nvtx = None

    if nvtx is not None:
        with nvtx.annotate(name):
            yield
        return

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "NVTX support is unavailable; install the 'nvtx' package in this environment"
        ) from exc

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def nvtx_range(
    name: str,
    *,
    accounting: StageAccounting | None = None,
    synchronize: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Emit an NVTX range, optionally with honest device-side boundaries.

    PyTorch and JAX both dispatch asynchronously, so a range that closes when the
    Python call returns measures *launch* time, not device execution time; a range
    can then contain another stage's kernels and report a cost that never happened.
    Passing `synchronize` drains the device before the range opens and again before
    it closes, so the reported interval covers exactly this stage's work.
    """
    if synchronize is not None:
        synchronize()
    started_ns = time.perf_counter_ns()
    with _raw_nvtx_range(name):
        try:
            yield
        finally:
            if synchronize is not None:
                synchronize()
            if accounting is not None:
                accounting.record(name, (time.perf_counter_ns() - started_ns) / 1_000_000)


def wrap_method(
    obj: Any,
    method_name: str,
    range_name: str,
    *,
    accounting: StageAccounting | None = None,
    synchronize: Callable[[], None] | None = None,
) -> Callable[..., Any]:
    """Wrap one bound method with an NVTX range and return the original method."""
    original = getattr(obj, method_name)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with nvtx_range(range_name, accounting=accounting, synchronize=synchronize):
            return original(*args, **kwargs)

    setattr(obj, method_name, wrapped)
    return original


def measure_call(call: Callable[[], Any], synchronize: Callable[[], None]) -> tuple[Any, float]:
    synchronize()
    started_ns = time.perf_counter_ns()
    result = call()
    synchronize()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    return result, elapsed_ms
