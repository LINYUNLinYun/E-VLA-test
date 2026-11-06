"""Explicit, bounded admission for future batch experiments; no import-time GPU work."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid as uuid_module


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEASES = ROOT / "results/idea1_v3_3/external_gpu_leases.json"
OWNER = "user-authorized idea2 fixed-batch pipeline experiment"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _gpu_index(gpu_uuid):
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("An explicit physical GPU UUID is required")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    matches = [row.split(",")[0].strip() for row in result.stdout.splitlines()
               if len(row.split(",")) == 2 and row.split(",")[1].strip() == gpu_uuid]
    if len(matches) != 1:
        raise RuntimeError("GPU UUID does not uniquely identify a physical GPU")
    return matches[0]


@contextmanager
def exclusive_gpu(gpu_uuid, output, *, leases_path=DEFAULT_LEASES):
    """Claim a shared lease and require an actually idle GPU, or fail immediately.

    Never waits for a GPU, starts a watcher, or stops another process. A lease
    is coordination with cooperating sessions, not hardware enforcement.
    Runtime callers must still check process occupancy between measured groups.
    """
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        raise RuntimeError("GPU admission must precede CUDA initialization")
    index = _gpu_index(gpu_uuid)
    from .layer_bench import require_idle_gpu

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = Path(leases_path)
    # Refuse an uncoordinated run if the shared registry is unavailable.
    if not path.is_file():
        raise RuntimeError(f"Shared GPU lease registry is missing: {path}")
    token = uuid_module.uuid4().hex
    receipt = dict(status="claiming", gpu_uuid=gpu_uuid, gpu_index=int(index),
                   token=token, owner=OWNER, owner_pid=os.getpid(), start_utc=_now())
    claimed = False
    try:
        with path.open("r+") as registry:
            fcntl.flock(registry, fcntl.LOCK_EX)
            values = json.load(registry)
            current = values.get(index, {})
            if current.get("active"):
                raise RuntimeError("Selected GPU already has an active lease: " + json.dumps(current))
            snapshot = require_idle_gpu(gpu_uuid)
            values[index] = dict(gpu=int(index), uuid=gpu_uuid, owner=OWNER, active=True,
                                 owner_pid=os.getpid(), token=token, time_utc=_now(),
                                 purpose="Fixed-batch D10 pipeline comparison", snapshot=snapshot)
            registry.seek(0)
            json.dump(values, registry, indent=2)
            registry.write("\n")
            registry.truncate()
            claimed = True
        # Catch an external process appearing after the atomic lease claim.
        snapshot = require_idle_gpu(gpu_uuid)
        receipt.update(status="active", gpu_before=snapshot)
        _save(output / "gpu_lease.json", receipt)
        yield receipt
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if claimed:
            with path.open("r+") as registry:
                fcntl.flock(registry, fcntl.LOCK_EX)
                values = json.load(registry)
                current = values.get(index, {})
                if current.get("token") != token or current.get("owner_pid") != os.getpid():
                    receipt.update(status="release_conflict", end_utc=_now(),
                                   release_error="Lease ownership changed; foreign entry left untouched")
                    _save(output / "gpu_lease.json", receipt)
                    raise RuntimeError(receipt["release_error"])
                current.update(active=False, released_utc=_now())
                values[index] = current
                registry.seek(0)
                json.dump(values, registry, indent=2)
                registry.write("\n")
                registry.truncate()
            if receipt["status"] == "active":
                receipt["status"] = "released"
            receipt.update(lease_active=False, end_utc=_now())
        _save(output / "gpu_lease.json", receipt)
