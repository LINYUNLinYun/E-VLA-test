import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIMITS = dict(zip(SUITES, (220, 280, 300, 520)))
PYTHONS = {
    "pi0": ROOT/"policies/openpi/.venv/bin/python",
    "pi05": ROOT/"policies/openpi/.venv/bin/python",
    "xiaomi-robotic0": ROOT/"policies/xiaomi-robotic0/.venv/bin/python",
    "db-cogact": ROOT/"policies/db-cogact/.venv/bin/python",
    "environment": ROOT/"policies/libero-eval-v21/.venv/bin/python",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def seed(suite, task, init, request):
    return int(digest(["idea1-v3.1", 1701, suite, task, init, request])[:8], 16)


def environment(gpu=None):
    env = dict(os.environ)
    paths = [str(ROOT), str(ROOT/"policies/openpi/src"), str(ROOT/"policies/openpi/packages/openpi-client/src"), str(ROOT/"LIBERO")]
    env.update(PYTHONPATH=":".join(paths+[env.get("PYTHONPATH", "")]), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               LIBERO_CONFIG_PATH=str(ROOT/"policies/realtime-vla-flash/.libero"))
    if gpu is not None:
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
    return env


def read_manifest(path, verify_sources=False):
    manifest = json.loads(Path(path).read_text())
    identity = manifest.pop("manifest_hash")
    if digest(manifest) != identity:
        raise ValueError("Frozen manifest hash mismatch")
    manifest["manifest_hash"] = identity
    if verify_sources:
        for filename,expected in manifest["source_files"].items():
            if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=expected:
                raise ValueError(f"Frozen implementation source changed: {filename}")
        for checkpoint in manifest["checkpoints"].values():
            for filename,expected in checkpoint["identity_artifacts"].items():
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
                    raise ValueError(f"Frozen checkpoint identity artifact changed: {filename}")
    return manifest
