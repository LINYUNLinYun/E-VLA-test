"""Frozen scientific conditions shared by preparation, service, and worker."""
import hashlib
from pathlib import Path

from preparations.idea1_v3_1.common import ROOT, SUITES, atomic_json, digest, read_manifest

MODULE = "preparations.ease_insights_20261103.vlm"
MODEL_PYTHON = ROOT / "policies/realtime-vla-flash/.venv/bin/python"
ENV_PYTHON = ROOT / "policies/libero-eval-v21/.venv/bin/python"
MASKS = {
    "fixed_v12_jacobian": (2, 3, 5, 14, 16, 17),
    "fixed_v12_action_mse": (2, 3, 5, 11, 16, 17),
    "fixed_v12_cosine": (2, 3, 5, 6, 8, 9),
}
CONDITIONS = ("full_d10", *MASKS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions():
    return [dict(id=cid, model="pi0", mode="full" if cid == "full_d10" else "periodic",
                 compensation=None if cid == "full_d10" else "extrapolate", period=2,
                 H=50, R=5, D=10 if cid == "full_d10" else 2, refresh_D=10,
                 skip=list(MASKS.get(cid, ()))) for cid in CONDITIONS]


def validate_condition(config):
    expected = next((c for c in conditions() if c["id"] == config.get("id")), None)
    if expected is None or config != expected:
        raise ValueError("Condition differs from fixed VLM experiment protocol")
    return tuple(config["skip"])


def cases(manifest, stage, shard_index=0, shards=1):
    if not 0 <= shard_index < shards:
        raise ValueError("Invalid shard")
    if stage != "discovery":
        raise ValueError("Independent init31-35 validation is locked until a pair and directions are frozen")
    pairs = [(s, t) for s in manifest["suites"] for t in manifest["tasks"]]
    return [dict(suite=s, task=t, init=i) for index, (s, t) in enumerate(pairs)
            if index % shards == shard_index for i in manifest["discovery_inits"]]
