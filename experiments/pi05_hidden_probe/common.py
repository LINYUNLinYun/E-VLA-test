"""Configuration, provenance, safe persistence, and source-feature reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = {
    "checkpoint": str(ROOT / "shared/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero"),
    "config_name": "pi05_libero",
    "openpi_root": str(ROOT / "policies/openpi"),
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def config_hash(config):
    # Output location and manifest location do not change the experimental setup.
    return digest({k: v for k, v in config.items()
                   if k not in {"output_dir", "manifest", "cache_dir", "_config_path", "frozen_hash"}})


def load_config(path):
    import yaml
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    base = path.parent
    for key in ("manifest", "output_dir", "cache_dir"):
        if config.get(key):
            config[key] = str((base / config[key]).resolve())
    config["model"] = {**DEFAULT_MODEL, **config.get("model", {})}
    for name in ("model", "target_model"):
        if name not in config:
            continue
        for key in ("checkpoint", "openpi_root", "tokenizer_path", "hf_runtime"):
            if config[name].get(key):
                config[name][key] = str((base / config[name][key]).resolve())
    config.setdefault("seed", 0)
    config.setdefault("stage", "calibration")
    config.setdefault("controls", [{"id": "gray", "kind": "constant", "value": 127},
                                   {"id": "black", "kind": "constant", "value": 0}])
    config.setdefault("conditions", ["real", "nopatch", "self", "shuffle", "random", "direct", "layer0"])
    config.setdefault("max_new_tokens", 8)
    config.setdefault("score_candidates", True)
    config.setdefault("length_normalize", False)
    config.setdefault("aliases", {})
    if config["stage"] not in {"smoke", "calibration", "evaluation"}:
        raise ValueError("stage must be smoke, calibration, or evaluation")
    if not config.get("probes") or len({p["id"] for p in config["probes"]}) != len(config["probes"]):
        raise ValueError("Provide nonempty probes with unique id")
    if len({c["id"] for c in config["controls"]}) != len(config["controls"]):
        raise ValueError("Control IDs must be unique")
    if config["stage"] == "evaluation":
        if not config.get("frozen_hash") or config["frozen_hash"] != config_hash(config):
            raise ValueError("Evaluation requires an intact frozen config produced with --freeze-config")
        if any(len(p["target_layers"]) != 1 for p in config["probes"]):
            raise ValueError("Evaluation must use one frozen target layer per probe")
    config["_config_path"] = str(path)
    return config


def cli_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit-groups", type=int, help="Bound workload using whole independent groups")
    return parser


def select_rows(config, limit_groups=None):
    from .data import read_manifest
    rows = read_manifest(config["manifest"])
    wanted = "test" if config["stage"] == "evaluation" else "calibration"
    rows = [r for r in rows if r.get("split") == wanted]
    task_types = {p["task_type"] for p in config["probes"]}
    rows = [r for r in rows if r["task_type"] in task_types]
    if limit_groups is not None:
        if limit_groups < 1:
            raise ValueError("--limit-groups must be positive")
        groups = sorted({r["group_id"] for r in rows})[:limit_groups]
        rows = [r for r in rows if r["group_id"] in groups]
    if not rows:
        raise ValueError(f"No {wanted} rows match configured task_type values")
    return rows


def observation_hash(obs):
    h = hashlib.sha256()
    for key in sorted(obs):
        h.update(key.encode())
        value = obs[key]
        if isinstance(value, str):
            h.update(value.encode())
        else:
            array = np.ascontiguousarray(value)
            h.update(str(array.dtype).encode())
            h.update(str(array.shape).encode())
            h.update(array.tobytes())
    return h.hexdigest()


def control_observation(obs, spec):
    """A control never carries the source instruction, image, or proprioception."""
    if spec.get("kind") != "constant" or not 0 <= int(spec["value"]) <= 255:
        raise ValueError("First implementation supports fixed uint8 constant control images")
    value = int(spec["value"])
    return {"observation/image": np.full_like(obs["observation/image"], value),
            "observation/wrist_image": np.full_like(obs["observation/wrist_image"], value),
            "observation/state": np.zeros_like(obs["observation/state"]), "prompt": ""}


def target_prompt(row, probe):
    # A literal string or the manifest question only; no formatting with label/state/instruction.
    prompt = probe.get("prompt")
    return prompt if prompt is not None else row["question"]


def source_layers(config, backend):
    depth = int(backend.audit()["depth"])
    layers = list(range(depth + 1)) if config.get("source_layers") == "all" else config["source_layers"]
    if not layers or len(set(layers)) != len(layers) or any(not isinstance(l, int) or not 0 <= l <= depth for l in layers):
        raise ValueError(f"source_layers must be distinct integers in 0..{depth}, or all")
    return layers


def make_backends(config):
    from .backend import Backend
    target_cfg = config.get("target_model", config["model"])
    if target_cfg.get("backend") == "hf_paligemma":
        from .hf_backend import HFBackend
        target = HFBackend(**{k: v for k, v in target_cfg.items() if k != "backend"})
        source = Backend(**config["model"])
    else:
        source = Backend(**config["model"])
        target = source if target_cfg == config["model"] else Backend(**target_cfg)
    return source, target


def model_identity(config, backend):
    audit = backend.audit()
    # Cheap immutable-run identity; full array hashing on every sample is unnecessary.
    checkpoint = Path(config["checkpoint"])
    files = [(str(p.relative_to(checkpoint)), p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted(checkpoint.rglob("*")) if p.is_file()]
    code_dir = Path(config.get("openpi_root", DEFAULT_MODEL["openpi_root"])) / "src/openpi"
    code = {str(p.relative_to(code_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(code_dir.rglob("*.py"))}
    implementation = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(__file__).parent.glob("*.py")}
    return {"model": config, "checkpoint_files_digest": digest(files),
            "openpi_code_digest": digest(code), "probe_code_digest": digest(implementation),
            "audit": audit}


class FeatureCache:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def get(self, backend, obs, selector, layers, object_text=None):
        from .backend import select_positions
        spec = {"identity": self.identity, "observation": observation_hash(obs),
                "selector": selector, "object_text": object_text, "layers": list(layers)}
        key = digest(spec)
        path = self.directory / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"].item()))
                if meta["spec"] != spec:
                    raise ValueError(f"Feature cache identity mismatch: {path}")
                return {l: data[f"layer_{l}"].copy() for l in layers}, meta["positions"], key
        prefix = backend.prepare_source(obs)
        positions = select_positions(prefix, selector, object_text)
        result = backend.forward(prefix, capture_layers=layers, capture_positions=positions)
        meta = {"spec": spec, "positions": positions}
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with tmp.open("wb") as stream:
            np.savez(stream, metadata=np.array(json.dumps(meta)),
                     **{f"layer_{l}": np.asarray(result.hidden_states[l], np.float32) for l in layers})
        tmp.replace(path)
        return result.hidden_states, positions, key


def prepare_run(config, rows, source, target, *, command):
    from .data import load_observation
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    source_id = model_identity(config["model"], source)
    target_id = source_id if target is source else model_identity(config["target_model"], target)
    run = {"config_hash": config_hash(config), "source_identity": source_id,
           "target_identity": target_id, "sample_ids": sorted(r["sample_id"] for r in rows),
           "manifest_digest": digest(rows), "command": command,
           "source_input_digests": {r["sample_id"]: observation_hash(load_observation(r)) for r in rows}}
    return out, run


def check_resume(path, record):
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != record:
            raise ValueError(f"Run identity changed; choose a new output directory: {path}")
    else:
        write_json(path, record)
