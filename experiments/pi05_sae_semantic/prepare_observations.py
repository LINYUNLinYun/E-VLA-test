"""Freeze task splits and reuse complete original pi0.5 baseline observations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_goal", "libero_spatial", "libero_object")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1105)
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = {}
    for value in json.loads(Path(args.inventory).read_text()):
        match = re.search(r"/(libero_goal|libero_spatial|libero_object)/task(\d+)/init(\d+)/inputs/replan0000.npz$", value)
        if not match or "/fresh18/" not in value:
            continue
        suite, task, init = match[1], int(match[2]), int(match[3])
        if init not in (0, 1):
            continue
        directory = (ROOT / value).parent.parent
        candidates.setdefault((suite, task, init), []).append(directory)
    chosen = {}
    for key, directories in candidates.items():
        ordered = sorted(directories, key=lambda d: ("closedloop200_20261025" not in str(d), str(d)))
        for directory in ordered:
            result_path, identity_path = directory / "result.json", directory / "identity.json"
            if not (directory / "complete.json").exists() or not result_path.exists() or not identity_path.exists():
                continue
            result, identity = json.loads(result_path.read_text()), json.loads(identity_path.read_text())
            if result.get("model") != "pi05" or not result.get("valid") or identity.get("skip"):
                continue
            chosen[key] = (directory, result, identity)
            break
    available = {(suite, task) for suite, task, init in chosen
                 if (suite, task, 0) in chosen and (suite, task, 1) in chosen}
    # Calibration is selected by tasks, never by splitting adjacent frames.
    calibration = [("libero_goal", 0), ("libero_goal", 1), ("libero_spatial", 0), ("libero_object", 0)]
    if not set(calibration) <= available:
        raise ValueError("Expected calibration source tasks missing; inspect inventory rather than silently replacing them")
    rng = random.Random(args.seed)
    discovery, test = [], []
    for suite in SUITES:
        tasks = sorted(t for s, t in available if s == suite and (s, t) not in calibration)
        rng.shuffle(tasks)
        discovery.extend((suite, t) for t in tasks[:4])
        needed = 2 if suite == "libero_goal" else 3
        test.extend((suite, t) for t in tasks[4:4+needed])
    if (len(discovery), len(calibration), len(test)) != (12, 4, 8):
        raise ValueError("Insufficient sources for the planned 12/4/8 task split")
    assignments = {key: split for split, keys in [("discovery", discovery), ("calibration", calibration), ("heldout", test)] for key in keys}
    tasks, episodes, observations = [], [], []
    for (suite, task), split in assignments.items():
        directory, result, identity = chosen[(suite, task, 0)]
        instruction = identity["instruction"]
        tasks.append(dict(suite=suite, task=task, split=split, instruction=instruction))
        if split == "heldout":
            continue
        for init in (0, 1):
            directory, result, identity = chosen[(suite, task, init)]
            if identity["instruction"] != instruction:
                raise ValueError("Task instruction differs between init sources")
            files = sorted((directory / "inputs").glob("replan*.npz"))
            indexes = np.unique(np.linspace(0, len(files)-1, min(24, len(files))).astype(int))
            pilot_indexes = set(indexes[np.unique(np.linspace(0, len(indexes)-1, min(2, len(indexes))).astype(int))])
            episodes.append(dict(suite=suite, task=task, init=init, split=split, source=str(directory),
                                 source_model=result["model"], source_kind=result.get("kind"),
                                 H=result.get("H"), R=result.get("R"), source_success=result.get("success"),
                                 source_termination=result.get("termination"), available_requests=len(files), selected=len(indexes)))
            for index in indexes:
                path = files[index]
                observations.append(dict(id=f"{suite}_task{task:02d}_init{init:02d}_{path.stem}", split=split,
                                         suite=suite, task=task, init=init, request=int(path.stem.removeprefix("replan")),
                                         instruction=instruction, observation_path=str(path),
                                         pilot=bool(index in pilot_indexes), source_episode=str(directory)))
    # Hash and check only pilot inputs now; full-source pointers are rechecked when consumed.
    for row in observations:
        if row["pilot"]:
            path = Path(row["observation_path"])
            row["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["instruction"].item()) != row["instruction"]:
                    raise ValueError(f"Saved input instruction mismatch: {path}")
    for name, values in [("observations.jsonl", observations), ("pilot_observations.jsonl", [r for r in observations if r["pilot"]])]:
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in values))
    manifest = dict(seed=args.seed, task_split_method="Fixed four calibration tasks; seeded suite-balanced discovery and heldout selection; no SR selection",
                    observation_source="Existing valid full pi05 fresh18 baseline episodes; hidden states will be recomputed",
                    tasks=tasks, episodes=episodes, n_observations=len(observations),
                    n_pilot_observations=sum(r["pilot"] for r in observations),
                    splits=dict(Counter(r["split"] for r in tasks)),
                    episode_sampling="Up to 24 unique uniform request indices per complete episode; short episodes are not padded",
                    caveat="Baseline backend/protocol is recorded, not claimed identical to current PyTorch execution; raw observations only are reused")
    (out / "observation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ["n_observations", "n_pilot_observations", "splits"]}), flush=True)


if __name__ == "__main__":
    main()
