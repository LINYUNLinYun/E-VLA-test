"""Fit the alignment by state MSE and a separate supervised information probe."""
import json
from pathlib import Path

import numpy as np

from experiments.pi05_hidden_probe.common import write_json
from experiments.pi05_hidden_probe.metrics import balanced_accuracy
from experiments.pi05_self_readout.run import sha
from .common import plan, source_states
from .linear import Affine, PairedStatistics, mean_squared_error
from .prepare import OUT


def save_map(path, mapping):
    with path.open("wb") as stream:
        np.savez(stream, weight=mapping.weight, bias=mapping.bias)


def fit_probe(config, out):
    samples = config["supervised_probe_samples"]
    pooled = [(r, source_states(r).mean(axis=0).astype(np.float64)) for r in samples]
    train = [(r, x) for r, x in pooled if r["partition"] == "train"]
    validation = [(r, x) for r, x in pooled if r["partition"] == "validation"]
    x = np.stack([v for _, v in train])
    y = np.array([1. if r["label"] == "blue" else -1. for r, _ in train])
    mx, my = x.mean(axis=0), y.mean()
    xc, yc = x - mx, y - my
    scale = float(np.square(xc).sum() / xc.shape[1])
    candidates, mappings = [], {}
    for alpha in config["ridge_candidates"]:
        # Dual ridge is cheap with 16 training images and 2048 pooled features.
        w = xc.T @ np.linalg.solve(xc @ xc.T + np.eye(len(xc)) * alpha * scale, yc)
        mapping = Affine(w.astype(np.float32), np.array(my - mx @ w, np.float32))
        scored = [{"labels": ["red", "blue"], "label": r["label"],
                   "prediction": "blue" if float(mapping(v)) >= 0 else "red"} for r, v in validation]
        candidates.append({"alpha": alpha, "validation_bacc": balanced_accuracy(scored)})
        mappings[alpha] = mapping
    selected = max(candidates, key=lambda r: (r["validation_bacc"], r["alpha"]))
    save_map(out / "linear_probe.npz", mappings[selected["alpha"]])
    return {"train_images": len(train), "validation_images": len(validation), "candidates": candidates,
            "selected": selected, "purpose": "supervised information-existence baseline only; not a language explanation"}


def main():
    config = plan()
    out = OUT / "fit"
    out.mkdir(exist_ok=True)
    if (out / "complete.json").exists():
        completed = json.loads((out / "complete.json").read_text())
        if completed["identity"] != config["identity"]:
            raise ValueError("Existing fit belongs to another plan")
        print("existing_fit_preserved", flush=True)
        return
    index = json.loads((OUT / "pairs/index.json").read_text())
    if index["identity"] != config["identity"] or len(index["items"]) != len(config["source_pairs"]):
        raise ValueError("Incomplete or mismatched paired states")
    batches = {"train": [], "validation": []}
    for item in index["items"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Paired-state file changed")
        with np.load(item["path"], allow_pickle=False) as data:
            batches[item["partition"]].append((data["x"].copy(), data["y"].copy()))
    stats = PairedStatistics(2048, 2048)
    for x, y in batches["train"]:
        stats.add(x, y)
    identity = Affine(np.eye(2048, dtype=np.float32), np.zeros(2048, np.float32))
    mean_scale = stats.mean_scale()
    baseline = {"identity": mean_squared_error(identity, batches["validation"]),
                "mean_scale": mean_squared_error(mean_scale, batches["validation"])}
    candidates, best, best_mse = [], None, float("inf")
    for alpha in config["ridge_candidates"]:
        mapping = stats.fit(alpha)
        error = mean_squared_error(mapping, batches["validation"])
        candidates.append({"alpha": alpha, "validation_state_mse": error})
        print("alignment_candidate", candidates[-1], flush=True)
        if error < best_mse:
            best_mse, best = error, mapping
    save_map(out / "affine.npz", best)
    save_map(out / "mean_scale.npz", mean_scale)
    probe = fit_probe(config, out)
    frozen = {"identity": config["identity"], "alignment_candidates": candidates,
              "alignment_selected": min(candidates, key=lambda r: r["validation_state_mse"]),
              "validation_state_baselines": baseline, "training_tokens": stats.n,
              "training_inputs": len(batches["train"]), "validation_inputs": len(batches["validation"]),
              "alignment_labels_used": False, "supervised_probe": probe,
              "model_sha256": {p.name: sha(p) for p in out.glob("*.npz")},
              "code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
              "test_used_for_fitting_or_selection": False}
    write_json(out / "frozen.json", frozen)
    write_json(out / "complete.json", {"complete": True, "identity": config["identity"], "frozen_sha256": sha(out / "frozen.json")})
    print("fit_complete", frozen["alignment_selected"], "probe", probe["selected"], flush=True)


if __name__ == "__main__":
    main()
