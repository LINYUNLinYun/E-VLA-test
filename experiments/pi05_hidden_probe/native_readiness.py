"""Bounded engineering/text diagnostic on one saved real LIBERO observation."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .backend import Backend, Patch, select_positions
from .common import DEFAULT_MODEL, append_jsonl, control_observation, write_json
from .workflow import difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-observation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL["checkpoint"])
    parser.add_argument("--model-config", default="pi05_libero")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    from preparations.idea1_v2_1.runtime_models import native_observation
    with np.load(args.raw_observation, allow_pickle=False) as archive:
        raw = {k: archive[k].copy() for k in archive.files if k not in {"instruction", "seed"}}
        instruction = str(archive["instruction"].item())
    images, state = native_observation(raw)
    obs = {"observation/image": np.asarray(images[0]), "observation/wrist_image": np.asarray(images[1]),
           "observation/state": state, "prompt": instruction}
    from PIL import Image
    Image.fromarray(obs["observation/image"]).save(out / "agent_view.png")
    Image.fromarray(obs["observation/wrist_image"]).save(out / "wrist_view.png")
    begin = time.monotonic()
    backend = Backend(**{**DEFAULT_MODEL, "checkpoint": args.checkpoint, "config_name": args.model_config})
    print(f"loaded in {time.monotonic()-begin:.2f}s", flush=True)
    source = backend.prepare_source(obs)
    write_json(out / "model_audit.json", backend.audit(source))
    checks = {}
    for name, prefix in [("source", source), ("target", backend.prepare_target(obs, "answer en What color is the bowl?\n"))]:
        reference = backend.reference_forward(prefix)
        repeated = backend.reference_forward(prefix)
        actual = backend.forward(prefix, capture_layers=[0, 6, 18])
        positions = select_positions(prefix, "images")
        same = backend.forward(prefix, patch=Patch(6, positions, actual.hidden_states[6][positions]))
        checks[name] = {"repeat": difference(repeated.normalized, reference.normalized, 1e-5, 1e-5),
                        "normalized": difference(actual.normalized, reference.normalized, 1e-5, 1e-5),
                        "logits": difference(actual.logits, reference.logits, 1e-5, 1e-5),
                        "self_patch": difference(same.normalized, actual.normalized, 1e-5, 1e-5)}
        write_json(out / "engineering_checks.json", checks)
        print(name, checks[name], flush=True)
    for prompt in ["caption en\n", "answer en What color is the bowl?\n", "answer en What color is the plate?\n"]:
        for condition in ["real", "gray"]:
            input_obs = obs if condition == "real" else control_observation(obs, {"kind": "constant", "value": 127})
            prefix = backend.prepare_target(input_obs, prompt)
            start = time.monotonic()
            generation = backend.generate(prefix, max_new_tokens=16)
            scores = backend.score_answers(prefix, ["black", "white", "red", "blue"])
            record = {"prompt": prompt, "condition": condition, **generation, "candidate_scores": scores,
                      "seconds": time.monotonic()-start, "scope": "readiness diagnostic; no accuracy claim"}
            append_jsonl(out / "generations.jsonl", record)
            print(record, flush=True)
    write_json(out / "complete.json", {"completed": True, "seconds": time.monotonic()-begin,
        "raw_observation": str(args.raw_observation.resolve()), "instruction": instruction})


if __name__ == "__main__":
    main()
