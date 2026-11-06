#!/usr/bin/env python3
"""Run Full18 and all single-layer VLM interventions on exported observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/spec"))
import spec_serve_policy  # noqa: E402
from scripts.spec.triton import triton_pi0_runtime  # noqa: E402

from layer_probe import collect_single_layer_probe  # noqa: E402


def _load_observation(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "agent_image": data["agent_image"],
            "wrist_image": data["wrist_image"],
            "robot_state": data["robot_state"],
            "prompt": str(data["prompt"].item()),
            "metadata": json.loads(str(data["metadata_json"].item())),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--base-triton-path", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="pi0_libero")
    parser.add_argument("--denoise-steps", type=int, choices=(3, 5, 7, 10), default=10)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    base_weights, artifact_dir = spec_serve_policy._resolve_triton_base_artifact(args.base_triton_path)
    train_config, input_transform, _ = spec_serve_policy._build_policy_components(
        config_name=args.config,
        checkpoint_dir=artifact_dir,
    )
    pool = triton_pi0_runtime.TritonRuntimePool(
        base_weights_path=base_weights,
        manifest_path=args.prompt_manifest.resolve(),
        num_views=2,
        chunk_size=int(train_config.model.action_horizon),
        vlm_layer_pattern="vlm_full18",
        ablation_layer_pattern="full18",
        denoise_steps=args.denoise_steps,
    )
    outputs = []
    for input_index, path in enumerate(args.inputs):
        observation = _load_observation(path.resolve())
        prompt = str(observation["prompt"])
        transformed = input_transform(
            {
                "observation/image": observation["agent_image"],
                "observation/wrist_image": observation["wrist_image"],
                "observation/state": observation["robot_state"],
                "prompt": prompt,
            }
        )
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, input_index]))
        noise = rng.standard_normal(
            (int(train_config.model.action_horizon), int(train_config.model.action_dim)), dtype=np.float32
        )
        images, state, noise_tensor = triton_pi0_runtime.prepare_triton_inputs_from_transformed(
            transformed=transformed,
            device="cuda",
            action_horizon=int(train_config.model.action_horizon),
            action_dim=int(train_config.model.action_dim),
            noise=noise,
        )
        session = pool.start_session(prompt)
        prepared = session.prepare_observation(images=images, state=state)
        output = args.output_dir.resolve() / f"{path.stem}.layer_probe.npz"
        collect_single_layer_probe(
            session=session,
            prepared=prepared,
            noise=noise_tensor,
            output_path=output,
            sample_id=str(observation["metadata"].get("sample_id", path.stem)),
            view_names=("agent_view", "wrist_view"),
            metadata={**observation["metadata"], "noise_seed": args.seed, "denoise_steps": args.denoise_steps},
        )
        outputs.append(str(output))
        print(f"[{input_index + 1}/{len(args.inputs)}] {output}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    previous: list[str] = []
    if manifest_path.is_file():
        try:
            previous = [str(value) for value in json.loads(manifest_path.read_text(encoding="utf-8")).get("probes", [])]
        except (json.JSONDecodeError, OSError, TypeError):
            previous = []
    merged = list(dict.fromkeys(previous + outputs))
    report = {"probe_count": len(merged), "probes": merged}
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
