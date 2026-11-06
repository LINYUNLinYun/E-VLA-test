"""Compare production CUDA-graph π0 output with an independent full 10x18 reference."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment1.run_intermediate_action_probe import _probe_full_forward  # noqa: E402
from scripts.spec.spec_serve_policy import _build_policy_components  # noqa: E402
from scripts.spec.triton import pi0_infer  # noqa: E402
from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed  # noqa: E402


def _resolve_weights_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path / "base_weights.pkl" if path.is_dir() else path


def _resolve_prompt_embedding(draft_artifact: Path, prompt: str) -> torch.Tensor:
    manifest = json.loads((draft_artifact / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["prompts"].get(prompt)
    if entry is None:
        raise KeyError(f"prompt missing from {draft_artifact / 'manifest.json'}: {prompt!r}")
    return torch.load(draft_artifact / entry["embed_path"], map_location="cpu", weights_only=True).to(torch.bfloat16)


def _production_output(checkpoint: dict[str, Any], images, state, noise) -> tuple[Any, torch.Tensor]:
    runtime = pi0_infer.Pi0Inference(
        checkpoint,
        num_views=2,
        chunk_size=50,
        ablation_layer_pattern="full18",
        ablation_flow_pattern="full10",
    )
    output = runtime.forward(images, state, noise).clone()
    torch.cuda.synchronize()
    return runtime, output


def _reference_output(runtime, images, state, noise) -> tuple[torch.Tensor, list[int]]:
    output, _, layer_visits = _probe_full_forward(runtime, images, state, noise)
    torch.cuda.synchronize()
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    return output.cpu(), layer_visits


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full18/full10 GPU bitwise preflight")

    base_weights_path = _resolve_weights_path(args.base_artifact)
    base_root = base_weights_path.parent
    dataset = np.load(args.observations, allow_pickle=False)
    metadata = json.loads(str(dataset["metadata_json"]))
    sample_idx = next(
        (
            index
            for index, item in enumerate(metadata)
            if item.get("suite") == "libero_goal" and int(item.get("task_id", -1)) == 5
        ),
        None,
    )
    if sample_idx is None:
        raise ValueError("observation archive has no libero_goal task 5 sample")
    sample = metadata[sample_idx]
    prompt = str(sample["task_description"])
    with base_weights_path.open("rb") as handle:
        base_weights = pickle.load(handle)
    checkpoint = dict(base_weights)
    checkpoint["language_embeds"] = _resolve_prompt_embedding(args.draft_artifact, prompt)

    _, input_transform, _ = _build_policy_components(config_name="pi0_libero", checkpoint_dir=base_root)
    transformed = input_transform(
        {
            "observation/image": dataset["image"][sample_idx],
            "observation/wrist_image": dataset["wrist_image"][sample_idx],
            "observation/state": dataset["state"][sample_idx],
            "prompt": prompt,
        }
    )
    rng = np.random.default_rng(args.noise_seed)
    noise_np = rng.standard_normal((50, 32), dtype=np.float32)
    images, state, noise = prepare_triton_inputs_from_transformed(
        transformed=transformed,
        device="cuda",
        action_horizon=50,
        action_dim=32,
        noise=noise_np,
    )

    runtime, production_output = _production_output(checkpoint, images, state, noise)
    reference_output, layer_visits = _reference_output(runtime, images, state, noise)
    production_output = production_output.cpu()
    difference = (production_output.float() - reference_output.float()).abs()
    return {
        "check": "production_cuda_graph_vs_independent_full_10x18_forward",
        "task_suite_name": sample["suite"],
        "task_id": int(sample["task_id"]),
        "sample_id": sample.get("sample_id"),
        "noise_seed": int(args.noise_seed),
        "production_patterns": {"layer": "full18", "flow": "full10"},
        "reference": "independent original full forward: 10 denoising steps x 18 decoder layers per step",
        "reference_layer_visits_per_layer": layer_visits,
        "shape": list(production_output.shape),
        "bitwise_equal": bool(torch.equal(production_output, reference_output)),
        "max_abs_diff": float(difference.max()),
        "gpu": torch.cuda.get_device_name(0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--draft-artifact", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--noise-seed", type=int, default=20260928)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any]
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "check": "production_cuda_graph_vs_independent_full_10x18_forward",
            "bitwise_equal": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("bitwise_equal") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
