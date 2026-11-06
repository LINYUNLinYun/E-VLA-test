"""Read a native preflight capture with the unchanged released SAE loader."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .check_public_sae import load_official_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    base, assets, out = Path(args.preflight).resolve(), Path(args.assets).resolve(), Path(args.output).resolve()
    preflight = json.loads((base / "native_preflight.json").read_text())
    if not preflight["noop_equal"] or not preflight["tail_equal"]:
        raise ValueError("Native path check must pass first")
    captures = torch.load(base / "native_preflight_tensors.pt", map_location="cpu", weights_only=True)
    loader = load_official_loader(assets)
    index = json.loads((assets / "concepts/concept_features_pi05.json").read_text())
    result = dict(scope="Single native observation interface check; semantic discrimination requires the frozen 64-observation pilot",
                  input=preflight["input"], input_sha256=preflight["input_sha256"], checkpoint=preflight["checkpoint"],
                  layout=preflight["layout"], device="cpu", layers=[])
    for layer in (5, 16):
        sae = loader.load_sae(assets / f"saes/per_token/paligemma/sae_layer{layer}.safetensors", device="cpu")
        x = captures[f"L{layer}"].reshape(-1, 2048).float()
        with torch.inference_mode():
            z = sae.encode(x)
            reconstruction = sae.decode(z)
        if not torch.isfinite(z).all() or not torch.isfinite(reconstruction).all():
            raise ValueError("Nonfinite local SAE output")
        groups = {"all_sampled_tokens": list(range(len(x)))}
        for kind in sorted(set(captures["kinds"])):
            groups[kind] = [i for i, label in enumerate(captures["kinds"]) if label == kind]
        diagnostics = {}
        for kind, positions in groups.items():
            real, recon = x[positions], reconstruction[positions]
            centered = ((real - real.mean(0, keepdim=True)) ** 2).sum().item()
            sse = ((real - recon) ** 2).sum().item()
            counts = (z[positions] != 0).sum(-1)
            diagnostics[kind] = dict(tokens=len(positions), normalized_mse=float((((real-recon)/sae.std) ** 2).mean()),
                                     fraction_variance_explained=1-sse/centered if centered > 0 else None,
                                     mean_nonzero_features=float(counts.float().mean()))
        concepts = []
        for category, entries in index[f"paligemma_layer_{layer}"].items():
            for concept, entry in entries.items():
                ids = entry["feature_indices"]
                a = z[:, ids].max(0).values
                concepts.append(dict(category=category, concept=concept, candidate_pool_size=len(ids),
                                     candidates_with_positive_evidence=int((a > 0).sum()),
                                     largest_candidate_activation=float(a.max()),
                                     evidence="Raw candidate activation, not a calibrated concept decision"))
        result["layers"].append(dict(layer=layer, diagnostics=diagnostics, concepts=concepts))
        del sae
    out.mkdir(parents=True, exist_ok=True)
    (out / "local_sae_smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(scope=result["scope"], diagnostics={r["layer"]:r["diagnostics"] for r in result["layers"]}), indent=2))


if __name__ == "__main__":
    main()
