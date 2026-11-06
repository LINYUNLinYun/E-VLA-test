"""Use the release's own loader to encode a bounded public activation sample."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


def load_official_loader(assets):
    spec = importlib.util.spec_from_file_location("action_atlas_release_loader", assets / "loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episode", default="baseline_v3/libero_goal/paligemma/layer{layer:02d}/task02_ep01_s123.npz")
    p.add_argument("--observations", type=int, default=32)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(args.threads)
    assets, cache, output = Path(args.assets).resolve(), Path(args.cache).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads(Path(args.inventory).read_text())
    available = {r["path"]: r for r in info["files"]}
    loader = load_official_loader(assets)
    receipt = dict(scope="Public-source activation compatibility; not local checkpoint transfer or routing validation",
                   repo_id=info["repo_id"], revision=info["revision"], device="cpu", threads=args.threads,
                   official_loader=str(assets / "loader.py"),
                   official_loader_sha256=hashlib.sha256((assets / "loader.py").read_bytes()).hexdigest(), layers=[])
    for layer in (5, 16):
        name = args.episode.format(layer=layer)
        if name not in available:
            raise FileNotFoundError(name)
        print(json.dumps(dict(event="downloading_public_activation", layer=layer, file=name,
                              bytes=available[name]["size"])), flush=True)
        path = Path(hf_hub_download(info["repo_id"], name, repo_type="dataset", revision=info["revision"], local_dir=cache))
        expected = (available[name].get("lfs") or {}).get("sha256")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if expected and expected != digest.hexdigest():
            raise ValueError(f"Activation checksum mismatch: {name}")
        print(json.dumps(dict(event="reading_public_activation", layer=layer)), flush=True)
        with np.load(path, allow_pickle=False) as archive:
            if "activations" not in archive:
                raise ValueError(f"Expected the documented 'activations' array; got {archive.files}")
            array = archive["activations"]
            shape, dtype = list(array.shape), str(array.dtype)
            if array.shape[-1] != 2048 or array.ndim < 3:
                raise ValueError(f"Not per-token PaliGemma activations: {shape}, {dtype}")
            obs_ids = np.unique(np.linspace(0, len(array) - 1, min(args.observations, len(array))).astype(int))
            observations = array[obs_ids].reshape(len(obs_ids), -1, array.shape[-1])
            token_ids = np.unique(np.linspace(0, observations.shape[1] - 1, min(args.tokens, observations.shape[1])).astype(int))
            sample = np.asarray(observations[:, token_ids], dtype=np.float32).copy()
            other = {}
            for key in archive.files:
                if key == "activations":
                    continue
                try:
                    value = archive[key]
                    if value.size <= 32:
                        other[key] = value.tolist()
                    else:
                        other[key] = dict(shape=list(value.shape), dtype=str(value.dtype))
                except ValueError:
                    other[key] = "Object metadata omitted; pickle decoding disabled"
        del array, observations
        sae_path = assets / f"saes/per_token/paligemma/sae_layer{layer}.safetensors"
        sae = loader.load_sae(sae_path, device="cpu")
        x = torch.from_numpy(sample.reshape(-1, 2048))
        if not torch.isfinite(x).all():
            raise ValueError("Public sample has nonfinite activations")
        started = time.perf_counter()
        sparse_counts, codes_max = [], []
        squared_error = standardized_error = standardized_energy = 0.
        all_reconstruction = []
        with torch.inference_mode():
            for chunk in x.split(128):
                z = sae.encode(chunk)
                recon = sae.decode(z)
                if not torch.isfinite(z).all() or not torch.isfinite(recon).all():
                    raise ValueError("Nonfinite SAE result")
                sparse_counts.append((z != 0).sum(-1))
                codes_max.append(z.max(-1).values)
                squared_error += ((recon - chunk) ** 2).sum().item()
                standardized_error += (((recon - chunk) / sae.std) ** 2).sum().item()
                standardized_energy += (((chunk - sae.mean) / sae.std) ** 2).sum().item()
                all_reconstruction.append(recon)
        counts = torch.cat(sparse_counts)
        centered_energy = ((x - x.mean(0, keepdim=True)) ** 2).sum().item()
        row = dict(layer=layer, activation_path=str(path), activation_sha256=digest.hexdigest(),
                   raw_shape=shape, raw_dtype=dtype, auxiliary_metadata=other,
                   observation_ids=obs_ids.tolist(), token_ids=token_ids.tolist(),
                   sampling="Uniform sequence positions; public dump does not yet establish instruction/image spans or padding masks",
                   sample_shape=list(sample.shape), metadata=sae.metadata,
                   raw_mse=squared_error / x.numel(), normalized_mse=standardized_error / x.numel(),
                   normalized_input_energy=standardized_energy / x.numel(),
                   fraction_variance_explained=1 - squared_error / centered_energy if centered_energy > 0 else None,
                   nonzero_features_min=int(counts.min()), nonzero_features_max=int(counts.max()),
                   nonzero_features_mean=float(counts.float().mean()), elapsed_encode_decode_s=time.perf_counter() - started,
                   semantic_separation="Not evaluated from a single episode",
                   local_checkpoint_transfer="Pending local observations")
        receipt["layers"].append(row)
        np.savez_compressed(output / f"public_activation_sample_L{layer}.npz", activations=sample,
                            observation_ids=obs_ids, token_ids=token_ids)
        (output / "public_sae_check.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(dict(event="public_sae_checked", **{k: row[k] for k in ["layer", "raw_shape", "sample_shape", "normalized_mse", "fraction_variance_explained", "nonzero_features_mean"]})), flush=True)
        del sae, x, all_reconstruction
    receipt["completed"] = True
    (output / "public_sae_check.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
