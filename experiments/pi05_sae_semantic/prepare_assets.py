"""Download only the two released SAEs and their concept/normalization metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


REQUIRED = ["concepts/concept_features_pi05.json",
            "saes/per_token/paligemma/sae_layer5.safetensors",
            "saes/per_token/paligemma/sae_layer16.safetensors"]
FILES = ["README.md", "REPRODUCIBILITY.md", "manifest.jsonl", "loader.py", *REQUIRED,
         "videos/baseline_index.json", "videos/ablation_index.json"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info("bag100/action-atlas-pi05", files_metadata=True)
    available = {f.rfilename: f for f in info.siblings}
    receipt_path = output / "asset_manifest.json"
    receipt = (json.loads(receipt_path.read_text()) if receipt_path.exists()
               else dict(repo_id=info.id, revision=info.sha, files=[], started_unix=time.time()))
    if receipt["revision"] != info.sha:
        raise ValueError("Release changed; choose a separate output rather than mixing revisions")
    for name in FILES:
        if args.metadata_only and name.endswith(".safetensors"):
            continue
        if name not in available:
            if name in REQUIRED:
                raise FileNotFoundError(f"Required release file missing: {name}")
            continue
        print(json.dumps(dict(event="downloading", filename=name, revision=info.sha)), flush=True)
        downloaded = Path(hf_hub_download(info.id, name, repo_type="dataset", revision=info.sha,
                                           local_dir=output))
        digest = hashlib.sha256()
        with downloaded.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        row = dict(path=str(downloaded), filename=name, bytes=downloaded.stat().st_size,
                   sha256=digest.hexdigest())
        if downloaded.suffix == ".safetensors":
            with safe_open(str(downloaded), framework="pt", device="cpu") as sf:
                row["metadata"] = sf.metadata()
                row["tensor_shapes"] = {k: list(sf.get_slice(k).get_shape()) for k in sf.keys()}
        receipt["files"] = [r for r in receipt["files"] if r["filename"] != name] + [row]
        (output / "asset_manifest.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(dict(event="downloaded", filename=name, bytes=row["bytes"])), flush=True)
    receipt["metadata_only"] = args.metadata_only
    receipt["completed_unix"] = time.time()
    (output / "asset_manifest.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(event="complete", files=len(receipt["files"]), revision=info.sha)), flush=True)


if __name__ == "__main__":
    main()
