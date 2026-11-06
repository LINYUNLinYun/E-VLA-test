"""Fetch small official summaries before downloading raw activation trajectories."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from huggingface_hub import hf_hub_download
    inventory, out = Path(args.inventory), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    receipt = []
    for repo, names in {
        "action-atlas-pi05-activations": [
            "README.md", "DATA_INVENTORY.md", "analysis/LAYER_ANALYSIS_REPORT.md",
            "analysis/layer_analysis_results.json", "analysis/comprehensive_layer_analysis.py",
            *[f"baseline_v3/libero_{suite}/summary_paligemma.json" for suite in ("goal", "object", "spatial")],
        ],
        "action-atlas-viz": [
            "README.md", "processed/contrastive/pi05_paligemma/feature_metadata.json",
            *[f"descriptions/contrastive/all_descriptions_pi05_paligemma_{suite}.json"
              for suite in ("goal", "object", "spatial")],
        ],
    }.items():
        info = json.loads((inventory / (repo + ".json")).read_text())
        available = {r["path"]: r for r in info["files"]}
        for name in names:
            if name not in available:
                continue
            dest = Path(hf_hub_download(info["repo_id"], name, revision=info["revision"], repo_type="dataset",
                                       local_dir=out / repo))
            row = dict(repo_id=info["repo_id"], revision=info["revision"], filename=name,
                       path=str(dest.resolve()), bytes=dest.stat().st_size,
                       sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
            receipt.append(row)
            (out / "reference_manifest.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(dict(event="downloaded", repo=repo, filename=name, bytes=row["bytes"])), flush=True)


if __name__ == "__main__":
    main()
