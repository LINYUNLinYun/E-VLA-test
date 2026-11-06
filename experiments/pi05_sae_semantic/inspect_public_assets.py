"""Inventory public artifacts before choosing which activations to download."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from huggingface_hub import HfApi
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for name in ["bag100/action-atlas-pi05", "bag100/action-atlas-pi05-activations", "bag100/action-atlas-viz"]:
        info = api.dataset_info(name, files_metadata=True)
        rows = [dict(path=f.rfilename, size=f.size, blob_id=f.blob_id,
                     lfs=f.lfs) for f in info.siblings]
        # Some hub versions expose LFS info as a dataclass rather than a dict.
        record = dict(repo_id=name, revision=info.sha, files=rows)
        (out / (name.split("/")[-1] + ".json")).write_text(json.dumps(record, indent=2, default=vars))
        interesting = [r for r in rows if re.search(r"pi05|concept|ablat|paligemma|layer(05|16)", r["path"], re.I)]
        print(json.dumps(dict(repo_id=name, revision=info.sha, n_files=len(rows),
                              total_bytes=sum(r["size"] or 0 for r in rows),
                              prefixes=Counter(r["path"].split("/")[0] for r in rows),
                              examples=interesting[:18] or rows[:12]), default=vars), flush=True)


if __name__ == "__main__":
    main()
