"""Compare native scanned Full and matched selected-scan Full benchmark artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import atomic_json


def compare(native,matched):
    if native["manifest_hash"]!=matched["manifest_hash"]:
        raise ValueError("Different frozen manifests")
    if native["condition"]["model"]!=matched["condition"]["model"]:
        raise ValueError("Different model/checkpoint")
    if native["condition"]["skip"] or matched["condition"]["skip"]:
        raise ValueError("Both full27 references must execute all vision blocks")
    checks = []
    for left,right in zip(native["rows"][:2],matched["rows"][:2],strict=True):
        if left["input_hash"]!=right["input_hash"] or left["noise_hash"]!=right["noise_hash"]:
            raise ValueError("Unpaired real observations or initial noise")
        a,b = np.asarray(left["parity_actions"]),np.asarray(right["parity_actions"])
        checks.append(dict(max_abs=float(np.max(np.abs(a-b))),rmse=float(np.sqrt(np.mean((a-b)**2))),
                           passed=bool(np.allclose(a,b,atol=1e-5,rtol=1e-5))))
    return dict(passed=len(checks)==2 and all(c["passed"] for c in checks),atol=1e-5,rtol=1e-5,checks=checks,
                manifest_hash=native["manifest_hash"],model=native["condition"]["model"],
                scope="All27 vision blocks; original scanned VLM/AE unchanged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native",required=True)
    parser.add_argument("--matched",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = compare(json.loads(Path(args.native).read_text()),json.loads(Path(args.matched).read_text()))
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
