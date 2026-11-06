"""Validate existing, independently labeled data and construct safe splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pi05_hidden_probe.data import (
    ManifestError, read_manifest, split_manifest, split_summary, write_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing labeled JSONL; relative data paths are resolved here")
    parser.add_argument("--output", help="Destination JSONL (required unless --check-only)")
    parser.add_argument("--calibration-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true", help="Validate metadata and existing splits without writing files")
    args = parser.parse_args(argv)
    if not args.check_only and not args.output:
        parser.error("--output is required unless --check-only is selected")
    try:
        rows = read_manifest(args.input)
        if not args.check_only:
            rows = split_manifest(rows, args.calibration_fraction, args.seed)
        summary = split_summary(rows)
        summary["requested_calibration_fraction"] = args.calibration_fraction
        summary["seed"] = args.seed
        summary["source_manifest"] = str(Path(args.input).resolve())
        summary["validation_scope"] = "metadata and split isolation; arrays are checked by load_observation at execution"
        if not args.check_only:
            destination = Path(args.output)
            summary_path = destination.with_name("split_manifest.json")
            if summary_path.resolve() == destination.resolve():
                raise ManifestError("--output cannot be named split_manifest.json (reserved for split summary)")
            write_jsonl(destination, rows)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except (ManifestError, OSError) as exc:
        parser.exit(2, f"manifest error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
