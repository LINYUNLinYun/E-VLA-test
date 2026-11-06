"""Download LIBERO HDF5 demonstrations directly from a Hugging Face mirror.

The standard ``huggingface_hub.snapshot_download`` path can fail on some
mirrors during metadata validation. This downloader uses the repository API and
plain HTTP file requests instead, with resumable ``.part`` files.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import quote

import requests


REPO_ID = "yifengzhu-hf/LIBERO-datasets"
DATASET_NAMES = (
    "libero_object",
    "libero_goal",
    "libero_spatial",
    "libero_10",
    "libero_90",
)


def list_files(dataset_name: str, endpoint: str) -> List[str]:
    """Return files in one dataset subdirectory of the public repository."""
    url = f"{endpoint.rstrip('/')}/api/datasets/{REPO_ID}/tree/main/{dataset_name}"
    response = requests.get(
        url, params={"recursive": "true", "expand": "false"}, timeout=(20, 60)
    )
    response.raise_for_status()
    entries = response.json()
    return [entry["path"] for entry in entries if entry.get("type") == "file"]


def download_file(
    file_path: str, output_dir: Path, endpoint: str, timeout: Tuple[int, int]
) -> Tuple[str, str]:
    """Download one file, resuming an interrupted ``.part`` file if present."""
    destination = output_dir / file_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if destination.is_file() and destination.stat().st_size > 0:
        return file_path, "already complete"

    downloaded = partial.stat().st_size if partial.exists() else 0
    url = (
        f"{endpoint.rstrip('/')}/datasets/{REPO_ID}/resolve/main/"
        f"{quote(file_path, safe='/')}"
    )
    # curl is more tolerant of intermittent TLS disconnects than the Python
    # requests client used by the original downloader. ``-C -`` resumes the
    # existing partial file; retries cover transient mirror failures.
    command = [
        "curl",
        "--fail",
        "--location",
        "--http1.1",
        "--retry",
        "12",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "0",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        # Some mirrors reject a Range request for an incomplete object. Retry
        # that one file from the beginning instead of leaving a bad partial.
        if downloaded:
            partial.unlink(missing_ok=True)
            command[command.index("--continue-at") + 1] = "0"
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1:] or ["curl failed"]
            raise RuntimeError(detail[0])
    os.replace(partial, destination)
    return file_path, "downloaded"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint", default="https://hf-mirror.com", help="Hugging Face endpoint"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for dataset_name in args.datasets:
        dataset_files = list_files(dataset_name, args.endpoint)
        print(f"{dataset_name}: {len(dataset_files)} files", flush=True)
        files.extend(dataset_files)
    print(f"Total files: {len(files)}", flush=True)

    timeout = (30, 300)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_file, file_path, args.output_dir, args.endpoint, timeout
            ): file_path
            for file_path in files
        }
        for index, future in enumerate(as_completed(futures), start=1):
            file_path = futures[future]
            try:
                _, status = future.result()
                print(f"[{index}/{len(files)}] {status}: {file_path}", flush=True)
            except Exception as error:
                print(f"[error] {file_path}: {error}", flush=True)


if __name__ == "__main__":
    main()
