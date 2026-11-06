#!/usr/bin/env python3
"""Explicit, read-only streaming verification of local HF safetensors shards."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

CHUNK_BYTES = 8 * 1024 * 1024


def confined_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid remote filename")
    parts = PurePosixPath(name)
    if parts.is_absolute() or any(p in (".", "..") for p in name.split("/")):
        raise ValueError(f"unsafe remote filename: {name}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"remote filename/symlink escapes checkpoint: {name}")
    return path


def stream_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        import os
        before = os.fstat(source.fileno())
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
        after = os.fstat(source.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise ValueError(f"file changed during hashing: {path}")
    return digest.hexdigest()


def record(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": stream_hash(path)}


def verify(checkpoint, hub_file_info, revision, index=None):
    checkpoint = checkpoint.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be exact 40-character HF commit SHA")
    if not checkpoint.is_dir() or checkpoint.name != revision:
        raise ValueError("checkpoint must exist at a directory named by exact revision")
    hub = json.loads(hub_file_info.read_text())
    if hub.get("sha") != revision:
        raise ValueError("hub_file_info revision does not match requested revision")
    siblings = {}
    for entry in hub.get("siblings", []):
        name = entry.get("rfilename")
        confined_file(checkpoint, name)
        if name in siblings:
            raise ValueError(f"duplicate remote filename: {name}")
        siblings[name] = entry
    index = index or checkpoint / "model.safetensors.index.json"
    index_bytes = index.read_bytes()
    index_entry = siblings.get("model.safetensors.index.json", {})
    blob = hashlib.sha1(b"blob " + str(len(index_bytes)).encode() + b"\0" + index_bytes).hexdigest()
    if index_entry.get("blobId") != blob:
        raise ValueError("index Git blob SHA does not match revision-bound hub metadata")
    weights = json.loads(index_bytes).get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("index has no nonempty weight_map")
    indexed_shards = set(weights.values())
    for name in indexed_shards:
        confined_file(checkpoint, name)
        if not name.endswith(".safetensors"):
            raise ValueError(f"index references a non-safetensors shard: {name}")
    all_shards = indexed_shards | {name for name in siblings if name.endswith(".safetensors")}
    if not all_shards:
        raise ValueError("no safetensors shards declared")
    results = []
    for name in sorted(all_shards):
        path = confined_file(checkpoint, name)
        remote = siblings.get(name, {})
        lfs = remote.get("lfs", {})
        expected_size, expected_sha = lfs.get("size"), lfs.get("sha256")
        row = {"file": name, "path": str(path), "indexed": name in indexed_shards,
               "expected_bytes": expected_size, "expected_sha256": expected_sha, "ok": False}
        if type(expected_size) is not int or expected_size < 0 or not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            row["error"] = "missing/invalid revision-bound LFS size or SHA256"
        elif remote.get("size", expected_size) != expected_size:
            row["error"] = "hub sibling size conflicts with LFS size"
        elif not path.is_file():
            row["error"] = "shard missing or not a regular file"
        elif path.stat().st_size != expected_size:
            row["actual_bytes"] = path.stat().st_size
            row["error"] = "size mismatch; hash skipped"
        else:
            try:
                row["actual_sha256"] = stream_hash(path)
                row["ok"] = row["actual_sha256"] == expected_sha
                if not row["ok"]:
                    row["error"] = "SHA256 mismatch"
            except (OSError, ValueError) as exc:
                row["error"] = str(exc)
        results.append(row)
    return {"schema_version": 1, "mode": "explicit_local_weight_verification", "revision": revision,
            "hub_repo_id": hub.get("id"), "checkpoint": str(checkpoint),
            "verifier_source": record(Path(__file__)),
            "hub_metadata": record(hub_file_info), "weight_index": record(index), "index_git_blob_sha1": blob,
            "shard_count": len(results), "indexed_shard_count": len(indexed_shards), "shards": results,
            "ok": all(row["ok"] for row in results), "model_loaded": False,
            "note": "Verifies bytes against supplied revision-bound HF API metadata; no inference or architecture parity."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hub-file-info", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--index", type=Path, help="explicit local index if not in checkpoint directory")
    parser.add_argument("--write-manifest", type=Path, help="optional NEW output file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.write_manifest and args.write_manifest.exists():
        parser.exit(2, "verification error: output already exists\n")
    try:
        result = verify(args.checkpoint, args.hub_file_info, args.revision, args.index)
        if args.write_manifest:
            with args.write_manifest.open("x", encoding="utf-8") as output:
                json.dump(result, output, indent=2)
                output.write("\n")
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"verification error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
