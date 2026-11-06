#!/usr/bin/env python3
"""Check tracked Git contents for heavy artifacts, credentials and size limits."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

from export_workspace import BINARY_EXTS, IMAGES, MIB, SECRET_NAME

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github_token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    'huggingface_token': re.compile(rb'\bhf_[A-Za-z0-9]{30,}\b'),
    'openai_key': re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b'),
    'aws_access_key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
}


def snapshot_hashes():
    expected, errors = {}, []
    manifests = [ROOT / 'docs/repository/snapshot_files.tsv']
    seen = set()
    while manifests:
        manifest = manifests.pop()
        if manifest in seen:
            continue
        seen.add(manifest)
        with manifest.open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                path = ROOT / row['path']
                path.resolve().relative_to(ROOT)
                expected[row['path']] = row['sha256']
                if row['path'].startswith('docs/repository/snapshot_files/') and path.suffix == '.tsv':
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                        errors.append(f"{row['path']}: manifest hash mismatch")
                    else:
                        manifests.append(path)
    return expected, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify-snapshot', action='store_true', help='Also verify the published source snapshot hashes')
    ap.add_argument('--paths-file', type=Path, help='Check a NUL-separated publication file list before staging')
    args = ap.parse_args()
    if args.paths_file:
        listed = b'\0'.join(b'100644 candidate 0\t' + name for name in args.paths_file.read_bytes().split(b'\0') if name)
    else:
        listed = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-s', '-z'], capture_output=True, check=True).stdout
    expected, errors = snapshot_hashes() if args.verify_snapshot else ({}, [])
    entries = [entry for entry in listed.split(b'\0') if entry]

    def inspect(entry):
        metadata, rawname = entry.split(b'\t', 1)
        name = rawname.decode()
        path = ROOT / name
        failures = []
        if metadata.split()[0] == b'160000':
            return name, 0, [f'{name}: nested Git link instead of source files']
        if not path.is_file() or path.is_symlink():
            return name, 0, [f'{name}: missing file or symlink']
        size = path.stat().st_size
        suffix = path.suffix.lower()
        if suffix in BINARY_EXTS:
            failures.append(f'{name}: prohibited artifact extension')
        if size > 10 * MIB or (suffix in IMAGES | {'.svg'} and size > MIB):
            failures.append(f'{name}: exceeds repository file size limit')
        if SECRET_NAME.match(path.name) or suffix in {'.pem', '.key'}:
            failures.append(f'{name}: credential filename')
        if suffix not in IMAGES or name in expected:
            raw = path.read_bytes()
            if name in expected and hashlib.sha256(raw).hexdigest() != expected[name]:
                failures.append(f'{name}: snapshot hash mismatch')
            if suffix not in IMAGES:
                if b'\0' in raw:
                    failures.append(f'{name}: binary content')
                for label, pattern in PATTERNS.items():
                    match = pattern.search(raw)
                    if match:
                        line = raw[:match.start()].count(b'\n') + 1
                        failures.append(f'{name}:{line}: possible {label} (value withheld)')
        return name, size, failures

    sizes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for name, size, failures in pool.map(inspect, entries):
            sizes.append((size, name))
            errors.extend(failures)
    if args.verify_snapshot:
        missing = set(expected) - {name for _, name in sizes}
        errors.extend(f'{name}: snapshot file not tracked' for name in sorted(missing))
    report = dict(files=len(sizes), bytes=sum(s for s, _ in sizes), largest_files=sorted(sizes, reverse=True)[:10], errors=errors)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
