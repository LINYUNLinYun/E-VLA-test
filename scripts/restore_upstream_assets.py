#!/usr/bin/env python3
"""Restore omitted simulator assets from a recorded upstream Git commit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', action='append', default=[], help='Exact workspace path; repeatable')
    parser.add_argument('--list', action='store_true', help='List available projects without downloading')
    args = parser.parse_args()
    records = json.loads((ROOT / 'docs/repository/upstreams.json').read_text())
    known = {r['path'] for r in records}
    if set(args.repo) - known:
        parser.error('Unknown project: ' + ', '.join(sorted(set(args.repo) - known)))
    selected = [r for r in records if not args.repo or r['path'] in args.repo]
    if not args.repo or args.list:
        for record in selected:
            print(f"{record['path']}: {len(record['restore_paths'])} omitted assets at {record['commit']}")
        return
    for record in selected:
        project = (ROOT / record['path']).resolve()
        project.relative_to(ROOT)
        paths = []
        for name in record['restore_paths']:
            rel = PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Unsafe asset path: {name}')
            target = project.joinpath(*rel.parts)
            target.resolve().relative_to(ROOT)
            if not target.exists() and not target.is_symlink():
                paths.append((rel, target))
        if not paths:
            print(f"{record['path']}: no missing assets")
            continue
        with tempfile.TemporaryDirectory(prefix='evla-upstream-') as tmp:
            checkout = Path(tmp)
            subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'fetch', '--depth=1', record['url'], record['commit']], check=True)
            for rel, target in paths:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Fetching creates no checkout; copy only the requested Git blobs.
                with tempfile.TemporaryFile() as stream:
                    subprocess.run(['git', '-C', str(checkout), 'show', f"FETCH_HEAD:{rel.as_posix()}"], stdout=stream, check=True)
                    stream.seek(0)
                    with target.open('xb') as output:
                        shutil.copyfileobj(stream, output)
        print(f"{record['path']}: restored {len(paths)} assets; existing files preserved")


if __name__ == '__main__':
    main()
