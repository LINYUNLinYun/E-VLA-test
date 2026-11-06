#!/usr/bin/env python3
"""Export source, documentation and small results from a research workspace."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone

MIB = 1024 * 1024
ROOTS = ('docs', 'experiments', 'preparations', 'reports', 'results', 'scripts',
         'tests', 'profiling', 'profile_results', 'profiles', 'nsys', 'logs', '.agents')
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache',
             '.ruff_cache', 'node_modules', 'site-packages', '_deps', '.cache',
             'wandb', 'tensorboard', '.vscode', '.codex', '.claude',
             'calvin_debug_dataset', 'task_D_D', 'task_ABC_D', 'task_ABCD_D'}
BINARY_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif', '.wav', '.mp3',
               '.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.pkl', '.pickle',
               '.npy', '.npz', '.h5', '.hdf5', '.hdf', '.parquet', '.arrow',
               '.tfrecord', '.rlds', '.zarr', '.zip', '.tar', '.gz', '.bz2', '.xz',
               '.7z', '.whl', '.so', '.a', '.o', '.pyc', '.pyo', '.sqlite', '.db',
               '.nsys-rep', '.qdrep', '.nsight-cuprof-report', '.ncu-rep', '.onnx',
               '.engine', '.plan', '.obj', '.stl', '.msh', '.dae', '.glb', '.fbx',
               '.init', '.pruned_init', '.ttf', '.woff', '.woff2', '.otf'}
IMAGES = {'.png', '.jpg', '.jpeg', '.webp', '.pdf'}
CODE_EXTS = {'.py', '.sh', '.bash', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu',
             '.cuh', '.js', '.ts', '.tsx', '.jsx', '.rs', '.go', '.lua'}
# Simulator assets are restored from their exact upstream revisions as needed.
ASSET_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.obj', '.stl', '.msh', '.dae',
              '.glb', '.fbx', '.init', '.pruned_init', '.mtl'}
SECRET_NAME = re.compile(r'^(?:\.env(?:\..*)?|id_rsa|id_ed25519|credentials(?:\..*)?|token|\.netrc)$', re.I)


def git(repo, *args):
    return subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(repo), *args],
                          check=True, capture_output=True).stdout


def walk(root, relroot, skipped):
    # rg prunes environment directories without an NFS stat for every file.
    extensions = CODE_EXTS | {'.md', '.rst', '.txt', '.json', '.jsonl', '.csv', '.tsv',
        '.yaml', '.yml', '.toml', '.in', '.png', '.jpg', '.jpeg', '.webp', '.pdf',
        '.svg', '.xml', '.urdf', '.bddl', '.ipynb', '.html', '.css', '.scss', '.lock'}
    command = ['rg', '--files', '--hidden', '--no-ignore', '--null']
    for suffix in sorted(extensions):
        command += ['-g', '*' + suffix]
    for name in sorted(SKIP_DIRS):
        command += ['-g', '!**/' + name + '/**']
    for pattern in ('.venv*', '.cache*', '*.egg-info', '*.dist-info'):
        command += ['-g', '!**/' + pattern + '/**']
    result = subprocess.run(command + [str(root / relroot)], capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode())
    for raw in result.stdout.split(b'\0'):
        if not raw:
            continue
        rel = Path(os.fsdecode(raw)).relative_to(root)
        reason = exclude(rel, 0)
        if reason:
            skipped[reason] += 1
        else:
            yield rel


def exclude(path, size):
    name = path.name
    suffix = path.suffix.lower()
    if SECRET_NAME.match(name) or suffix in {'.pem', '.key'}:
        return 'credential_filename'
    if suffix in BINARY_EXTS:
        return 'video_weight_dataset_binary_or_trace'
    if name == '.gitmodules':
        return 'flattened_submodule_metadata'
    if suffix in {'.log', '.pid', '.tmp', '.sock'} or (suffix == '.lock' and name not in {'uv.lock', 'poetry.lock', 'yarn.lock'}):
        return 'runtime_log_lock_or_temporary'
    if suffix in ASSET_EXTS and any(p in {'assets', 'init_files', 'init_states', 'textures', 'meshes'} for p in path.parts):
        return 'upstream_simulation_asset'
    # Keep source and prose even inside raw result directories.
    is_result = any(p in {'results', 'logs', 'nsys', 'profiles', 'profile_results', 'data', 'rollouts'} for p in path.parts)
    if suffix not in CODE_EXTS | {'.md', '.rst'} and any(p in {'episodes', 'observations', 'trajectories', 'frames', 'captures', 'features', 'tensor_cache', 'source_cache', 'media_audits', 'media', 'logs'} for p in path.parts):
        return 'raw_episode_observation_or_feature_artifact'
    if suffix in IMAGES and path.parts[0] == 'results' and any(p in {'observations', 'frames', 'raw_images', 'camera_images'} for p in path.parts):
        return 'raw_observation_image'
    if is_result and suffix not in CODE_EXTS | {'.md', '.rst'} and any(re.match(r'^(?:task|init|sample|request|prediction|episode|ep)[_-]?[0-9]', part) for part in path.parts):
        return 'raw_per_case_or_per_request_metadata'
    archive_result = any(part in {'results', 'logs', 'profiles', 'profile_results', 'nsys'} for part in path.parts) or ('data' in path.parts and 'test' in path.parts)
    if archive_result and suffix == '.json':
        summary_name = any(word in name.lower() for word in ('summary', 'manifest', 'config', 'protocol', 'audit', 'metrics', 'status', 'progress', 'candidate', 'threshold', 'concept', 'rules', 'selection', 'operating_point', 'inventory', 'plan', 'report', 'integrity', 'check', 'readiness'))
        limit = 256 * 1024 if summary_name else 32 * 1024
        if size > limit:
            return 'raw_or_oversized_result_json'
    if archive_result and suffix == '.jsonl' and not any(word in name.lower() for word in ('manifest', 'observations')):
        return 'raw_result_stream'
    if suffix == '.json' and (re.match(r'prediction[0-9]+\.json$', name) or name.endswith('.command.json')):
        return 'raw_prediction_or_media_command'
    if 'demo_data' in path.parts and suffix not in CODE_EXTS | {'.md', '.rst'}:
        return 'upstream_demo_dataset'
    if suffix not in CODE_EXTS and is_result and path.parts[0] not in {'docs', 'reports'}:
        if suffix == '.jsonl' and any(x in name.lower() for x in ('request', 'trace', 'telemetry', 'ledger', 'event', 'monitor', 'stdout')):
            return 'raw_request_or_runtime_stream'
        if size > MIB:
            return 'result_over_1_MiB'
    if suffix in IMAGES | {'.svg'} and size > MIB:
        return 'illustration_over_1_MiB'
    if size > 10 * MIB:
        return 'file_over_10_MiB'
    return None


def write_snapshot_manifest(out, rows):
    """Keep source hash inventories below the repository's per-file size limit."""
    fields = ('path', 'bytes', 'sha256')
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter='\t')
    writer.writeheader()
    parts = []

    def save_part():
        raw = buffer.getvalue().encode()
        folder = out / 'snapshot_files'
        folder.mkdir(exist_ok=True)
        target = folder / f'part-{len(parts) + 1:04d}.tsv'
        target.write_bytes(raw)
        parts.append({'path': 'docs/repository/' + target.relative_to(out).as_posix(),
                      'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        buffer.seek(0)
        buffer.truncate(0)
        writer.writeheader()

    for row in rows:
        writer.writerow(row)
        if buffer.tell() >= 4 * MIB:
            save_part()
    if not parts:
        (out / 'snapshot_files.tsv').write_text(buffer.getvalue())
    else:
        if buffer.tell() > len('path\tbytes\tsha256\r\n'):
            save_part()
        with (out / 'snapshot_files.tsv').open('w', newline='') as stream:
            index = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
            index.writeheader()
            index.writerows(parts)
    folder = out / 'snapshot_files'
    active = {Path(part['path']).name for part in parts}
    if folder.exists():
        for old in folder.glob('part-*.tsv'):
            if old.name not in active:
                old.unlink()
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('destination', type=Path)
    args = ap.parse_args()
    root, dest = args.source.resolve(), args.destination.resolve()
    if root == dest or root in dest.parents or dest in root.parents:
        ap.error('Source and destination must be separate, non-nested directories')
    dest.mkdir(parents=True, exist_ok=True)
    skipped = collections.Counter()
    excluded = []
    candidates = set(p.relative_to(root) for p in root.iterdir() if p.is_file() and p.suffix.lower() in {'.md', '.txt'})
    for top in ROOTS:
        if (root / top).is_dir():
            candidates.update(walk(root, top, skipped))
            print(f'Indexed {top}', flush=True)
    upstreams = []
    for top in ('policies', 'LIBERO', 'SimplerEnv', 'calvin'):
        if not (root / top).is_dir():
            continue
        for base, dirs, files in os.walk(root / top, followlinks=False):
            repo = Path(base)
            has_git = '.git' in dirs or '.git' in files
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                       and not d.startswith(('.venv', '.cache'))
                       and not d.endswith(('.egg-info', '.dist-info'))
                       and d not in {'data', 'results', 'logs', 'assets', 'images', 'media', 'datasets', 'demo_data', 'calvin_debug_dataset', 'task_D_D', 'task_ABC_D', 'task_ABCD_D'}
                       and not (repo / d).is_symlink()]
            if not has_git:
                continue
            rel = repo.relative_to(root)
            tracked = git(repo, 'ls-files', '-z').decode().split('\0')
            untracked = git(repo, 'ls-files', '-z', '--others', '--exclude-standard').decode().split('\0')
            candidates.update(rel / n for n in tracked + untracked if n and exclude(rel / n, 0) is None and (repo / n).is_file())
            upstreams.append({'path': rel.as_posix(), 'url': git(repo, 'remote', 'get-url', 'origin').decode().strip(),
                              'commit': git(repo, 'rev-parse', 'HEAD').decode().strip(),
                              'tracked_changes': git(repo, 'diff', 'HEAD', '--name-only').decode().splitlines(),
                              'tracked_paths': tracked, 'restore_paths': []})
        # Include standalone local policies and ignored source modifications too.
        for rel in walk(root, top, skipped):
            if rel.suffix.lower() in CODE_EXTS | {'.md', '.rst', '.json', '.jsonl', '.csv', '.tsv', '.yaml', '.yml', '.toml', '.png', '.svg', '.txt', '.in', '.pdf', '.jpg', '.jpeg', '.webp'} or rel.name in {'uv.lock', 'poetry.lock', 'yarn.lock'}:
                candidates.add(rel)
        print(f'Indexed {top}', flush=True)
    rows, failed = [], []

    def copy_one(rel):
        src = root / rel
        if src.is_symlink():
            return 'skip', 'symlink_file'
        try:
            size = src.stat().st_size
            reason = exclude(rel, size)
            if not reason:
                raw = src.read_bytes()
                if rel.suffix.lower() not in IMAGES:
                    if b'\0' in raw:
                        reason = 'binary_content'
                    else:
                        try:
                            raw.decode('utf-8')
                        except UnicodeDecodeError:
                            reason = 'non_utf8_content'
            if reason:
                return 'excluded', {'path': rel.as_posix(), 'bytes': size, 'reason': reason}
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists() or out.read_bytes() != raw:
                out.write_bytes(raw)
            out.chmod(0o755 if src.stat().st_mode & 0o111 else 0o644)
            return 'row', {'path': rel.as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        except OSError as exc:
            return 'error', {'path': rel.as_posix(), 'error': str(exc)}

    print(f'Copying {len(candidates)} candidate paths with 16 I/O workers', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for index, (kind, value) in enumerate(pool.map(copy_one, sorted(candidates)), 1):
            if kind == 'row':
                rows.append(value)
            elif kind == 'excluded':
                skipped[value['reason']] += 1
                excluded.append(value)
            elif kind == 'skip':
                skipped[value] += 1
            else:
                failed.append(value)
            if index % 10000 == 0:
                print(f'Processed {index}/{len(candidates)}; retained {len(rows)}', flush=True)
    retained = {r['path'] for r in rows}
    for upstream in upstreams:
        upstream['restore_paths'] = sorted(n for n in upstream.pop('tracked_paths') if n and
            (Path(upstream['path']) / n).as_posix() not in retained and
            Path(n).suffix.lower() in ASSET_EXTS)
    out = dest / 'docs/repository'
    out.mkdir(parents=True, exist_ok=True)
    write_snapshot_manifest(out, rows)
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_workspace': str(root),
               'snapshot_isolation': 'Each copied file is hashed; active experiments are not paused. Files are not a transactional snapshot of all runs.',
               'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
               'by_root': dict(collections.Counter(Path(r['path']).parts[0] for r in rows)),
               'excluded_counts': dict(skipped), 'read_errors': failed,
               'selection_note': 'Source/prose/config/table/figure extensions selected by rg; binaries, environments and caches are not exhaustively inventoried. Upstream simulator asset paths are in upstreams.json.'}
    for name, data in [('snapshot_summary.json', summary), ('upstreams.json', upstreams), ('excluded_files.json', excluded)]:
        (out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failed:
        raise SystemExit('Some files could not be copied; review snapshot_summary.json')


if __name__ == '__main__':
    main()
