"""Bounded public asset/provenance metadata reads; never fetch model weights.

Every remote read has a byte cap. Hub model metadata and small JSON/Markdown
files are sufficient to pin acquisition URLs without downloading checkpoints.
"""
import argparse
import ast
from collections import Counter
import csv
import stat
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT/'results/idea1_v3_2_e4_supplement/asset_audit'
MAX_METADATA_BYTES = 4*1024*1024
MAX_SMALL_FILE_BYTES = 1024*1024


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def bounded_read(url, maximum=MAX_METADATA_BYTES, timeout=45):
    if not url.startswith('https://'):
        raise ValueError('Only explicit HTTPS metadata URLs are allowed')
    request = urllib.request.Request(url, headers={'User-Agent': 'benchmark-asset-provenance/1.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length = response.headers.get('Content-Length')
        if length is not None and int(length) > maximum:
            raise ValueError('Remote file exceeds byte cap before reading')
        data = response.read(maximum+1)
        if len(data) > maximum:
            raise ValueError('Remote body exceeds metadata byte cap')
        return data, dict(requested_url=url, resolved_url=response.url, bytes=len(data),
                          sha256=digest_bytes(data), content_type=response.headers.get('Content-Type'),
                          etag=response.headers.get('ETag'), last_modified=response.headers.get('Last-Modified'))


def hf_search(term, author=None, limit=50):
    params = dict(search=term, limit=min(limit, 100), full='true')
    if author:
        params['author'] = author
    url = 'https://huggingface.co/api/models?'+urllib.parse.urlencode(params)
    body, receipt = bounded_read(url)
    return json.loads(body), receipt


def hf_model(repository):
    url = 'https://huggingface.co/api/models/'+urllib.parse.quote(repository, safe='/')+'?blobs=true'
    body, receipt = bounded_read(url)
    model = json.loads(body)
    revision = model.get('sha')
    if not revision or len(revision) != 40:
        raise ValueError('Hub API did not return an immutable revision')
    return model, receipt


def resolve_url(repository, revision, filename):
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('A full immutable lowercase commit SHA is required')
    if filename.startswith('/') or '..' in Path(filename).parts:
        raise ValueError('Unsafe repository filename')
    return f'https://huggingface.co/{repository}/resolve/{revision}/'+urllib.parse.quote(filename, safe='/')


def file_record(repository, revision, row):
    path = row['rfilename']; lfs = row.get('lfs') or {}
    return dict(repository=repository, revision=revision, path=path, size=row.get('size', lfs.get('size')),
                url=resolve_url(repository, revision, path),
                sha256=lfs.get('sha256') or lfs.get('oid'), git_blob_sha1=row.get('blobId'),
                large_file=bool(lfs), metadata_only=Path(path).suffix.lower() in ('.json', '.md', '.yaml', '.yml', '.txt'))


def inventory(repository, output, small_files=True):
    output = Path(output)/repository.replace('/', '__')
    output.mkdir(parents=True, exist_ok=False)
    model, api_receipt = hf_model(repository)
    revision = model['sha']
    files = [file_record(repository, revision, row) for row in model.get('siblings', [])]
    save_json(output/'hub_model.json', model)
    small_receipts = []
    if small_files:
        for item in files:
            if not item['metadata_only'] or item['size'] is None or item['size'] > MAX_SMALL_FILE_BYTES:
                continue
            # Only small textual metadata; safetensors, pickle, pt and archives
            # cannot pass this extension whitelist, even with a misleading size.
            try:
                body, receipt = bounded_read(item['url'], MAX_SMALL_FILE_BYTES)
                target = output/'small_files'/item['path']; target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(body)
                small_receipts.append(dict(path=item['path'], local=str(target), **receipt))
            except (urllib.error.HTTPError, urllib.error.URLError, ValueError, TimeoutError) as error:
                small_receipts.append(dict(path=item['path'], status='metadata_read_failed', error=str(error)))
    receipt = dict(status='metadata_inventory_complete', repository=repository, revision=revision,
                   acquired_utc=datetime.now(timezone.utc).isoformat(), api=api_receipt,
                   files=files, small_file_receipts=small_receipts, checkpoint_bytes_downloaded=0,
                   qualification='Inventory is not model compatibility or benchmark-contract certification')
    save_json(output/'receipt.json', receipt)
    return receipt


def audit_libero_plus(repository_root, original_task_map, output):
    """Static suite audit without importing LIBERO or touching its global config."""
    root = Path(repository_root).resolve(); output = Path(output)
    classification = root/'libero/libero/benchmark/task_classification.json'
    task_map_path = root/'libero/libero/benchmark/libero_suite_task_map.py'
    def literal_map(path):
        tree = ast.parse(Path(path).read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets))
        return ast.literal_eval(assignment.value)
    original = literal_map(original_task_map); current = literal_map(task_map_path)
    categories = json.loads(classification.read_text()); rows = []; suite_counts = {}; errors = []
    for suite, entries in categories.items():
        names = current[suite]; by_name = {name: i for i, name in enumerate(names)}
        for row in entries:
            name = row['name']; candidates = [n for n in original[suite] if name == n or name.startswith(n+'_')]
            matched = max(candidates, key=len) if candidates else None
            # Official ControlEnv rewrites robot/view pseudo-task filenames to
            # their underlying BDDL, then applies view/init parameters itself.
            effective_name = name.split('_view_')[0] if '_view_' in name and '_initstate_' in name else name
            bddl = root/'libero/libero/bddl_files'/suite/(effective_name+'.bddl')
            record = dict(suite=suite, classification_id=row['id'], suite_index=by_name.get(name),
                          task_name=name, category=row['category'], difficulty=row['difficulty_level'],
                          original_task=matched, original_index=original[suite].index(matched) if matched else None,
                          effective_bddl=str(bddl), bddl_exists=bddl.exists(),
                          bddl_sha256=digest_bytes(bddl.read_bytes()) if bddl.exists() else None)
            rows.append(record)
            if record['suite_index'] is None or matched is None or not bddl.exists():
                errors.append(record)
        suite_counts[suite] = dict(classified=len(entries), registered=len(names),
            categories=dict(Counter(r['category'] for r in entries)),
            difficulties=dict(Counter(str(r['difficulty_level']) for r in entries)))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'task_mapping.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(status='static_audit_complete', repository_root=str(root),
                  source_files={str(p):digest_bytes(Path(p).read_bytes()) for p in [classification,task_map_path,Path(original_task_map)]},
                  total_classified=len(rows), suite_counts=suite_counts, unresolved=errors,
                  official_trials_per_task=1, config_imported=False, gpu_calls=0,
                  qualification='Original mapping is longest exact-name prefix; not an environment equivalence claim')
    save_json(output/'task_audit.json', result)
    return result


def inspect_evaluation_zip(path, maximum_uncompressed=50*1024**3, wrapper_prefix=''):
    """Validate all ZIP members before any extraction or environment import."""
    with zipfile.ZipFile(path) as archive:
        rows = []; seen = set(); total = 0
        for entry in archive.infolist():
            raw_name = entry.filename; raw_member = Path(raw_name)
            if raw_member.is_absolute() or '..' in raw_member.parts or '\\' in raw_name:
                raise ValueError('Unsafe ZIP path: '+raw_name)
            if wrapper_prefix and not raw_name.startswith(wrapper_prefix):
                raise ValueError('ZIP member outside explicitly approved wrapper')
            name = raw_name[len(wrapper_prefix):]; member = Path(name)
            if name in seen or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Duplicate or symlink ZIP member: '+name)
            seen.add(name); total += entry.file_size
            if total > maximum_uncompressed:
                raise ValueError('ZIP exceeds approved uncompressed byte budget')
            if not member.parts or member.parts[0] != 'assets':
                raise ValueError('Unexpected non-evaluation root: '+name)
            rows.append(dict(path=name, archive_path=raw_name, bytes=entry.file_size, compressed_bytes=entry.compress_size,
                             crc32=entry.CRC, directory=entry.is_dir()))
    return dict(status='archive_member_safety_passed', archive=str(Path(path).resolve()),
                wrapper_prefix=wrapper_prefix, members=rows, total_uncompressed_bytes=total, member_count=len(rows))


def extract_evaluation_zip(path, destination_parent, receipt_path, wrapper_prefix=''):
    """Extract only an audited assets root, without replacing existing files."""
    audit = inspect_evaluation_zip(path, wrapper_prefix=wrapper_prefix)
    parent = Path(destination_parent).resolve(); target = parent/'assets'
    if target.exists() or target.is_symlink():
        raise FileExistsError('Existing assets root will not be replaced')
    staging = Path(tempfile.mkdtemp(prefix='.evaluation-assets-', dir=str(parent)))
    try:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                output = staging/entry.filename[len(wrapper_prefix):]
                if entry.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    # Reading to EOF validates each member's ZIP CRC.
                    with archive.open(entry) as source, output.open('xb') as sink:
                        shutil.copyfileobj(source, sink, length=1024*1024)
        if target.exists() or target.is_symlink():
            raise FileExistsError('Assets target appeared during extraction')
        (staging/'assets').rename(target); staging.rmdir()
        audit.update(status='extracted_crc_verified', target=str(target),
                     original_install_or_global_config_modified=False)
        save_json(receipt_path, audit)
        return audit
    except Exception:
        # Material partial output is retained for diagnosis; never recursively
        # remove a path supplied by a user or an unresolved environment variable.
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', action='append', default=[])
    p.add_argument('--search'); p.add_argument('--author'); p.add_argument('--output', default=str(DEFAULT_OUTPUT))
    p.add_argument('--metadata', action='store_true', help='Perform bounded metadata network reads, never model downloads')
    args = p.parse_args()
    if not args.metadata:
        print(json.dumps(dict(dry_run=True, gpu_calls=0, checkpoint_downloads=0, **vars(args))))
        return
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if args.search:
        models, receipt = hf_search(args.search, args.author)
        save_json(output/f'search_{args.search.replace("/", "_").replace(" ", "_")}.json', dict(models=models, receipt=receipt))
        print(json.dumps([dict(id=r['id'], sha=r.get('sha')) for r in models], indent=2))
    for repository in args.repository:
        try:
            receipt = inventory(repository, output)
            print(json.dumps(dict(repository=repository, revision=receipt['revision'], files=len(receipt['files']),
                                  total_file_bytes=sum(x['size'] or 0 for x in receipt['files']),
                                  downloaded_checkpoint_bytes=0), indent=2))
        except Exception as error:
            save_json(output/(repository.replace('/', '__')+'_error.json'), dict(repository=repository, error=str(error)))
            print(json.dumps(dict(repository=repository, error=str(error))))


if __name__ == '__main__':
    main()
