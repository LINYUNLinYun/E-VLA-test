"""Revision-pinned acquisition plans with bounded, resumable SHA-verified fetches."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from .assets import ROOT, save_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def payload_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_target(root, relative):
    root = Path(root).resolve(); p = (root/relative).resolve()
    if p == root or root not in p.parents:
        raise ValueError('Artifact escapes acquisition root')
    return p


def build(receipt_paths, target_root):
    files, repositories = [], []
    for path in receipt_paths:
        path = Path(path).resolve(); r = json.loads(path.read_text())
        model = json.loads((path.parent/'hub_model.json').read_text())
        small_hashes = {f['path']: f['sha256'] for f in r['small_file_receipts'] if f.get('sha256')}
        repositories.append(dict(repository=r['repository'], revision=r['revision'], inventory_path=str(path),
                                 inventory_sha256=sha(path), license=(model.get('cardData') or {}).get('license'),
                                 qualification='Target-domain checkpoint; not matched to a LIBERO-trained draft without additional proof'))
        for item in r['files']:
            if item['path'] == '.gitattributes':
                continue
            expected = item.get('sha256') or small_hashes.get(item['path'])
            if not expected or len(expected) != 64 or item['size'] is None:
                raise ValueError('Size/SHA256 missing; fetch bounded metadata before planning: '+item['path'])
            relative = r['repository']+'/'+item['path']
            files.append(dict(item, sha256=expected, relative_target=relative,
                              target=str(safe_target(target_root, relative))))
    plan = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(), repositories=repositories,
                target_root=str(Path(target_root).resolve()), files=files,
                total_bytes=sum(f['size'] for f in files), gpu_calls=0, training=False,
                comparison_rule='Native positive controls are labelled as target-trained; no matched ablation against unrelated LIBERO draft weights')
    plan['manifest_hash'] = payload_digest(plan)
    return plan


def load(path):
    plan = json.loads(Path(path).read_text())
    if payload_digest({k: v for k, v in plan.items() if k != 'manifest_hash'}) != plan.get('manifest_hash'):
        raise ValueError('Acquisition manifest changed')
    for f in plan['files']:
        if str(safe_target(plan['target_root'], f['relative_target'])) != f['target']:
            raise ValueError('Artifact target changed')
    return plan


def download_file(item, retries=4):
    target = Path(item['target']); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != item['size'] or sha(target) != item['sha256']:
            raise FileExistsError('Existing final artifact differs; never overwrite: '+str(target))
        return dict(path=str(target), status='verified_existing', size=item['size'], sha256=item['sha256'])
    partial = target.with_name(target.name+'.part')
    identity_path = target.with_name(target.name+'.download_identity.json')
    identity = {k: item[k] for k in ('repository', 'revision', 'path', 'size', 'sha256', 'url')}
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Partial artifact identity differs')
    else:
        if partial.exists():
            raise ValueError('Unidentified partial file; do not reuse')
        save_json(identity_path, identity)
    started, errors = time.time(), []
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item['size']:
            raise ValueError('Partial exceeds pinned artifact size')
        if offset == item['size']:
            break
        headers = {'User-Agent': 'benchmark-pinned-acquisition/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(item['url'], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    raise RuntimeError('Server ignored Range; preserve partial and stop rather than overwrite')
                if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Range response offset differs')
                length = response.headers.get('Content-Length')
                if length and int(length) > item['size']-offset:
                    raise ValueError('Server response exceeds pinned remaining size')
                with partial.open('ab' if offset else 'xb') as stream:
                    count = offset
                    while True:
                        block = response.read(8*1024*1024)
                        if not block:
                            break
                        count += len(block)
                        if count > item['size']:
                            raise ValueError('Received bytes exceed pinned artifact size')
                        stream.write(block)
            if partial.stat().st_size == item['size']:
                break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as error:
            errors.append(str(error))
            if attempt+1 >= retries:
                raise
            time.sleep(min(5, attempt+1))
    if not partial.exists() or partial.stat().st_size != item['size']:
        raise IOError('Download incomplete; resumable partial preserved')
    actual = sha(partial)
    if actual != item['sha256']:
        raise ValueError('SHA256 mismatch; partial preserved for inspection')
    if target.exists():
        raise FileExistsError('Final target appeared; refusing to replace')
    partial.rename(target)
    return dict(path=str(target), status='downloaded_verified', size=item['size'], sha256=actual,
                elapsed_s=time.time()-started, transient_errors=errors)


def execute(plan, receipt_path, allowed_repositories):
    allowed = set(allowed_repositories)
    if {r['repository'] for r in plan['repositories']} - allowed:
        raise ValueError('Explicit download grant does not cover every repository')
    receipt = dict(status='running', manifest_hash=plan['manifest_hash'], started_utc=datetime.now(timezone.utc).isoformat(),
                   repositories=plan['repositories'], files=[], checkpoint_conversion=False, gpu_used=False)
    destination = Path(receipt_path)
    def update():
        temp = destination.with_name(destination.name+'.temporary')
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False)); temp.replace(destination)
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get('manifest_hash') != plan['manifest_hash']:
            raise ValueError('Different acquisition receipt already exists')
    update()
    try:
        for item in plan['files']:
            print(f'ACQUIRE {item["repository"]}/{item["path"]} {item["size"]} bytes', flush=True)
            receipt['files'].append(download_file(item)); update()
        receipt.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                       verified_bytes=sum(f['size'] for f in receipt['files']))
    except Exception as error:
        receipt.update(status='failed_partial_preserved', error=str(error)); raise
    finally:
        update()
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', action='append', default=[]); p.add_argument('--manifest', required=True)
    p.add_argument('--target-root', default=str(ROOT/'shared/checkpoints/e4_supplement'))
    p.add_argument('--execute', action='store_true'); p.add_argument('--allow-repository', action='append', default=[])
    p.add_argument('--receipt')
    args = p.parse_args()
    if args.inventory:
        plan = build(args.inventory, args.target_root); save_json(args.manifest, plan)
    else:
        plan = load(args.manifest)
    print(json.dumps(dict(manifest_hash=plan['manifest_hash'], total_bytes=plan['total_bytes'], files=len(plan['files']), execute=args.execute)))
    if args.execute:
        if not args.receipt or not args.allow_repository:
            p.error('Execution requires a receipt and explicit parent-granted repository allow-list')
        execute(plan, args.receipt, args.allow_repository)


if __name__ == '__main__':
    main()
