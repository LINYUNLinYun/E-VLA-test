"""Bounded CPU-only independent video decoding; never mutates scientific data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from common import atomic_json, digest, exclusive, planned, sha

SCHEMA = 1
TIMEOUT = 900


def read(path):
    return json.loads(Path(path).read_text())


def unique_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publish, never replace an existing receipt.
    finally:
        temporary.unlink(missing_ok=True)


def evidence_path(attempt, evidence):
    attempt, evidence = Path(attempt).resolve(), Path(evidence).resolve()
    if evidence.parent != attempt/'media_audits' or not evidence.name:
        raise ValueError('Evidence must be a direct child of attempt/media_audits')
    return attempt, evidence


def tools():
    result = {}
    for name in ('ffmpeg', 'ffprobe'):
        found = shutil.which(name)
        if not found:
            raise ValueError(f'Missing {name}; do not install implicitly')
        binary = Path(found).resolve()
        run = subprocess.run([str(binary), '-version'], capture_output=True, timeout=30, check=True)
        result[name] = dict(path=str(binary), sha256=sha(binary), version_stdout=run.stdout.decode(errors='replace'),
            version_stdout_sha256=hashlib.sha256(run.stdout).hexdigest())
    return result


def prepare(attempt, evidence):
    attempt, evidence = evidence_path(attempt, evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    identity = dict(schema=SCHEMA, kind='independent_cpu_media_audit', attempt=str(attempt),
        manifest_sha256=sha(attempt/'condition_manifest.json'), source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
        tools=tools(), workers=1, threads=1, nice=19, software_decode=True, default_max_items=50, timeout_seconds=TIMEOUT)
    identity['expected_canonical_count'] = len(list(planned(read(attempt/'condition_manifest.json'))))
    unique_write(evidence/'source_manifest.json', identity)
    return identity


def verify_source(attempt, evidence):
    attempt, evidence = evidence_path(attempt, evidence)
    identity = read(evidence/'source_manifest.json')
    accepted = read(evidence/'source_acceptance.json')
    if (identity.get('schema') != SCHEMA or identity.get('kind') != 'independent_cpu_media_audit'
            or identity['attempt'] != str(attempt) or identity['source'] != str(Path(__file__).resolve())
            or identity['source_sha256'] != sha(__file__) or identity['manifest_sha256'] != sha(attempt/'condition_manifest.json')
            or identity['tools'] != tools() or identity['workers'] != 1 or identity['threads'] != 1
            or identity['nice'] != 19 or identity['software_decode'] is not True or identity['timeout_seconds'] != TIMEOUT
            or accepted.get('accepted') is not True or accepted.get('reviewer_model') != 'gpt-6.1-sol'
            or accepted.get('reasoning_effort') != 'medium' or accepted.get('schema') != SCHEMA
            or accepted.get('source_manifest_sha256') != sha(evidence/'source_manifest.json')):
        raise ValueError('Media audit source/tool/independent acceptance changed')
    return identity


def snapshot(attempt):
    raw = (attempt/'episode_ledger.json').read_bytes()
    ledger = json.loads(raw)
    entries = ledger['entries']
    if len({row['key'] for row in entries}) != len(entries):
        raise ValueError('Duplicate canonical ledger key')
    manifest = read(attempt/'condition_manifest.json')
    canonical = {row['key']: row for row in planned(manifest)}
    for row in entries:
        if row['key'] not in canonical or any(row.get(field) != value for field, value in canonical[row['key']].items()):
            raise ValueError('Ledger row differs from canonical scientific protocol')
    return raw, ledger, manifest, canonical


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), sha256=sha(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
        inode=stat.st_ino, device=stat.st_dev)


def binding(attempt, manifest, row):
    if row['status'] != 'valid':
        raise ValueError('Only committed valid episodes may be audited')
    expected = attempt/'episodes'/row['condition']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/row['token']
    directory = Path(row['directory']).resolve()
    if directory != expected or expected.resolve() != expected or not directory.is_relative_to(attempt/'episodes'):
        raise ValueError('Episode path is not its canonical physical directory')
    paths = {name: directory/name for name in ('validated.json', 'result.json', 'video.mp4')}
    if any(path.resolve().parent != directory for path in paths.values()):
        raise ValueError('Media/metadata path escaped canonical directory')
    before = {name: file_identity(path) for name, path in paths.items()}
    validated, result = read(paths['validated.json']), read(paths['result.json'])
    if (before['validated.json']['sha256'] != row['validated_sha256'] or validated.get('episode_key') != row['key']
            or validated.get('valid') is not True or result.get('valid') is not True
            or validated.get('kind') != 'formal' or result.get('kind') != 'formal'
            or result.get('video') != 'video.mp4' or validated.get('video') != 'video.mp4'
            or validated['artifacts'].get('result.json') != before['result.json']['sha256']
            or validated['artifacts'].get('video.mp4') != before['video.mp4']['sha256']):
        raise ValueError('Committed media/metadata hashes or formal identity differ')
    for value in (validated, result):
        if (any(value[field] != row[field] for field in ('model', 'suite', 'task', 'init', 'success'))
                or Path(value['directory']).resolve() != directory or value['actions'] != result['actions']
                or value['frames'] != result['frames'] or value['frames'] != value['actions']+1):
            raise ValueError('Media metadata action/frame/scientific identity differs')
    if not isinstance(result['frames'], int) or isinstance(result['frames'], bool) or result['frames'] <= 0:
        raise ValueError('Invalid expected frame count')
    fields = ('key', 'condition', 'model', 'suite', 'task', 'init', 'protocol_hash', 'schedule', 'mask', 'token')
    bound = dict((field, row[field]) for field in fields)
    bound.update(manifest_sha256=sha(attempt/'condition_manifest.json'), canonical_manifest_hash=manifest['manifest_hash'],
        directory=str(directory), validated_sha256=row['validated_sha256'], result_sha256=before['result.json']['sha256'],
        video_sha256=before['video.mp4']['sha256'], actions=result['actions'], expected_frames=result['frames'])
    return bound, before


def command_log(command, output, timeout=TIMEOUT):
    try:
        run = subprocess.run(command, capture_output=True, timeout=timeout)
        stdout, stderr, code = run.stdout, run.stderr, run.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, code = error.stdout or b'', error.stderr or b'', 'timeout'
    paths = {}
    for name, content in (('stdout', stdout), ('stderr', stderr)):
        path = output.with_suffix('.'+name)
        # x mode: logs are immutable, including failures and incomplete attempts.
        with path.open('xb') as stream:
            stream.write(content)
        paths[name] = dict(path=str(path), sha256=sha(path))
    record = dict(command=command, exit_code=code, timeout_seconds=timeout, **paths)
    unique_write(output.with_suffix('.command.json'), record)
    record['command_record'] = dict(path=str(output.with_suffix('.command.json')), sha256=sha(output.with_suffix('.command.json')))
    return record, stdout, stderr


def commands_for(identity, video):
    probe = [identity['tools']['ffprobe']['path'], '-v', 'error', '-threads', '1', '-count_frames',
        '-show_entries', 'stream=codec_type,codec_name,nb_read_frames,width,height,r_frame_rate,duration', '-of', 'json', video]
    decode = [identity['tools']['ffmpeg']['path'], '-nostdin', '-hwaccel', 'none', '-threads', '1', '-xerror',
        '-err_detect', 'explode', '-i', video, '-map', '0:v:0', '-an', '-sn', '-dn', '-threads', '1',
        '-vsync', '0', '-progress', 'pipe:1', '-nostats', '-f', 'null', '-']
    return probe, decode


def verify_receipt_outputs(evidence, identity, old):
    commands = commands_for(identity, str(Path(old['binding']['directory'])/'video.mp4'))
    logs = old['commands']
    if len(logs) != 2:
        raise ValueError('Missing probe/decode evidence')
    for log, command in zip(logs, commands):
        if log['command'] != command or log['exit_code'] != 0 or log['timeout_seconds'] != TIMEOUT:
            raise ValueError('Saved software decode command/exit differs')
        for channel in ('stdout', 'stderr', 'command_record'):
            path = Path(log[channel]['path']).resolve()
            if not path.is_relative_to(evidence/'logs') or sha(path) != log[channel]['sha256']:
                raise ValueError('Preserved decode log changed or escaped evidence directory')
        sealed = read(log['command_record']['path'])
        if sealed != {key:value for key,value in log.items() if key != 'command_record'}:
            raise ValueError('Command receipt differs from original immutable log')
    actual = frame_counts(Path(logs[0]['stdout']['path']).read_bytes(), Path(logs[1]['stdout']['path']).read_bytes())
    if actual != old['binding']['expected_frames'] or old['decoded_frames'] != actual:
        raise ValueError('Preserved output frames differ')


def frame_counts(probe_stdout, decode_stdout):
    streams = json.loads(probe_stdout)['streams']
    if len(streams) != 1 or streams[0].get('codec_type') != 'video':
        raise ValueError('Expected exactly one video stream')
    count = streams[0].get('nb_read_frames', '')
    if not str(count).isdigit() or int(count) <= 0:
        raise ValueError('Probe did not count actual decoded frames')
    progress, final = {}, None
    for line in decode_stdout.decode().splitlines():
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        progress[key] = value.strip()
        if key == 'progress':
            if value.strip() == 'end':
                final = dict(progress)
            progress = {}
    if not final or not final.get('frame', '').isdigit():
        raise ValueError('Decode did not reach a final actual frame count')
    decoded = int(final['frame'])
    if decoded != int(count):
        raise ValueError('Decode and probe frame counts differ')
    return decoded


def audit_one(attempt, evidence, identity, manifest, row):
    bound, before = binding(attempt, manifest, row)
    receipt = evidence/'receipts'/f"{row['key']}.json"
    source_hash = sha(evidence/'source_manifest.json')
    if receipt.exists():
        old = read(receipt)
        if (old.get('passed') is not True or old.get('binding') != bound or old.get('source_manifest_sha256') != source_hash
                or old.get('source_acceptance_sha256') != sha(evidence/'source_acceptance.json')
                or old.get('before') != before or old.get('after') != before):
            raise ValueError('Failed/stale/changed media receipt cannot be resumed')
        verify_receipt_outputs(evidence, identity, old)
        return receipt
    logs = evidence/'logs'/row['key']
    logs.mkdir(parents=True, exist_ok=False)
    record = dict(schema=SCHEMA, passed=False, binding=bound, before=before,
        source_manifest_sha256=source_hash, source_acceptance_sha256=sha(evidence/'source_acceptance.json'), commands=[], started_epoch=time.time())
    try:
        video = str(Path(bound['directory'])/'video.mp4')
        probe, decode = commands_for(identity, video)
        p, po, pe = command_log(probe, logs/'probe')
        record['commands'].append(p)
        d, do, de = command_log(decode, logs/'decode')
        record['commands'].append(d)
        if p['exit_code'] != 0 or d['exit_code'] != 0:
            raise ValueError('Independent probe/decode failed or timed out')
        actual = frame_counts(po, do)
        if actual != bound['expected_frames']:
            raise ValueError('Actual decoded count differs from committed expected frames')
        record.update(decoded_frames=actual, probe=json.loads(po))
        after = {name: file_identity(Path(bound['directory'])/name) for name in before}
        record['after'] = after
        if before != after:
            raise ValueError('Media or metadata changed during independent decoding')
        verify_source(attempt, evidence)
        record['passed'] = True
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        record['ended_epoch'] = time.time()
        unique_write(receipt, record)
        raise
    record['ended_epoch'] = time.time()
    unique_write(receipt, record)
    return receipt


def audit(attempt, evidence, max_items=50, keys=None, final=False):
    if not isinstance(max_items, int) or not 1 <= max_items <= 50:
        raise ValueError('A single bounded batch permits 1..50 videos')
    attempt, evidence = evidence_path(attempt, evidence)
    with exclusive(evidence/'audit.lock'):
        identity = verify_source(attempt, evidence)
        raw, ledger, manifest, canonical = snapshot(attempt)
        if len(canonical) != identity['expected_canonical_count']:
            raise ValueError('Canonical media target count differs from prepared scientific scope')
        batch = evidence/'batches'/uuid.uuid4().hex
        batch.mkdir(parents=True, exist_ok=False)
        snapshot_path = batch/'episode_ledger.snapshot.json'
        with snapshot_path.open('xb') as stream:
            stream.write(raw)
        unique_write(batch/'snapshot_receipt.json', dict(sha256=sha(snapshot_path), source=str(attempt/'episode_ledger.json'),
            source_manifest_sha256=sha(evidence/'source_manifest.json'), read_completed_epoch=time.time()))
        valid = {row['key']: row for row in ledger['entries'] if row['status'] == 'valid'}
        historical = {}
        for path in (evidence/'receipts').glob('*.json'):
            saved = read(path)
            key = saved['binding']['key']
            if path.name != key+'.json' or key in historical or key not in valid:
                raise ValueError('Unknown or duplicate historical media receipt')
            if saved.get('schema') != SCHEMA or saved.get('source_manifest_sha256') != sha(evidence/'source_manifest.json'):
                raise ValueError('Historical receipt source identity changed')
            if saved.get('source_acceptance_sha256') != sha(evidence/'source_acceptance.json'):
                raise ValueError('Historical receipt independent source acceptance changed')
            for field in ('key', 'condition', 'model', 'suite', 'task', 'init', 'protocol_hash', 'schedule', 'mask', 'token', 'validated_sha256'):
                if saved['binding'][field] != valid[key][field]:
                    raise ValueError('Historical receipt has a stale canonical ledger tuple')
            if saved.get('passed') is True:
                if saved.get('before') != saved.get('after'):
                    raise ValueError('Historical receipt did not preserve stable input files')
                verify_receipt_outputs(evidence, identity, saved)
            historical[key] = dict(path=str(path), sha256=sha(path), passed=saved.get('passed') is True)
        requested = list(keys) if keys else (sorted(valid) if final else sorted(set(valid)-set(historical)))
        if len(set(requested)) != len(requested) or any(key not in valid for key in requested):
            raise ValueError('Requested keys are duplicate or not committed valid')
        if final and (keys or set(valid) != set(canonical) or any(row['status'] != 'valid' for row in ledger['entries'])):
            raise ValueError('Cannot finalize partial scientific coverage')
        done, processed = {}, 0
        for key in requested:
            receipt = evidence/'receipts'/f'{key}.json'
            if not receipt.exists() and processed >= max_items:
                continue
            existed = receipt.exists()
            try:
                path = audit_one(attempt, evidence, identity, manifest, valid[key])
            except BaseException as error:
                unique_write(batch/'failure.json', dict(passed=False, key=key, error=f'{type(error).__name__}: {error}',
                    ledger_snapshot=str(snapshot_path), ledger_snapshot_sha256=sha(snapshot_path)))
                raise
            done[key] = dict(path=str(path), sha256=sha(path))
            processed += not existed
        if final and set(done) != set(canonical):
            raise ValueError('Cannot finalize incomplete independent video coverage')
        verify_source(attempt, evidence)
        if final and (attempt/'episode_ledger.json').read_bytes() != raw:
            raise ValueError('Ledger changed while sealing final media coverage')
        historical.update({key:dict(value, passed=True) for key,value in done.items()})
        index = dict(schema=SCHEMA, kind='independent_media_coverage', partial=not final, complete=final,
            independent_final_C_acceptance_required=True, source_manifest_sha256=sha(evidence/'source_manifest.json'),
            source_acceptance_sha256=sha(evidence/'source_acceptance.json'),
            ledger_snapshot=str(snapshot_path), ledger_snapshot_sha256=sha(snapshot_path), manifest_sha256=sha(attempt/'condition_manifest.json'),
            canonical_count=len(canonical), valid_snapshot_count=len(valid), covered_count=sum(row['passed'] for row in historical.values()),
            failed_count=sum(not row['passed'] for row in historical.values()), pending_count=len(canonical)-len(historical),
            remaining_not_passed_count=len(canonical)-sum(row['passed'] for row in historical.values()),
            current_batch_reverified_keys=sorted(done), historical_receipts_not_all_current_rehashed=not final,
            new_decodes=processed, receipts=historical)
        output = batch/'index.json'
        unique_write(output, index)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'audit', 'finalize'))
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--max-items', type=int, default=50)
    parser.add_argument('--key', action='append')
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(args.attempt, args.evidence)
    else:
        os.nice(max(0, 19-os.nice(0)))
        print(audit(args.attempt, args.evidence, args.max_items, args.key, final=args.operation == 'finalize'))


if __name__ == '__main__':
    main()
