"""CPU-only count observer and frozen eight-control auditor invocation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/nfs/embodied/benchmark')
REPORT = Path(__file__).resolve().parent
STUDY = ROOT / 'results/ease_insights_20261103/ae_complement_beam_v2/calibration'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    lock = (REPORT / 'calibration_observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(STUDY / 'manifest.json')
    auditor = REPORT / 'calibration_audit.py'
    if sha(auditor) != manifest['source_files'][str(auditor.relative_to(ROOT))]:
        raise ValueError('Frozen calibration auditor changed')
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    source_sha256=sha(__file__), time=time.time(), CPU_only=True, GPU_jobs_launched=False)
    (REPORT / 'calibration_observer.json').write_text(json.dumps(identity, indent=2) + '\n')
    while True:
        counts = {cid: len(list((STUDY / cid / 'episodes').glob('**/episode_receipt.json'))) for cid in manifest['planned_conditions']}
        markers = len(list(STUDY.glob('*/calibration_shard*of02_complete.json')))
        attention = any(read(p)['phase'] == 'attention_holding' for p in (ROOT / 'results/ease_insights_20261103/resource_pool_v4/sessions').glob('*/status.json'))
        value = dict(time=time.time(), observer=identity, completed=counts, markers=markers, expected_known_controls=8, attention=attention,
                     independent_quality_cases=0, all_complete=sum(counts.values()) == 8 and markers == 4)
        temporary = REPORT / 'calibration_progress.tmp'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(REPORT / 'calibration_progress.json')
        if attention: raise RuntimeError('Calibration needs technical attention; no retry or new scientific launch')
        if value['all_complete']: break
        time.sleep(2)
    with (REPORT / 'calibration_acceptance.log').open('a') as log:
        subprocess.run([str(ROOT / 'policies/realtime-vla-flash/.venv/bin/python'), '-B', str(auditor)], cwd=ROOT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    accepted = read(REPORT / 'calibration_acceptance.json')
    print(json.dumps(dict(status=accepted['status'], all_eight_controls_bitwise_identical=accepted['all_eight_controls_bitwise_identical'],
                          manifest_hash=manifest['manifest_hash'])), flush=True)


if __name__ == '__main__': main()
