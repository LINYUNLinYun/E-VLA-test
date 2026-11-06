"""CPU-only observer: validate and select only after the full replay matrix."""
import fcntl
import json
import os
from pathlib import Path
import time
from .common import OUTPUT, ROOT, atomic_json, read_manifest, sha


def main():
    manifest = read_manifest(OUTPUT / 'manifest.json', verify_sources=True)
    lock = (OUTPUT / 'completion_driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    started=time.time(), source_sha256=sha(__file__), CPU_only=True, GPU_jobs_launched=False)
    atomic_json(OUTPUT / 'completion_driver.json', identity)
    last = None
    while True:
        count = len(list((OUTPUT / 'joint16_d2/replays').glob('*/replay_receipt.json')))
        markers = len(list((OUTPUT / 'joint16_d2').glob('screen_shard*of02_complete.json')))
        holders = [json.loads(p.read_text()) for p in (ROOT / 'results/ease_insights_20261103/resource_pool_v4/sessions').glob('*/status.json')]
        attention = any(h['phase'] == 'attention_holding' for h in holders)
        value = dict(time=time.time(), manifest_hash=manifest['manifest_hash'], completed_snapshots=count,
                     expected_snapshots=110, completed_shards=markers, expected_shards=2,
                     physical_episodes=0, attention=attention, all_complete=count == 110 and markers == 2,
                     proxy_errors_read_for_selection=False, observer=identity)
        atomic_json(OUTPUT / 'screen_progress.json', value)
        if (count // 10, markers, attention) != last:
            print(json.dumps({k: v for k, v in value.items() if k != 'observer'}), flush=True)
            last = (count // 10, markers, attention)
        if attention:
            raise RuntimeError('Technical attention; preserve completed replay records and do not select')
        if value['all_complete']:
            break
        time.sleep(2)
    from .analyze import analyze
    selected = analyze()
    report = ROOT / 'reports/ease_insights_20261103/ae_complement_beam_v1/screen_acceptance.json'
    result = dict(status='pass', time=time.time(), manifest_hash=manifest['manifest_hash'], physical_episodes=0,
                  replay_snapshots=110, repeated_noise_inputs=330, joint_mask_samples=50490, dense_reference_samples=330,
                  proxy_selection_status=selected['selection']['status'], selected=selected['selection']['selected'],
                  artifacts={str(p): sha(p) for p in [OUTPUT / 'manifest.json', OUTPUT / 'analysis/analysis.json', OUTPUT / 'analysis/selection.json']},
                  goal_complete=False, quality_evidence=False, confirmation_launched=False,
                  next='Freeze a separate closed-loop discovery only for a usable proxy candidate; otherwise preserve the bounded negative screen and continue research')
    atomic_json(report, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
