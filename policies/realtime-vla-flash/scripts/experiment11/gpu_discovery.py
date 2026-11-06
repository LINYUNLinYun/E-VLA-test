"""Share the five-minute GPU discovery clock across next-round schedulers."""
import fcntl
import json
import time
from scripts.experiment11.protocol import RESULTS
from scripts.experiment10.common import dump_json
from scripts.experiment10.run_pipeline import free_gpus


def throttled_free_gpus():
    clock = RESULTS / 'next_round_gpu_discovery.json'
    with (RESULTS / 'next_round_gpu_discovery.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        latest = 0.0
        for path in list(RESULTS.glob('*status*.json')) + [clock]:
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            latest = max(latest, float(record.get('last_gpu_scan_epoch', 0)))
        now = time.time()
        if now-latest < 300:
            return []
        result = free_gpus()
        dump_json(clock, {'last_gpu_scan_epoch': now, 'available_at_discovery': result})
        return result
