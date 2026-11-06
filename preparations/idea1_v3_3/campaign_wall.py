"""Observed campaign wall, distinct from active episode and neural timings."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import time
from preparations.idea1_v3.io import atomic
from .analyze import OUT


def run(groups):
    rows=[r for own in groups.values() for r in own];ends=[(Path(r['_directory'])/'complete.json').stat().st_mtime for r in rows]
    start=min(end-r['wall_s'] for end,r in zip(ends,rows));end=max(ends);elapsed=end-start
    result=dict(time=time.time(),scientific_valid=len(rows),start_UTC=datetime.fromtimestamp(start,timezone.utc).isoformat(),end_UTC=datetime.fromtimestamp(end,timezone.utc).isoformat(),
        elapsed_s=elapsed,observed_aggregate_science_episodes_per_hour=len(rows)*3600/elapsed,
        active_episode_worker_s_sum=sum(r['wall_s'] for r in rows),policy_wall_s_sum=sum(r['policy_wall_s'] for r in rows),
        scope='First valid episode start approximated from complete mtime minus original episode wall, to final scientific complete mtime. Includes intervening engineering400, bank/probe work, calibration waits, worker handoffs, changing GPU1 lease and idle gaps; excludes preparation/model loads before first real episode and final CPU reporting. Aggregate multi-model/multi-GPU study throughput, not a configuration speedup or official benchmark.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic(OUT/'reports/campaign_wall_receipt.json',result);return result

