"""Apply the frozen Full health gate only after all40 cases are accepted."""
import json
from pathlib import Path
from .common import SUITES, HEALTH_RULE, atomic_json, read_manifest, cases, sha
from .prepare import OUTPUT, REPORT
from .worker import completed, episode_dir


def decide(counts):
    if set(counts)!=set(SUITES) or any(type(v) is not int or not 0<=v<=10 for v in counts.values()):
        raise ValueError('Need exactly four ten-task suite counts')
    return sum(counts.values())>=HEALTH_RULE['minimum_total_successes'] and all(v>=HEALTH_RULE['minimum_suite_successes'] for v in counts.values())


def analyze():
    manifest=read_manifest(OUTPUT/'manifest.json',verify_sources=True)
    counts={s:0 for s in SUITES};evidence=[]
    for scene in cases(manifest):
        if not completed(OUTPUT,'full_d10',scene,manifest['manifest_hash']):
            raise ValueError('Calibration incomplete; do not evaluate the health gate')
        path=episode_dir(OUTPUT,'full_d10',scene)/'episode_receipt.json'
        row=json.loads(path.read_text())
        counts[scene['suite']]+=int(row['success'])
        evidence.append(dict(case=scene,receipt=str(path),sha256=sha(path),success=row['success']))
    value=dict(status='complete',manifest_hash=manifest['manifest_hash'],episodes=40,successes=sum(counts.values()),
        suite_successes=counts,health_rule=HEALTH_RULE,health_passed=decide(counts),evidence=evidence,
        interpretation='Health calibration only. Preserve all tasks and outcomes; no complementary-path or acceleration inference.')
    atomic_json(REPORT/'analysis.json',value)
    return {k:v for k,v in value.items() if k!='evidence'}


if __name__=='__main__':
    print(json.dumps(analyze()))
