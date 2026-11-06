"""Rebuild the proxy metric/ranking with Python sums from original action arrays."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT = Path('/root/nfs/embodied/benchmark')
STUDY = ROOT / 'results/ease_insights_20261103/ae_complement_beam_v1'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(v): return math.fsum(v) / len(v)
def require(v, message):
    if not v: raise ValueError(message)


def main():
    manifest = read(STUDY / 'manifest.json')
    selected = read(STUDY / 'analysis/selection.json')
    reported = read(STUDY / 'analysis/analysis.json')
    accepted = read(ROOT / 'reports/ease_insights_20261103/ae_complement_beam_v1/screen_acceptance.json')
    require(accepted['status'] == 'pass' and reported['complete'] and accepted['replay_snapshots'] == 110, 'Need complete accepted replay')
    for p, pin in accepted['artifacts'].items(): require(sha(p) == pin, 'Accepted replay artifact changed')
    for p, pin in manifest['source_files'].items(): require(sha(ROOT / p) == pin, 'Frozen replay source changed')
    masks = list(itertools.combinations(range(1, 19), 2))
    require(manifest['masks'] == [list(v) for v in masks], 'Not all153 skip-two combinations')
    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    grouped = {(s, t): [] for s in suites for t in range(10)}
    evidence, max_error = [], 0.
    for obs in manifest['observations']:
        p = STUDY / 'joint16_d2/replays' / obs['sample_id']
        receipt = read(p / 'replay_receipt.json')
        for name, pin in receipt['artifacts'].items(): require(sha(p / name) == pin, 'Raw replay artifact changed')
        require(sha(obs['path']) == obs['sha256'], 'Snapshot changed')
        with np.load(p / 'arrays.npz', allow_pickle=False) as arrays:
            actions, stored = arrays['actions'], arrays['nmse']
            require(actions.shape == (3, 154, 50, 7), 'Incomplete decoded arrays')
            values = [[] for _ in masks]
            for si in range(3):
                base = actions[si, 0, :5].ravel().tolist()
                scale = max(mean([v * v for v in base]), 1e-4)
                for mi in range(153):
                    candidate = actions[si, mi + 1, :5].ravel().tolist()
                    value = mean([(a - b) ** 2 for a, b in zip(candidate, base)]) / scale
                    require(math.isfinite(value) and value >= 0, 'Invalid reconstructed error')
                    max_error = max(max_error, abs(value - float(stored[si, mi])))
                    values[mi].append(value)
            grouped[obs['suite'], obs['task']].append([mean(v) for v in values])
        evidence.append(dict(path=str(p / 'replay_receipt.json'), sha256=sha(p / 'replay_receipt.json')))
    require(len(evidence) == 110 and max_error < 1e-12 and all(grouped.values()), 'Incomplete/different proxy metric')
    task = {(s, t): [mean([v[i] for v in grouped[s, t]]) for i in range(153)] for s in suites for t in range(10)}
    suite = {s: [mean([task[s, t][i] for t in range(10)]) for i in range(153)] for s in suites}
    overall = [mean([suite[s][i] for s in suites]) for i in range(153)]
    rule = manifest['proxy_rule']
    eligible = [i for i in range(153) if overall[i] <= rule['overall_task_mean_NMSE_max'] and max(suite[s][i] for s in suites) <= rule['max_suite_mean_NMSE']]
    best, best_key, count = None, None, 0
    for a, b in itertools.combinations(eligible, 2):
        good = {}
        for s in suites:
            for better, worse in [(a, b), (b, a)]:
                delta = [task[s, t][worse] - task[s, t][better] for t in range(10)]
                total = math.fsum(delta)
                if (total / 10 >= rule['minimum_opposite_suite_mean_error_gap'] and sum(v > 0 for v in delta) >= rule['minimum_positive_tasks'] and all(total - v > 0 for v in delta)):
                    good[better, s] = total / 10
        for x, y in itertools.permutations(suites, 2):
            if (a, x) not in good or (b, y) not in good: continue
            da, db = good[a, x], good[b, y]
            count += 1
            key = (-min(da, db), -(da + db), max(overall[a], overall[b]), *masks[a], *masks[b], x, y)
            if best_key is None or key < best_key:
                best_key, best = key, dict(mask_indices=[a, b], masks=[list(masks[a]), list(masks[b])], suites=[x, y], gaps=[da, db])
    decision = reported['decision']
    require(len(eligible) == decision['eligible_masks'] and count == decision['eligible_crossings'], 'Proxy eligible set/count differs')
    if best:
        require(decision['status'] == 'proxy_candidate' and best['masks'] == decision['selected']['masks'] == selected['selection']['selected']['masks'], 'Independent proxy winner differs')
        require(best['suites'] == [d['suite'] for d in decision['selected']['directions']], 'Independent proxy directions differ')
    else: require(decision['status'] == 'no_proxy_candidate', 'Independent no-candidate differs')
    output = Path(__file__).with_name('independent_audit.json')
    value = dict(status='pass', time=time.time(), manifest_hash=manifest['manifest_hash'], observations=110, repeated_noise_inputs=330,
                 physical_episodes=0, joint_mask_samples=50490, maximum_NMSE_reconstruction_error=max_error,
                 eligible_masks=len(eligible), eligible_crossings=count, selected=best,
                 original_selector_or_metric_code_imported=False, quality_evidence=False, evidence=evidence,
                 source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                 selection=dict(path=str(STUDY / 'analysis/selection.json'), sha256=sha(STUDY / 'analysis/selection.json')))
    output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'evidence'}))


if __name__ == '__main__': main()
