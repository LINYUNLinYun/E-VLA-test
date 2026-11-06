"""Complete-only proxy selection; masks must subsequently face fresh rollouts."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from .common import *
from .metrics import aggregate
from .worker import validate_record


def choose(tasks, means, overall, suites, rule):
    eligible = [i for i in range(153) if overall[i] <= rule['overall_task_mean_NMSE_max'] and
                max(means[s][i] for s in suites) <= rule['max_suite_mean_NMSE']]
    candidates = []
    for a, b in itertools.combinations(eligible, 2):
        a_suites, b_suites = [], []
        for suite in suites:
            diff = tasks[suite][:, b] - tasks[suite][:, a]
            for signed, target in [(diff, a_suites), (-diff, b_suites)]:
                total = float(signed.sum())
                if (total / 10 >= rule['minimum_opposite_suite_mean_error_gap'] and
                        int((signed > 0).sum()) >= rule['minimum_positive_tasks'] and
                        np.all(total - signed > 0)):
                    target.append((suite, total / 10))
        for (x, dx), (y, dy) in itertools.product(a_suites, b_suites):
            if x != y:
                candidates.append(dict(mask_indices=[a, b], masks=[MASKS[a], MASKS[b]],
                    directions=[dict(suite=x, lower_error_mask=MASKS[a], higher_error_mask=MASKS[b], proxy_gap=dx),
                                dict(suite=y, lower_error_mask=MASKS[b], higher_error_mask=MASKS[a], proxy_gap=dy)],
                    rank=[min(dx, dy), dx + dy, -float(max(overall[a], overall[b]))],
                    tie=[*MASKS[a], *MASKS[b], x, y]))
    candidates.sort(key=lambda r: tuple(-v for v in r['rank']) + tuple(r['tie']))
    return dict(status='proxy_candidate' if candidates else 'no_proxy_candidate',
                selected=candidates[0] if candidates else None, eligible_masks=len(eligible), eligible_crossings=len(candidates),
                quality_evidence=False, independent_validation=False,
                interpretation='Offline same-input command-error heuristic only. A selected pair is not evidence of task-dependent rollout success.')


def analyze():
    manifest = read_manifest(OUTPUT / 'manifest.json', verify_sources=True)
    paths = list((OUTPUT / 'joint16_d2/replays').glob('*/replay_receipt.json'))
    markers = list((OUTPUT / 'joint16_d2').glob('screen_shard*of02_complete.json'))
    require(len(paths) == 110 and len(markers) == 2, 'Await all110 snapshots and both complete services')
    target = OUTPUT / 'analysis/selection.json'
    if target.exists():
        value = json.loads(target.read_text())
        require(value['manifest_hash'] == manifest['manifest_hash'] and value['selection_hash'] == digest({k: v for k, v in value.items() if k != 'selection_hash'}), 'Existing proxy selection changed')
        return value
    evidence, rows = [], []
    for scene in cases(manifest):
        directory = replay_dir(OUTPUT, 'joint16_d2', scene)
        receipt = json.loads((directory / 'replay_receipt.json').read_text())
        require(receipt == validate_record(OUTPUT, 'joint16_d2', scene, manifest), 'Read-only replay reconstruction differs')
        with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
            rows.append(dict(sample_id=scene['sample_id'], suite=scene['suite'], task=scene['task'], phase=scene['phase'], nmse=arrays['nmse'].copy()))
        evidence.append(dict(path=str(directory / 'replay_receipt.json'), sha256=sha(directory / 'replay_receipt.json'), artifacts=receipt['artifacts']))
    services = []
    for run in sorted((OUTPUT / 'joint16_d2/runs').iterdir()):
        values = {name: json.loads((run / name).read_text()) for name in ['identity.json', 'result.json', 'protocol.json', 'gate.json', 'cross_service.json', 'qk_receipt.json']}
        for name in ['identity.json', 'result.json', 'protocol.json', 'gate.json', 'cross_service.json']:
            require(values[name]['manifest_hash'] == manifest['manifest_hash'], 'Mixed replay service identities')
        require(values['result.json']['complete'] is True and values['gate.json']['passed'] and values['cross_service.json']['passed'] and
                values['qk_receipt.json']['passed'] and not values['qk_receipt.json']['unsafe_legacy_kernel_executed'], 'Failed replay service gate')
        protocol = values['protocol.json']
        require(protocol['config'] == CONDITION and protocol['physical_episodes'] == 0 and protocol['mode'] == 'offline_snapshot_replay' and
                protocol['checkpoint_sha256'] == manifest['checkpoints']['pi0']['sha256'] and protocol['normalization_actual_sha256'] == manifest['normalization']['sha256'], 'Replay service asset/config identity differs')
        services.append(dict(path=str(run), artifacts={name: sha(run / name) for name in values}))
    require(len(services) == 2, 'Require exactly two completed replay services')
    tasks, means, overall = aggregate(rows, SUITES)
    decision = choose(tasks, means, overall, SUITES, manifest['proxy_rule'])
    initial = [r for r in rows if r['phase'] == 'initial']
    it, im, io = aggregate(initial, SUITES)
    value = dict(manifest_hash=manifest['manifest_hash'], time=time.time(), complete=True, observations=110, tasks=40,
                 repeated_noise_inputs=330, joint_mask_samples=50490, dense_reference_samples=330, physical_episodes=0,
                 decision=decision, initial_only_sensitivity=choose(it, im, io, SUITES, manifest['proxy_rule']),
                 masks=MASKS, overall_NMSE=overall.tolist(), suite_NMSE={k: v.tolist() for k, v in means.items()},
                 task_NMSE={k: v.tolist() for k, v in tasks.items()}, evidence=evidence, services=services,
                 scope='All registered phases, equal task weights. Initial-only analysis is a separately labeled sensitivity, never used to change the primary selected masks.',
                 heldout_physical_initial_states_used=False, closed_loop_quality_or_speed_claim=False)
    atomic_json(OUTPUT / 'analysis/analysis.json', value)
    frozen = dict(manifest_hash=manifest['manifest_hash'], time=time.time(), selection=decision,
                  analysis=dict(path=str(OUTPUT / 'analysis/analysis.json'), sha256=sha(OUTPUT / 'analysis/analysis.json')),
                  complete_observations=110, physical_episodes=0, closed_loop_candidates_frozen=decision.get('selected'))
    frozen['selection_hash'] = digest(frozen)
    atomic_json(target, frozen)
    return frozen


if __name__ == '__main__':
    print(json.dumps(analyze()))
