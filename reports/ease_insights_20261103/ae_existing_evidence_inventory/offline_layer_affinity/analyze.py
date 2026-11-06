"""Retrospective fixed-input AE sensitivity; no rollout or quality claims.

Recompute the first-five-action error from archived decoded commands. Noise
seeds and phase observations are repeated measurements, not independent tasks.
All 153 layer pairs are reported; no pair is promoted to a scientific cohort.
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/root/nfs/embodied/benchmark')
SOURCE = ROOT / 'policies/realtime-vla-flash/results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/probe_shards'
DEST = Path(__file__).resolve().parent
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
SEEDS = (12345, 23456, 34567)
CHECKPOINT = 'c423a508742f8461fa17d8fb42951837b4419804842ed27a4890a9903042f891'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(values):
    # Average ranks for ties; rank correlation is descriptive, not a p-value.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2)[inverse]


def analyze():
    groups, per_seed, identities, source_files = {}, {}, {}, []
    phases = {}
    for seed in SEEDS:
        shard = SOURCE / ('seed' + str(seed))
        summary_path = shard / 'summary.json'
        summary = json.loads(summary_path.read_text())
        if (summary.get('complete') is not True or summary.get('checkpoint_unchanged') is not True
                or summary.get('checkpoint_sha256') != CHECKPOINT or summary.get('noise_seeds') != [seed]):
            raise ValueError('Historical shard summary differs')
        source_files.append(dict(path=str(summary_path), sha256=sha(summary_path)))
        files = sorted((shard / 'samples').glob('*.npz'))
        if len(files) != summary['sample_count']:
            raise ValueError('Historical shard file count differs')
        identities[seed] = {}
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data['metadata_json'].item()))
                baseline = np.asarray(data['baseline_actions'], dtype=np.float64)
                candidate = np.asarray(data['skip_actions'], dtype=np.float64)
                noise = np.asarray(data['noise'])
            if (meta.get('checkpoint_sha256') != CHECKPOINT or meta.get('noise_seed') != seed
                    or meta.get('vlm_layer_pattern') != 'vlm_full18' or meta.get('denoise_steps') != 10
                    or meta.get('full_baseline_bitwise_parity') is not True
                    or meta.get('repeat_baseline_bitwise_parity') is not True
                    or hashlib.sha256(noise.tobytes()).hexdigest() != meta.get('noise_sha256')
                    or baseline.shape != (50, 7) or candidate.shape != (18, 50, 7)
                    or not np.isfinite(baseline).all() or not np.isfinite(candidate).all()):
                raise ValueError('Saved fixed-input identity/shape/noise contract failed: ' + str(path))
            suite, task, sample = meta['suite'], meta['task_id'], meta['sample_id']
            if suite not in SUITES or type(task) is not int or not 0 <= task < 10:
                raise ValueError('Unexpected historical task identity')
            if sample in identities[seed]:
                raise ValueError('Duplicate historical sample identity')
            identities[seed][sample] = (suite, task, meta['observation_sha256'], meta['semantic_phase'])
            denominator = max(float(np.sum(baseline[:5] ** 2)), 1e-12)
            nmse = np.sum((candidate[:, :5] - baseline[None, :5]) ** 2, axis=(1, 2)) / denominator
            flips = np.mean((candidate[:, :5, 6] > 0) != (baseline[None, :5, 6] > 0), axis=1)
            groups.setdefault((suite, task), []).append((nmse, flips))
            per_seed.setdefault((seed, suite, task), []).append(nmse)
            phases[(suite, sample)] = meta['semantic_phase']
            source_files.append(dict(path=str(path), sha256=sha(path), sample_id=sample, seed=seed))
    if any(identities[seed] != identities[SEEDS[0]] for seed in SEEDS[1:]):
        raise ValueError('Cross-seed observation/task identities differ')
    expected = {(suite, task) for suite in SUITES for task in range(10)}
    if set(groups) != expected:
        raise ValueError('Historical task matrix is incomplete')
    task_scores = {key: np.mean([x[0] for x in values], axis=0) for key, values in groups.items()}
    task_flips = {key: np.mean([x[1] for x in values], axis=0) for key, values in groups.items()}
    suite_scores = {suite: np.mean([task_scores[suite, task] for task in range(10)], axis=0) for suite in SUITES}
    suite_flips = {suite: np.mean([task_flips[suite, task] for task in range(10)], axis=0) for suite in SUITES}
    global_scores = np.mean(list(suite_scores.values()), axis=0)
    layer_order = sorted(range(18), key=lambda i: (global_scores[i], i))
    pairs = []
    for a, b in itertools.combinations(range(18), 2):
        differences = {suite: float(suite_scores[suite][a] - suite_scores[suite][b]) for suite in SUITES}
        per_suite_seed_signs = {}
        for suite in SUITES:
            values = []
            for seed in SEEDS:
                scores = np.mean([np.mean(per_seed[seed, suite, task], axis=0) for task in range(10)], axis=0)
                values.append(int(np.sign(scores[a] - scores[b])))
            per_suite_seed_signs[suite] = values
        negative = [s for s, d in differences.items() if d < 0]
        positive = [s for s, d in differences.items() if d > 0]
        crossing = bool(negative and positive)
        stable_negative = [s for s in negative if per_suite_seed_signs[s] == [-1] * len(SEEDS)]
        stable_positive = [s for s in positive if per_suite_seed_signs[s] == [1] * len(SEEDS)]
        pairs.append(dict(skip_layer_A=a + 1, skip_layer_B=b + 1,
            mean_first5_NMSE_A_minus_B_by_suite=differences,
            signs_by_suite_and_noise_seed=per_suite_seed_signs,
            opposite_mean_directions=crossing,
            opposite_directions_repeated_in_all_three_noise_seeds=bool(stable_negative and stable_positive),
            lower_A_suites=negative, lower_B_suites=positive))
    correlation = [{"suite_A": a, "suite_B": b,
        "spearman_descriptive": float(np.corrcoef(rank(suite_scores[a]), rank(suite_scores[b]))[0, 1])}
        for a, b in itertools.combinations(SUITES, 2)]
    report = dict(schema='historical-AE-offline-layer-affinity-v1',
        scope='Retrospective fixed-input decoded action perturbation; not closed-loop task success or independent confirmation',
        identity=dict(source=str(SOURCE), source_files=len(source_files), archived_NPZ_records=sum(len(v) for v in identities.values()),
            physical_observation_snapshots=len(identities[SEEDS[0]]), tasks=40, tasks_per_suite=10,
            noise_seeds=list(SEEDS), checkpoint_sha256=CHECKPOINT, checkpoint_binary_rehashed=False,
            current_within_suite_outcomes_read=False, GPU_or_model_or_environment_started=False,
            historical_source_and_norm_hash_bound=False, saved_noise_bytes_rehashed=True,
            saved_baseline_parity_flags_checked=True, fresh_model_parity_rerun=False),
        metric='Sum squared deviation of decoded7D commands in first5 positions / max(dense first5 energy,1e-12); task means equally weighted',
        baseline_note='Historical fixed-input Full18 VLM/Full18 AE, D10; first5 is recomputed from saved50-action chunks, not a newR5 rollout',
        layer_rows=[dict(skipped_AE_layer=i+1, global_task_mean_first5_NMSE=float(global_scores[i]),
            suite_task_mean_first5_NMSE={s:float(suite_scores[s][i]) for s in SUITES},
            suite_task_mean_first5_gripper_flip={s:float(suite_flips[s][i]) for s in SUITES}) for i in layer_order],
        suite_rank_correlations=correlation,
        all_layer_pairs=pairs,
        observed_pair_counts=dict(total=len(pairs), opposite_mean_directions=sum(p['opposite_mean_directions'] for p in pairs),
            opposite_directions_repeated_in_all_three_noise_seeds=sum(p['opposite_directions_repeated_in_all_three_noise_seeds'] for p in pairs)),
        historical_L2_L8_pair=next(p for p in pairs if p['skip_layer_A']==2 and p['skip_layer_B']==8),
        per_task=[dict(suite=s, task=t, observations_and_noise_repeats=len(groups[s,t]), first5_NMSE=task_scores[s,t].tolist())
                  for s,t in sorted(task_scores)],
        phase_snapshot_counts={s:{p:sum(s0==s and phase==p for (s0,_),phase in phases.items()) for p in sorted(set(phases.values()))} for s in SUITES},
        limits=['All153pairs inspected retrospectively; no multiplicity-controlled claim or selected winner.',
                'Three noise seeds are repeated inputs, not independent task or heldout replicates.',
                'Lower action error need not mean higher closed-loop success; Full actions are a reference, not ground truth.',
                'Only single-layer AE skips, all atD10. No inference that multi-layer effects add or that a15% speed gain exists.',
                'Historical baseline-trajectory phase selection and suite composition can confound apparent preference.',
                'The archived checkpoint hash and source flags are labels; the binary/source/norm provenance was not recreated.',
                'Current432 study, frozen candidates and heldout cases are unchanged.'])
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'source_files.json').write_text(json.dumps(source_files,indent=2)+'\n')
    report['source_inventory_sha256'] = sha(DEST/'source_files.json')
    report['analysis_script_sha256'] = sha(__file__)
    (DEST/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('identity','observed_pair_counts','suite_rank_correlations','historical_L2_L8_pair')}))


if __name__ == '__main__':
    analyze()
