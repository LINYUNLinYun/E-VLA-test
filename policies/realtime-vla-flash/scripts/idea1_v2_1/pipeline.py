"""Explicit CPU stages. No environment or model imports, calls, or rollouts."""
from dataclasses import asdict
import json
from pathlib import Path
from itertools import islice
import numpy as np
from ..idea1_v2 import data as old_data
from ..idea1_v2.core import Kalman as OldKalman, sha, write_json
from .filter import Parameters, CommandKalman, candidate_manifest, digest, fit_shapes, predict_episode
from .diagnostics import decompose, phase_shift_oracle, scene_annotations

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO/'results/idea1_v2_1_multimodel_kalman'


def output_root(path):
    path = Path(path).resolve()
    if path.name != 'e4_kalman' or RESULTS.resolve() not in path.parents or path.parent == RESULTS.resolve():
        raise ValueError('Output must be results/idea1_v2_1_multimodel_kalman/<attempt>/e4_kalman')
    return path


def code_hashes():
    paths = sorted(Path(__file__).parent.glob('*.py')) + sorted((REPO/'scripts/idea1_v2').glob('*.py'))
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def load_json(path):
    return json.loads(Path(path).read_text())


def freeze(root):
    manifest = candidate_manifest()
    manifest['code_hashes'] = code_hashes()
    manifest['sha256'] = digest(manifest)
    write_json(root/'candidate_manifest.json', manifest, exclusive=True)
    return manifest


def frozen(root):
    manifest = load_json(root/'candidate_manifest.json')
    expected = manifest.pop('sha256')
    if digest(manifest) != expected or manifest['code_hashes'] != code_hashes():
        raise ValueError('Frozen manifest or code changed; use a new attempt')
    manifest['sha256'] = expected
    return manifest


def source_groups(source):
    # Reuse the read-only old loader: 6979 unique IDs, 200 episodes, prefix,
    # telemetry clocks, checkpoint/skip, input and deterministic noise hashes.
    groups, sources = old_data.load(source.resolve())
    for records in groups.values():
        path = Path(records[0]['full_infer_path'])
        with path.open() as stream:
            logged = {int(r['infer_id']): r for r in map(json.loads, stream) if r}
        for record in records:
            event = logged[int(record['replan_id'])]
            if not np.allclose(event['chunk_actions'], record['full_actions'], atol=1e-6, rtol=0):
                raise ValueError('Full prediction differs from original inference log')
            t = int(record['environment_action_step'])
            if record['semantic_phase'] != record['_telemetry'][t]['semantic_phase']:
                raise ValueError('Replan phase differs from telemetry')
    sources['v2_1_checks'] = dict(full_chunk_infer_atol=1e-6, full_chunk_infer_rtol=0,
                                 shadow_v14_action_capture='hashed original predictions.jsonl event',
                                 replan_phase_matches_telemetry=True)
    return groups, sources


def sample_check(source):
    """Read only the first two source records per suite; no persistent outputs.

    This is an integration smoke, explicitly not the 6979-point preflight or
    frozen v2 reproduction gate. It checks actual serialized schema and slicing.
    """
    from ..idea1_v2.core import SUITES, noise
    checked = []
    source = source.resolve()
    for suite in SUITES:
        prediction_path = source/'shadow_supplement_v14/round2_20261011/V14_A03'/suite/'predictions.jsonl'
        with prediction_path.open() as stream:
            records = [json.loads(line) for line in islice(stream, 2)]
        if len(records) != 2 or any((r['task_id'], r['episode_idx']) != (0, 0) for r in records):
            raise ValueError('Unexpected sample ordering')
        cutoff = max(int(r['environment_action_step']) for r in records)
        infer_path = Path(records[0]['full_infer_path'])
        telemetry_path = infer_path.with_name('telemetry.jsonl')
        with infer_path.open() as stream:
            events = {r['infer_id']: r for r in [json.loads(line) for line in islice(stream, 2)]}
        with telemetry_path.open() as stream:
            telemetry = [json.loads(line) for line in islice(stream, cutoff+6)]
        actions = np.asarray([r['action'] for r in telemetry[1:]], dtype=float)
        for record in records:
            t = int(record['environment_action_step'])
            for name in ('full_actions', 'variant_actions'):
                a = np.asarray(record[name])
                if a.shape != (50, 7) or not np.isfinite(a).all():
                    raise ValueError('Invalid source chunk')
            if not np.allclose(events[record['replan_id']]['chunk_actions'], record['full_actions'], atol=1e-6, rtol=0):
                raise ValueError('Full sample differs from infer event')
            if not np.allclose(actions[t:t+5], np.asarray(record['full_actions'])[:5], atol=1e-5, rtol=1e-4):
                raise ValueError('Source sample executed prefix mismatch')
            if [r['step'] for r in telemetry] != list(range(len(telemetry))):
                raise ValueError('Telemetry sample step gap')
            if not np.allclose(np.diff([r['sim_time_s'] for r in telemetry]), .05, atol=1e-8, rtol=0):
                raise ValueError('Telemetry sample period mismatch')
            with np.load(record['full_input_path'], allow_pickle=False) as cache:
                if not np.array_equal(cache['noise'], noise(record['episode_idx'], record['replan_id'])):
                    raise ValueError('Sample noise differs from seed rule')
            record.update(_actions=actions, _telemetry=telemetry, _prefix=5)
            scene_annotations(record)
        predictions = list(predict_episode(records, Parameters(.01, .01), np.eye(3), np.eye(3)))
        if any(mean.shape != (50, 3) or cov.shape != (50, 3, 3) for _, mean, cov, _ in predictions):
            raise ValueError('Sample filter output shape mismatch')
        checked.append(dict(suite=suite, points=2, executed_history_counts=[r['environment_action_step'] for r in records],
                            source_prediction_sha256=sha(prediction_path)))
    return dict(status='read_only_sample_schema_passed', points=8, suites=checked, persistent_outputs=0,
                model_calls=0, environment_actions=0, limitation='Sample only; not full source validation or v2 reproduction')


def old_predictions(records, params):
    k, last = OldKalman(params['q0'], params['r0'], params['dt']), 0
    for rec in records:
        t = int(rec['environment_action_step'])
        for a in rec['_actions'][last:t]:
            k.observe(a)
        last = t
        mean, cov = k.forecast()
        yield rec, mean, cov


def reproduce(root, source, v2):
    manifest = frozen(root)
    groups, sources = source_groups(source)
    params_path = v2/'protocol/kalman_parameters.json'
    params = load_json(params_path)
    predictions_path = v2/'kalman/predictions.npz'
    metadata_path = v2/'kalman/records.jsonl'
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line]
    with np.load(predictions_path, allow_pickle=False) as archive:
        stored = {name: archive[name] for name in ('raw_mean', 'covariance')}
    if len(metadata) != 6979 or stored['raw_mean'].shape != (6979, 50, 3):
        raise ValueError('Old prediction coverage/shape mismatch')
    index, maximum = 0, 0.
    for key, records in sorted(groups.items()):
        for rec, mean, cov in old_predictions(records, params):
            m = metadata[index]
            if [m['suite'], m['task'], m['init'], m['replan']] != [*key, rec['replan_id']]:
                raise ValueError('Old prediction identity mismatch')
            for actual, expected in ((mean, stored['raw_mean'][index]), (cov, stored['covariance'][index])):
                if not np.allclose(actual, expected, atol=manifest['reproduction_atol'], rtol=manifest['reproduction_rtol']):
                    raise ValueError('v2 reproduction failed at '+str(index))
                maximum = max(maximum, float(np.max(np.abs(actual-expected))))
            index += 1
    if index != 6979:
        raise ValueError('Incomplete reproduction')
    # Pointwise arrays imply prefix and offset identity; also freeze the original
    # derived tables so later reporting can independently recompute them.
    paths = [params_path, predictions_path, metadata_path, v2/'analysis/kalman_prefixes.csv', v2/'analysis/kalman_offsets.csv']
    receipt = dict(status='reproduced', points=index, max_array_difference=maximum,
                   old_parameters=params, manifest_sha256=manifest['sha256'], source=str(source.resolve()),
                   source_manifest_sha256=digest(sources), sources=sources,
                   v2_files={str(p.resolve()): sha(p) for p in paths},
                   model_calls=0, environment_actions=0)
    write_json(root/'reproduction.json', receipt, exclusive=True)
    return {k: v for k, v in receipt.items() if k != 'sources'}


def verified_groups(root, source):
    manifest = frozen(root)
    receipt = load_json(root/'reproduction.json')
    if receipt['status'] != 'reproduced' or receipt['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Successful v2 reproduction is required')
    if str(source.resolve()) != receipt['source']:
        raise ValueError('Source path changed')
    for path, expected in receipt['v2_files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen v2 file changed: '+path)
    groups, sources = source_groups(source)
    if digest(sources) != receipt['source_manifest_sha256']:
        raise ValueError('Source provenance changed')
    return groups, receipt


def equal_mean(values):
    """replan -> episode -> task -> suite; missing cells never become zero."""
    suites = []
    for suite in sorted({k[0] for k in values}):
        tasks = []
        for task in sorted({k[1] for k in values if k[0] == suite}):
            tasks.append(np.mean([np.mean(v) for k, v in values.items() if k[:2] == (suite, task)]))
        suites.append(np.mean(tasks))
    return float(np.mean(suites)) if suites else None


def score_predictions(groups, predictions):
    means, l2_tail, lateral_tail, by_suite = {}, [], [], {}
    for key, rec, mean in predictions:
        n = min(5, int(rec['_prefix']))
        d = decompose(mean, np.asarray(rec['full_actions'])[:len(mean)])
        value = float(d['l2_mm'][:n].mean())
        means.setdefault(key, []).append(value)
        l2_tail.extend(d['l2_mm'][:n])
        lateral_tail.extend(d['lateral_mm'][:n][np.isfinite(d['lateral_mm'][:n])])
    if not l2_tail or not lateral_tail:
        raise ValueError('Insufficient defined prefix/tangent coverage')
    for suite in sorted({k[0] for k in groups}):
        by_suite[suite] = equal_mean({k: v for k, v in means.items() if k[0] == suite})
    return dict(prefix_equal_mean_mm=equal_mean(means), sample_l2_p95_mm=float(np.percentile(l2_tail, 95)),
                sample_lateral_p95_mm=float(np.percentile(lateral_tail, 95)), suites=by_suite,
                points=sum(len(v) for v in means.values()), valid_lateral_samples=len(lateral_tail))


def candidate_score(training, validation, parameters):
    q, r, shapes = fit_shapes(training, parameters.shape)
    def predictions():
        for key, records in sorted(validation.items()):
            for rec, mean, _, _ in predict_episode(records, parameters, q, r, horizon=5):
                yield key, rec, mean
    return score_predictions(validation, predictions()), shapes


def ranking(result):
    p = result['parameters']
    complexity = int(p['g'] != 1)+int(p['beta'] != 1)+int(p['shape'] != 'isotropic')
    return (not result['tail_pass'], result['score_mm'], result['lateral_p95_mm'], complexity, digest(p))


def tune(root, source, resume=False):
    groups, receipt = verified_groups(root, source)
    manifest = frozen(root)
    winner_path = root/'winner.json'
    if winner_path.exists():
        if not resume:
            raise FileExistsError('Winner exists; use --resume to inspect the validated completed stage')
        previous = load_json(winner_path)
        expected = previous.pop('sha256')
        if digest(previous) != expected or previous['manifest_sha256'] != receipt['manifest_sha256']:
            raise ValueError('Completed winner/provenance changed')
        previous['sha256'] = expected
        return previous
    calibration = {k: v for k, v in groups.items() if k[1] == 0}
    if {k[2] for k in calibration} != set(range(5)):
        raise ValueError('Expected five official calibration inits')
    score_dir = root/'calibration'
    if score_dir.exists() and not resume:
        raise FileExistsError('Calibration exists; explicitly use --resume')
    score_dir.mkdir(parents=True, exist_ok=True)
    baseline = {}
    for init in range(5):
        validation = {k: v for k, v in calibration.items() if k[2] == init}
        baseline[init] = score_predictions(validation, ((key, rec, mean) for key, records in sorted(validation.items())
                                                      for rec, mean, _ in old_predictions(records, receipt['old_parameters'])))
    write_json(score_dir/'old_reference.json', baseline)

    def evaluate(parameters):
        identity = digest(asdict(parameters))
        path = score_dir/(identity+'.json')
        if resume and path.exists():
            saved = load_json(path)
            if saved['parameters'] != asdict(parameters) or saved['manifest_sha256'] != receipt['manifest_sha256']:
                raise ValueError('Invalid resume candidate')
            return saved
        folds = []
        try:
            for init in range(5):
                training = {k: v for k, v in calibration.items() if k[2] != init}
                validation = {k: v for k, v in calibration.items() if k[2] == init}
                score, shapes = candidate_score(training, validation, parameters)
                b = baseline[init]
                passed = all(score[field] <= 1.05*b[field]+1e-12 for field in ('sample_l2_p95_mm', 'sample_lateral_p95_mm'))
                folds.append(dict(held_out_init=init, score=score, shapes=shapes, tail_pass=passed))
            result = dict(status='scored', parameters=asdict(parameters), folds=folds,
                          tail_pass=all(f['tail_pass'] for f in folds),
                          score_mm=float(np.mean([f['score']['prefix_equal_mean_mm'] for f in folds])),
                          lateral_p95_mm=float(np.mean([f['score']['sample_lateral_p95_mm'] for f in folds])))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            result = dict(status='failed', parameters=asdict(parameters), folds=folds, failure=str(error),
                          tail_pass=False, score_mm=None, lateral_p95_mm=None)
        result['manifest_sha256'] = receipt['manifest_sha256']
        write_json(path, result, exclusive=True)
        write_json(root/'progress.json', dict(stage='calibration', completed_candidates=len(list(score_dir.glob('*.json')))-1,
                                             latest_candidate=identity, latest_status=result['status'], candidate_budget=208))
        return result

    first = [evaluate(Parameters(**p)) for p in manifest['stage1']]
    eligible = sorted([r for r in first if r['status'] == 'scored'], key=ranking)[:3]
    if len(eligible) != 3:
        raise ValueError('Fewer than three valid stage1 parents; stop')
    # Freeze the data-dependent parents before stage2. The expansion rule was
    # frozen before any scores; exact duplicate hypotheses are scored once.
    parents = [r['parameters'] for r in eligible]
    parent_path = root/'stage2_parents.json'
    if resume and parent_path.exists():
        if load_json(parent_path)['parents'] != parents:
            raise ValueError('Stage2 parents changed during resume')
    else:
        write_json(parent_path, dict(parents=parents, manifest_sha256=receipt['manifest_sha256']), exclusive=True)
    all_results = {digest(r['parameters']): r for r in first}
    for parent in parents:
        for g in (.5, 1., 1.5):
            for beta in (.7, .9, 1.):
                for shape in manifest['stage2']['shapes']:
                    p = Parameters(parent['q_u'], parent['q_w'], g, beta, shape)
                    identity = digest(asdict(p))
                    if identity not in all_results:
                        all_results[identity] = evaluate(p)
    if len(all_results) > 211:
        raise ValueError('Candidate budget exceeded')
    qualified = sorted([r for r in all_results.values() if r['status'] == 'scored' and r['tail_pass']], key=ranking)
    reference = float(np.mean([b['prefix_equal_mean_mm'] for b in baseline.values()]))
    winner = qualified[0] if qualified else None
    improved = winner is not None and winner['score_mm'] < reference
    frozen_winner = dict(status='frozen' if improved else 'no_qualified_improvement',
                         parameters=winner['parameters'] if winner else None,
                         calibration_score_mm=winner['score_mm'] if winner else None,
                         old_reference_score_mm=reference, candidate_count=len(all_results),
                         candidate_budget_slots=211, manifest_sha256=receipt['manifest_sha256'],
                         disclosure='Previously inspected reused data; not blind evaluation')
    if winner:
        p = Parameters(**winner['parameters'])
        q, r, shapes = fit_shapes(calibration, p.shape)
        frozen_winner.update(shape_q=q.tolist(), shape_r=r.tolist(), shapes=shapes,
                             boundary_winner=any(v in (1e-4, 100.) for v in (p.q_u, p.q_w)),
                             fold_scores=[f['score'] for f in winner['folds']])
    frozen_winner['sha256'] = digest(frozen_winner)
    write_json(root/'winner.json', frozen_winner, exclusive=True)
    return frozen_winner


def evaluate(root, source):
    groups, receipt = verified_groups(root, source)
    winner = load_json(root/'winner.json')
    expected = winner.pop('sha256')
    if digest(winner) != expected or winner['manifest_sha256'] != receipt['manifest_sha256']:
        raise ValueError('Winner/provenance changed')
    if winner['parameters'] is None:
        raise ValueError('No valid frozen candidate: evaluation unavailable')
    destination = root/'evaluation'
    destination.mkdir(exist_ok=False)
    p = Parameters(**winner['parameters'])
    means, covs, fulls, variants, holds, innovations, metadata = [], [], [], [], [], [], []
    old_means, traces, trajectories = [], [], []
    for key, records in sorted(groups.items()):
        if key[1] == 0:
            continue
        old_means.extend([mean for _, mean, _ in old_predictions(records, receipt['old_parameters'])])
        trace_filter = CommandKalman(p, winner['shape_q'], winner['shape_r'])
        for step, action in enumerate(records[0]['_actions'], 1):
            trace_filter.observe(action)
            traces.append(dict(suite=key[0], task=key[1], init=key[2], step=step,
                               innovation=trace_filter.innovation.tolist(), posterior=trace_filter.x.tolist(),
                               covariance=trace_filter.P.tolist()))
        telemetry = records[0]['_telemetry']
        trajectories.append(dict(suite=key[0], task=key[1], init=key[2],
                                 telemetry_path=str(Path(records[0]['full_infer_path']).with_name('telemetry.jsonl')),
                                 video_candidates=[str(v) for v in Path(records[0]['full_infer_path']).parent.glob('*.mp4')],
                                 eef_xyz_m=[r['eef_pos'] for r in telemetry],
                                 phases=[r.get('semantic_phase', 'unknown') for r in telemetry]))
        for rec, mean, cov, innovation in predict_episode(records, p, winner['shape_q'], winner['shape_r']):
            t = int(rec['environment_action_step'])
            hold = np.tile(np.clip(rec['_actions'][t-1, :3], -1, 1) if t else np.zeros(3), (50, 1))
            means.append(mean); covs.append(cov); fulls.append(np.asarray(rec['full_actions'])[:, :3])
            variants.append(np.asarray(rec['variant_actions'])[:, :3]); holds.append(hold)
            innovations.append(np.full(3, np.nan) if innovation is None else innovation)
            metadata.append(dict(suite=key[0], task=key[1], init=key[2], replan=rec['replan_id'], step=t,
                                 valid_executed_prefix=rec['_prefix'], history_count=t,
                                 clip_fraction=float(np.mean(np.abs(mean)>1)), innovation_defined=innovation is not None,
                                 input_sha256=rec['full_input_sha256'], noise_sha256=rec['noise_sha256'],
                                 winner_sha256=expected, scene=scene_annotations(rec)))
    if len(metadata) != 6345:
        raise ValueError('Evaluation coverage mismatch')
    np.savez_compressed(destination/'predictions.npz', raw_mean=means, clipped_mean=np.clip(means, -1, 1),
                        covariance=covs, full=fulls, v14=variants, hold_last=holds, innovation=innovations,
                        old_kalman=old_means)
    write_json(destination/'innovation_trace.json', traces, exclusive=True)
    write_json(destination/'eef_trajectories.json', trajectories, exclusive=True)
    write_json(destination/'records.json', metadata, exclusive=True)
    status = dict(status='predictions_saved_metrics_pending', points=len(metadata), winner_sha256=expected,
                  qualified_improvement=winner['status']=='frozen', source_manifest_sha256=receipt['source_manifest_sha256'],
                  prediction_sha256=sha(destination/'predictions.npz'), records_sha256=sha(destination/'records.json'),
                  model_calls=0, environment_actions=0)
    write_json(destination/'status.json', status, exclusive=True)
    return status


def decomposition(root):
    frozen(root)
    destination = root/'evaluation'
    status = load_json(destination/'status.json')
    for name, field in (('predictions.npz', 'prediction_sha256'), ('records.json', 'records_sha256')):
        if sha(destination/name) != status[field]:
            raise ValueError('Evaluation array/metadata changed')
    with np.load(destination/'predictions.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = load_json(destination/'records.json')
    out = root/'decomposition'
    out.mkdir(exist_ok=False)
    pairs = {'OldKalman-Full': ('old_kalman', 'full'), 'Kalman-Full': ('clipped_mean', 'full'), 'HoldLast-Full': ('hold_last', 'full'),
             'V14-Full': ('v14', 'full'), 'Kalman-V14': ('clipped_mean', 'v14')}
    summaries = []
    for pair, (a, b) in pairs.items():
        details, oracles = [], []
        for i in range(len(metadata)):
            details.append(decompose(arrays[a][i], arrays[b][i]))
            oracles.append(phase_shift_oracle(arrays[a][i], arrays[b][i]))
        np.savez_compressed(out/(pair+'.npz'), **{field: np.asarray([d[field] for d in details]) for field in details[0]})
        write_json(out/(pair+'_time_oracle.json'), oracles, exclusive=True)
        # Per-offset and fixed intervals. Future blocks are explicitly diagnostic.
        windows = [(str(h+1), h, h+1) for h in range(50)]+[('executed_1_5', 0, 5), ('diagnostic_6_12', 5, 12), ('diagnostic_13_50', 12, 50)]
        for label, lo, hi in windows:
            for field in ('l2_mm', 'along_mm', 'lateral_mm', 'magnitude_difference_mm', 'cosine'):
                values, samples = {}, []
                for m, d in zip(metadata, details):
                    end = min(hi, m['valid_executed_prefix']) if label == 'executed_1_5' else hi
                    sample = d[field][lo:end]
                    sample = sample[np.isfinite(sample)]
                    if len(sample):
                        values.setdefault((m['suite'], m['task'], m['init']), []).append(float(sample.mean()))
                        samples.extend(sample)
                summaries.append(dict(pair=pair, window=label, metric=field, equal_mean=equal_mean(values),
                                      sample_p50=float(np.percentile(samples, 50)) if samples else None,
                                      sample_p95=float(np.percentile(samples, 95)) if samples else None,
                                      valid_samples=len(samples), covered_episodes=len(values),
                                      source_prediction_sha256=status['prediction_sha256'], winner_sha256=status['winner_sha256']))
    write_json(out/'offset_interval_summary.json', summaries, exclusive=True)
    from .reporting import finish
    return finish(root, metadata, arrays, summaries, status)
