"""Posthoc late-Fast compensation counterfactuals, within the 1000-call limit.

Run only after the formal evaluator releases the selected GPU. No quality
rollouts or frozen experiment source are changed. On the first run, sixteen
fixed pilot episodes each receive two complete-VLM/D2 warmups, then three
complete-VLM/D2 references and three FFN-skip/D2 calls without compensation:
32 + 96 = 128 native requests. Completed observations and verified references
resume from their artifacts. A resumed partial episode needs two new warmups
for its fresh model runtime; these calls also count against the shared limit.
A failed numerical-reference gate is terminal for this output revision.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v3_4_E2_module_skip.offline import action_error

EXPERIMENT = ROOT / 'results/idea1_v3_4_E2_module_skip'
TARGETS = (12, 13, 14)
MAX_REQUESTS = 1000
FIRST_RUN_REQUESTS = 128
ATOL = RTOL = 1e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_bundle(experiment):
    """Read-only identity and data checks; no torch import or GPU work."""
    from preparations.idea1_v2_1.runtime_client import input_digest
    experiment = Path(experiment).resolve()
    matrix = json.loads((experiment / 'manifest.json').read_text())
    mechanism = experiment / 'pilot_mechanism'
    summary = json.loads((mechanism / 'summary.json').read_text())
    if summary['status'] != 'complete' or summary['manifest_hash'] != matrix['manifest_hash']:
        raise ValueError('Mechanism analysis and formal manifest must be complete and identical')
    baseline = mechanism / 'requests.jsonl'
    base_rows = json_rows(baseline)
    expected = {(budget, c['suite'], c['task'], c['init']) for budget in ('B4', 'B6') for c in matrix['pilot_cases']}
    if len(expected) != 16:
        raise ValueError('The frozen pilot set must contain sixteen budget/case identities')
    sources, episodes, seen = {}, [], set()
    frozen = {row['key']: row for row in summary['protocol']['source_identity']['sources']}

    def remember(path):
        path = Path(path).resolve()
        sources[str(path)] = sha(path)
        return path

    for p in (experiment / 'manifest.json', mechanism / 'manifest.json', mechanism / 'summary.json', baseline):
        remember(p)
    for filename in summary['reports']:
        report_path = remember(mechanism / filename)
        report = json.loads(report_path.read_text())
        case, budget = report['case'], report['budget']
        identity_key = (budget, case['suite'], case['task'], case['init'])
        if identity_key not in expected or identity_key in seen:
            raise ValueError('Unexpected or duplicate pilot episode')
        seen.add(identity_key)
        directory = experiment / f'{budget}_F-R+' / 'episodes' / case['suite'] / f"task{case['task']:02d}" / f"init{case['init']:02d}"
        identity_path = remember(directory / 'identity.json')
        identity = json.loads(identity_path.read_text())
        if (identity['case'] != case or identity['manifest_hash'] != matrix['manifest_hash']
                or identity['config']['id'] != f'{budget}_F-R+'):
            raise ValueError('Online episode identity does not match its pilot report')
        remember(directory / 'complete.json')
        inference = remember(directory / 'infer.jsonl')
        if sha(inference) != report['source_infer_sha256'] or sha(inference) != frozen[report['key']]['infer_sha256']:
            raise ValueError('Pilot inference journal changed since mechanism analysis')
        rows = {int(row['replan']): row for row in json_rows(inference)}
        observations = {int(row['replan']): row for row in report['observations']}
        input_hashes = {int(row['replan']): row['input_file_sha256'] for row in frozen[report['key']]['saved_inputs']}
        online_path = remember(directory / 'arrays.npz')
        reference_path = remember(mechanism / report['action_arrays'])
        targets = []
        with np.load(online_path, allow_pickle=False) as online, np.load(reference_path, allow_pickle=False) as references:
            for index in TARGETS:
                row, previous_reference = rows[index], observations[index]
                route = row['route']
                if (row['seed'] != previous_reference['seed'] or row['step'] != previous_reference['step']
                        or row['step'] != 5 * index or route['path'] != 'fast' or route['D'] != 2
                        or route['history_full_steps'] != [25, 50]
                        or not np.isclose(route['extrapolation_ratio'], (index - 10) / 5)
                        or route['compensation'] != 'extrapolate'):
                    raise ValueError('Saved target is not the declared late Fast with own Full history')
                path = remember(row['input_path'])
                if sources[str(path)] != row['input_file_sha256'] or sources[str(path)] != input_hashes[index]:
                    raise ValueError('Saved input hash differs from its online/mechanism identity')
                with np.load(path, allow_pickle=False) as saved:
                    prompt, seed = str(saved['instruction'].item()), int(saved['seed'].item())
                    obs = {k: saved[k].copy() for k in saved.files if k not in ('instruction', 'seed')}
                if seed != int(row['seed']) or prompt != identity['prompt'] or input_digest(obs, prompt) != row['input_hash']:
                    raise ValueError('Saved prompt, seed, or observation digest differs from the online request')
                for prefix, bank in (('', online), ('reference_', references)):
                    for name, shape in (('raw', (50, 32)), ('actions', (50, 7))):
                        value = bank[f'{prefix}{name}_{index}']
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError('Invalid saved raw/action array')
                targets.append(dict(replan=index, step=row['step'], seed=seed, prompt=prompt,
                    input_path=str(path), input_file_sha256=row['input_file_sha256'], input_hash=row['input_hash'],
                    skip_ffn=list(route['selected_skip']), age_env_steps=route['full_age_env_steps'],
                    extrapolation_ratio=route['extrapolation_ratio']))
        episodes.append(dict(key=report['key'], budget=budget, case=case, controller=identity['controller'],
            online_arrays=str(online_path), old_reference_arrays=str(reference_path), targets=targets))
    if seen != expected:
        raise ValueError('A declared pilot is missing; no outcome-dependent replacement is permitted')
    native_sources = (
        'preparations/idea1_v3_4_E2_module_skip/backend.py',
        'preparations/idea1_v3_4_E2_module_skip/offline.py',
        'preparations/idea1_v3/policy.py', 'preparations/idea1_v3/safe_qk.py',
        'preparations/idea1_v2_2/pi0_adapter.py',
        'preparations/idea1_v2_2/schedules.py',
        'policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py',
    )
    for name in native_sources:
        remember(ROOT / name)
    remember(__file__)
    return dict(matrix=matrix, episodes=episodes, sources=sources,
                base_ledger=str(baseline), base_requests=len(base_rows))


class Budget:
    def __init__(self, path, base_ledger, base_hash):
        self.path, self.base_ledger = Path(path), Path(base_ledger)
        self.base_hash = base_hash
        self.rows = json_rows(self.path) if self.path.exists() else []
        self.stream = self.path.open('a', buffering=1)
        self.counts()

    def counts(self):
        if sha(self.base_ledger) != self.base_hash:
            raise ValueError('Original offline request ledger changed; stop concurrent analysis')
        base = len(json_rows(self.base_ledger))
        total = base + len(self.rows)
        if total > MAX_REQUESTS:
            raise ValueError('The shared offline request limit was already exceeded')
        return dict(base_model_requests=base, probe_native_requests=len(self.rows),
                    combined_model_requests=total, maximum_model_requests=MAX_REQUESTS)

    def consume(self, episode, purpose, invocation):
        counts = self.counts()
        if counts['combined_model_requests'] >= MAX_REQUESTS:
            raise RuntimeError('Shared 1000-model-request budget exhausted before dispatch')
        row = dict(request=len(self.rows) + 1, episode=episode, purpose=purpose,
                   invocation=invocation, status='dispatched_or_attempted', time=time.time())
        self.stream.write(json.dumps(row) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.rows.append(row)
        return row['request']

    def close(self):
        self.stream.close()


def save_arrays(path, raw, actions):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, raw=raw, actions=actions)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_reference(raw, actions, old_raw, old_actions):
    result = dict(raw_equal=bool(np.allclose(raw, old_raw, atol=ATOL, rtol=RTOL)),
                  actions_equal=bool(np.allclose(actions, old_actions, atol=ATOL, rtol=RTOL)),
                  raw_max_abs=float(np.max(np.abs(raw - old_raw))),
                  actions_max_abs=float(np.max(np.abs(actions - old_actions))), atol=ATOL, rtol=RTOL)
    result['passed'] = result['raw_equal'] and result['actions_equal']
    return result


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[record['budget'], record['replan']].append(record)
    output = []
    for (budget, index), rows in sorted(groups.items()):
        if len(rows) != 8:
            raise ValueError('Each budget/age comparison must retain all eight fixed pilots')
        a = np.asarray([r['without_compensation']['first5']['raw35_rmse'] for r in rows])
        c = np.asarray([r['with_compensation']['first5']['raw35_rmse'] for r in rows])
        output.append(dict(budget=budget, replan=index, age_env_steps=5 * (index - 10),
            ratio=(index - 10) / 5, n=len(rows), uncompensated_mean=float(a.mean()), compensated_mean=float(c.mean()),
            uncompensated_median=float(np.median(a)), compensated_median=float(np.median(c)),
            paired_delta_mean=float((c - a).mean()), paired_delta_median=float(np.median(c - a)),
            mean_relative_change_pct=None if a.mean() == 0 else float((c.mean() / a.mean() - 1) * 100),
            compensation_improved=int((c < a).sum()), compensation_worsened=int((c > a).sum()), equal=int((c == a).sum()),
            uncompensated_gripper_flips=sum(r['without_compensation']['first5']['gripper_sign_flips'] for r in rows),
            compensated_gripper_flips=sum(r['with_compensation']['first5']['gripper_sign_flips'] for r in rows),
            gripper_action_denominator=5 * len(rows),
            uncompensated_translation_l2_rmse_m_mean=float(np.mean([r['without_compensation']['first5']['physical_translation_l2_rmse_m'] for r in rows])),
            compensated_translation_l2_rmse_m_mean=float(np.mean([r['with_compensation']['first5']['physical_translation_l2_rmse_m'] for r in rows])),
            uncompensated_rotation_vector_l2_rmse_rad_mean=float(np.mean([r['without_compensation']['first5']['physical_rotation_vector_l2_rmse_rad'] for r in rows])),
            compensated_rotation_vector_l2_rmse_rad_mean=float(np.mean([r['with_compensation']['first5']['physical_rotation_vector_l2_rmse_rad'] for r in rows]))))
    return output


def finish_summary(records, budget, matrix_manifest_hash, output):
    if len(records) != 48:
        raise ValueError('The fixed 48-observation probe is incomplete')
    summary = dict(status='complete', posthoc=True, quality_rollouts_added=0,
        included_in_original_40_case_evaluation=False, matrix_manifest_hash=matrix_manifest_hash,
        observations=48, pilot_episodes=16, numerical_reference_gates_passed=48,
        aggregates=aggregate(records), **budget.counts(),
        caveat='Paired same-input action perturbation on the sixteen F-R+ pilot trajectories; no new success-rate evidence and no online-policy modification',
        time=time.time())
    atomic(output / 'summary.json', summary)
    atomic(output / 'status.json', summary)
    return summary


def run(gpu, experiment=EXPERIMENT, output=None):
    # This must precede torch, Triton, and backend imports in this standalone process.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      TORCHINDUCTOR_COMPILE_THREADS='4', OMP_NUM_THREADS='4')
    bundle = source_bundle(experiment)
    output = Path(output) if output is not None else Path(experiment) / 'late_fast_probe'
    output.mkdir(parents=True, exist_ok=True)
    from preparations.idea1_v3_1.common import lock
    with lock(output / 'probe.lock', blocking=False):
        source_identity = dict(matrix_manifest_hash=bundle['matrix']['manifest_hash'], sources=bundle['sources'],
                               episodes=bundle['episodes'], base_model_requests=bundle['base_requests'])
        protocol = dict(schema_version=1, posthoc=True, quality_rollouts_added=0,
            included_in_original_40_case_evaluation=False, source_identity=source_identity,
            fixed_observations='16 declared pilots, saved replans 12/13/14; no replacement',
            comparison='actual online compensated F-R+ versus same-mask native FFN skip without compensation, on the same input/seed/current complete-VLM-D2 reference',
            warmup='two complete-VLM-D2 calls per episode on its replan12 input; every resumed partial episode warms a fresh runtime twice and charges both calls',
            first_run_native_requests=FIRST_RUN_REQUESTS, shared_model_request_limit=MAX_REQUESTS,
            reference_gate=dict(atol=ATOL, rtol=RTOL, failure='terminal; retain consumed calls, do not retry to match'),
            counterfactuals_enter_online_policy=False, source_sha256=sha(__file__))
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != protocol:
                raise ValueError('Probe sources/protocol changed; use a new output revision')
        else:
            if bundle['base_requests'] + FIRST_RUN_REQUESTS > MAX_REQUESTS:
                raise RuntimeError('Insufficient shared budget for the predeclared probe')
            atomic(manifest_path, protocol)
        if (output / 'summary.json').exists():
            return json.loads((output / 'summary.json').read_text())
        failure_path = output / 'reference_failure.json'
        if failure_path.exists():
            raise RuntimeError('A prior numerical-reference gate failed; this revision cannot retry to match')
        budget = Budget(output / 'requests.jsonl', bundle['base_ledger'], bundle['sources'][bundle['base_ledger']])
        invocation = uuid.uuid4().hex
        records = []
        pending = []
        for episode in bundle['episodes']:
            unfinished = []
            for target in episode['targets']:
                record_path = output / f"{episode['key']}_replan{target['replan']:02d}.json"
                if record_path.exists():
                    saved = json.loads(record_path.read_text())
                    if saved['source_identity_hash'] != digest(source_identity):
                        raise ValueError('Completed probe observation has a different source identity')
                    if (saved['episode'] != episode['key'] or saved['replan'] != target['replan']
                            or saved['seed'] != target['seed'] or saved['mask'] != target['skip_ffn']
                            or not saved['reference']['gate']['passed']
                            or sha(saved['uncompensated_arrays']) != saved['uncompensated_arrays_sha256']
                            or sha(Path(str(record_path)[:-5] + '_reference.npz')) != saved['reference']['arrays_sha256']):
                        raise ValueError('Completed probe artifact changed or has the wrong target identity')
                    records.append(saved)
                else:
                    prior_reference = Path(str(record_path)[:-5] + '_reference.json')
                    if prior_reference.exists() and not json.loads(prior_reference.read_text())['gate']['passed']:
                        raise RuntimeError('Saved numerical-reference gate failed; do not retry this revision')
                    unfinished.append(target)
            if unfinished:
                pending.append((episode, unfinished))
        if not pending:
            try:
                return finish_summary(records, budget, bundle['matrix']['manifest_hash'], output)
            finally:
                budget.close()
        import torch
        from preparations.idea1_v3.safe_qk import install
        from preparations.idea1_v3_4_E2_module_skip.backend import Backend
        from preparations.kv_reuse_probe.probe import load_observation
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        gpu_uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        install(str(output / f'qk_receipt_{invocation}.json'))
        start = time.perf_counter()
        backend = Backend()
        model_load_s = time.perf_counter() - start
        expected_checkpoint = bundle['matrix']['cost_table']['provenance']['protocol']['checkpoint_sha256']
        if backend.protocol['checkpoint_sha256'] != expected_checkpoint or backend.protocol['dtype'] != 'bfloat16':
            budget.close()
            raise ValueError('Native checkpoint/precision differs from the original experiment')
        atomic(output / f'invocation_{invocation}.json', dict(invocation=invocation, pid=os.getpid(), gpu=gpu,
            gpu_uuid=gpu_uuid, model_load_s=model_load_s, protocol=backend.protocol, started=time.time()))

        def native(episode, target, purpose, skip=()):
            side = dict(path=target['input_path'], sha256=target['input_file_sha256'],
                        seed=target['seed'], input_hash=target['input_hash'])
            obs, prompt = load_observation(side)
            request = budget.consume(episode['key'], purpose, invocation)
            tick = time.perf_counter()
            try:
                c, timing = backend.prepare_modules(obs, prompt, skip_ffn=skip)
                actions, raw, costs = backend.sample(c, target['seed'], 2)
                raw, actions = raw.float().cpu().numpy().copy(), np.asarray(actions, np.float32).copy()
                if raw.shape != (50, 32) or actions.shape != (50, 7) or not np.isfinite(raw).all() or not np.isfinite(actions).all():
                    raise ValueError('Nonfinite or incorrectly shaped native output')
                result = dict(request=request, invocation=invocation, purpose=purpose, episode=episode['key'],
                    replan=target['replan'], seed=target['seed'], timing=dict(timing, **costs),
                    wall_s=time.perf_counter() - tick, status='complete')
                atomic(output / f'request_{request:04d}.json', result)
                return raw, actions, result
            except Exception:
                atomic(output / f'request_{request:04d}.json', dict(request=request, invocation=invocation,
                    purpose=purpose, episode=episode['key'], status='failed', error=traceback.format_exc()))
                raise

        try:
            for episode, unfinished in pending:
                warm_target = episode['targets'][0]
                for n in range(2):
                    native(episode, warm_target, f'warmup{n + 1}')
                with np.load(episode['online_arrays'], allow_pickle=False) as online, np.load(episode['old_reference_arrays'], allow_pickle=False) as old:
                    for target in unfinished:
                        index = target['replan']
                        stem = output / f"{episode['key']}_replan{index:02d}"
                        ref_meta_path = Path(str(stem) + '_reference.json')
                        ref_array_path = Path(str(stem) + '_reference.npz')
                        if ref_meta_path.exists():
                            ref_meta = json.loads(ref_meta_path.read_text())
                            if (ref_meta['source_identity_hash'] != digest(source_identity)
                                    or not ref_meta['gate']['passed'] or sha(ref_array_path) != ref_meta['arrays_sha256']):
                                raise ValueError('Saved reference is invalid or changed')
                            with np.load(ref_array_path, allow_pickle=False) as saved:
                                ref_raw, ref_actions = saved['raw'].copy(), saved['actions'].copy()
                        else:
                            ref_raw, ref_actions, receipt = native(episode, target, 'complete_vlm_d2_reference')
                            save_arrays(ref_array_path, ref_raw, ref_actions)
                            gate = check_reference(ref_raw, ref_actions, old[f'reference_raw_{index}'], old[f'reference_actions_{index}'])
                            ref_meta = dict(source_identity_hash=digest(source_identity), gate=gate,
                                receipt=receipt, arrays_sha256=sha(ref_array_path))
                            atomic(ref_meta_path, ref_meta)
                            if not gate['passed']:
                                atomic(failure_path, dict(episode=episode['key'], replan=index, reference=ref_meta,
                                    **budget.counts(), error='Native complete-VLM/D2 reference differs from the saved mechanism reference'))
                                raise RuntimeError('Native numerical-reference gate failed; no counterfactual dispatched')
                        array_path = Path(str(stem) + '_uncompensated.npz')
                        uncomp_meta_path = Path(str(stem) + '_uncompensated.json')
                        if uncomp_meta_path.exists():
                            uncomp_meta = json.loads(uncomp_meta_path.read_text())
                            if (uncomp_meta['source_identity_hash'] != digest(source_identity)
                                    or uncomp_meta['mask'] != target['skip_ffn']
                                    or sha(array_path) != uncomp_meta['arrays_sha256']):
                                raise ValueError('Saved uncompensated output is invalid or changed')
                            receipt = uncomp_meta['receipt']
                            with np.load(array_path, allow_pickle=False) as saved:
                                raw, actions = saved['raw'].copy(), saved['actions'].copy()
                        else:
                            raw, actions, receipt = native(episode, target, 'same_ffn_mask_without_compensation', target['skip_ffn'])
                            save_arrays(array_path, raw, actions)
                            atomic(uncomp_meta_path, dict(source_identity_hash=digest(source_identity),
                                mask=target['skip_ffn'], receipt=receipt, arrays_sha256=sha(array_path)))
                        record = dict(source_identity_hash=digest(source_identity), episode=episode['key'],
                            budget=episode['budget'], case=episode['case'], replan=index, step=target['step'],
                            seed=target['seed'], mask=target['skip_ffn'], age_env_steps=target['age_env_steps'],
                            extrapolation_ratio=target['extrapolation_ratio'], reference=ref_meta, native_receipt=receipt,
                            uncompensated_arrays=str(array_path), uncompensated_arrays_sha256=sha(array_path),
                            without_compensation=action_error(ref_raw, raw, ref_actions, actions, episode['controller']),
                            with_compensation=action_error(ref_raw, online[f'raw_{index}'], ref_actions, online[f'actions_{index}'], episode['controller']))
                        atomic(Path(str(stem) + '.json'), record)
                        records.append(record)
                        atomic(output / 'status.json', dict(status='running', completed_observations=len(records),
                            target_observations=48, **budget.counts(), time=time.time()))
            return finish_summary(records, budget, bundle['matrix']['manifest_hash'], output)
        except Exception:
            atomic(output / f'failure_{invocation}.json', dict(error=traceback.format_exc(), **budget.counts(), time=time.time()))
            raise
        finally:
            budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--experiment', type=Path, default=EXPERIMENT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true', help='Read sources/budget without importing torch or creating outputs')
    args = parser.parse_args()
    if args.validate_only:
        bundle = source_bundle(args.experiment)
        print(json.dumps(dict(status='sources_validated', pilot_episodes=len(bundle['episodes']),
            observations=sum(len(e['targets']) for e in bundle['episodes']), base_model_requests=bundle['base_requests'],
            planned_native_requests=FIRST_RUN_REQUESTS, planned_combined_requests=bundle['base_requests'] + FIRST_RUN_REQUESTS)))
    else:
        print(json.dumps(run(args.gpu, args.experiment, args.output)), flush=True)


if __name__ == '__main__':
    main()
