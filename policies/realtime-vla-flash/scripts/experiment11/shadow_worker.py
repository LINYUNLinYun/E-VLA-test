"""Predict one selected skip mask on every saved Full18 replan input; never steps the environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.experiment11.protocol import RESULTS, SOURCE, SUITES
from scripts.experiment10.common import dump_json, json_lines, sha256


BASELINE = 'Full18_D3_R5_FORMAL'


def selected_variants(path: Path | None = None) -> dict[str, dict]:
    path = path or RESULTS / 'protocol/selected_shadow_manifest.json'
    manifest = json.loads(path.read_text())
    if manifest.get('admission') == 'user_approved_exception':
        from scripts.experiment11.launch_shadow_supplement import validate_manifest
        validate_manifest(manifest)
        return {row['variant_id']: row for row in manifest['selected_new_masks']}
    if manifest.get('status') != 'eligible' or not manifest.get('selected_new_masks'):
        raise RuntimeError('No new mask passed the D3 80% gate; shadow may not launch without user direction')
    result = {row['variant_id']: row for row in manifest['selected_new_masks']}
    reference = manifest['original_v13_reference']
    result[reference['variant_id']] = reference
    if len(result) > 5:
        raise RuntimeError('Shadow selection exceeds four new masks plus original V13 reference')
    return result


def _input_digest(data, prompt: str, suite: str) -> str:
    digest = hashlib.sha256()
    for name in ('agent_image', 'wrist_image', 'robot_state', 'noise'):
        digest.update(np.ascontiguousarray(data[name]).tobytes())
    digest.update(prompt.encode('utf-8'))
    digest.update(suite.encode('utf-8'))
    return digest.hexdigest()


def run(variant_id: str, suite: str, gpu_index: int, *,
        selection_path: Path | None = None, output_dir: Path | None = None) -> int:
    if suite not in SUITES:
        raise ValueError(f'Unsupported suite: {suite}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu_index):
        raise RuntimeError('Shadow worker is not bound to its leased physical GPU')
    import torch
    from scripts.experiment10.backend import Backend
    from scripts.experiment11.call_budget import CallBudget, accounted_infer

    budget_path = os.environ.get('EXP11_V14_CALL_LEDGER')
    budget = CallBudget(Path(budget_path), 7000) if budget_path else None

    torch.cuda.init()
    status_path = RESULTS / 'experiment11_closed_loop_status.json'
    status = json.loads(status_path.read_text())
    audit_path = RESULTS / 'analysis/closed_loop_d3/completion_audit.json'
    if (not status.get('complete') or status.get('active') or status.get('pending') or
            status.get('failures') or not audit_path.is_file()):
        raise RuntimeError('Full D3 closed-loop matrix and audit must be complete before shadow')
    audit = json.loads(audit_path.read_text())
    if audit.get('condition_suite_runs_validated') != 124 or audit.get('episodes_validated') != 6200:
        raise RuntimeError('D3 completion audit is incomplete')

    selection_path = selection_path or RESULTS / 'protocol/selected_shadow_manifest.json'
    candidates = selected_variants(selection_path)
    if variant_id not in candidates:
        raise ValueError(f'{variant_id} is not in the frozen shadow manifest')
    candidate = candidates[variant_id]
    skip = tuple(int(value) for value in candidate['skip_layers'])
    if not skip:
        raise ValueError('Shadow variant must be a VLM layer-skip mask')
    condition_name = candidate['condition_name']
    checkpoint_hash = json.loads((RESULTS / 'protocol/candidate_manifest.json').read_text())['checkpoint_sha256']

    base_run = RESULTS / 'closed_loop' / BASELINE / suite / 'run'
    base_validation = json.loads((base_run / 'validated.json').read_text())
    if not base_validation.get('complete') or base_validation.get('episodes') != 50:
        raise RuntimeError(f'Full reference suite not validated: {suite}')
    episodes = json.loads((base_run / 'episode_log.json').read_text())
    if len(episodes) != 50:
        raise RuntimeError(f'Expected 50 Full episodes in {suite}, got {len(episodes)}')

    source_entries = json.loads((SOURCE / 'observations' / suite / 'manifest.json').read_text())['observations']
    source_metadata = []
    for row in source_entries:
        with np.load(row['path'], allow_pickle=False) as data:
            source_metadata.append(json.loads(str(data['metadata_json'].item())))
    backend = Backend(source_metadata)
    output_dir = output_dir or RESULTS / 'shadow' / variant_id / suite
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / 'predictions.jsonl'
    partial_path = output_dir / 'predictions.jsonl.partial'
    validated_path = output_dir / 'validated.json'
    if final_path.exists() or validated_path.exists() or partial_path.exists():
        raise FileExistsError(f'Shadow output already exists; inspect before retrying: {output_dir}')

    source_hashes = {'full_episode_log_sha256': sha256(base_run / 'episode_log.json'),
                     'selected_shadow_manifest_sha256': sha256(selection_path),
                     'candidate_manifest_sha256': sha256(RESULTS / 'protocol/candidate_manifest.json')}
    expected_records = 0
    parity_error_max = None
    with partial_path.open('x', encoding='utf-8') as stream:
        for episode in sorted(episodes, key=lambda row: (int(row['task_id']), int(row['episode_idx']))):
            infer_path = base_run / episode['infer_path']
            inference = json_lines(infer_path)
            if len(inference) != int(episode['infer_record_count']) or not inference:
                raise RuntimeError(f'Full inference coverage mismatch: {infer_path}')
            telemetry = json_lines(base_run / episode['telemetry_path'])
            telemetry_by_step = {int(row['step']): row for row in telemetry}
            for infer in inference:
                infer_id = int(infer['infer_id'])
                source_meta = infer.get('experiment11', {})
                if (source_meta.get('condition') != BASELINE or
                        source_meta.get('checkpoint_sha256') != checkpoint_hash or
                        source_meta.get('candidate_skip_layers_1based') != [] or
                        source_meta.get('denoise_steps') != 3 or infer.get('accepted_prefix_len') != 5):
                    raise RuntimeError(f'Full source provenance/replan mismatch: {infer_path}#{infer_id}')
                input_path = Path(source_meta.get('full_input_path', ''))
                if not input_path.is_file() or input_path.resolve().parent.parent.parent.name != suite:
                    raise RuntimeError(f'Invalid saved Full observation path: {input_path}')
                with np.load(input_path, allow_pickle=False) as data:
                    input_meta = json.loads(str(data['metadata_json'].item()))
                    prompt = str(data['prompt'].item())
                    observed_hash = _input_digest(data, prompt, suite)
                    noise = np.asarray(data['noise'], dtype=np.float32).copy()
                    if (observed_hash != source_meta.get('full_input_sha256') or
                            observed_hash != input_meta.get('input_sha256') or
                            input_meta.get('checkpoint_sha256') != checkpoint_hash or
                            input_meta.get('skip_layers') != [] or input_meta.get('denoise_steps') != 3 or
                            input_meta.get('suite') != suite or
                            int(input_meta.get('task_id', -1)) != int(episode['task_id']) or
                            int(input_meta.get('episode_idx', -1)) != int(episode['episode_idx']) or
                            int(input_meta.get('infer_id', -1)) != infer_id or
                            hashlib.sha256(noise.tobytes()).hexdigest() != source_meta.get('noise_sha256')):
                        raise RuntimeError(f'Saved Full input hash/provenance mismatch: {input_path}')
                    observation = {
                        'observation/image': data['agent_image'],
                        'observation/wrist_image': data['wrist_image'],
                        'observation/state': data['robot_state'], 'prompt': prompt,
                    }
                    backend.prepare(observation, {'suite': suite, 'task_description': prompt}, noise)

                full_actions = np.asarray(infer['chunk_actions'], dtype=np.float32)
                if full_actions.shape != (50, 7) or not np.isfinite(full_actions).all():
                    raise RuntimeError(f'Full prediction is not finite 50x7: {infer_path}#{infer_id}')
                sample_parity = None
                if parity_error_max is None:
                    _, recomputed_full, _ = accounted_infer(
                        backend, (), 3, budget=budget, identity={'suite': suite, 'kind': 'full_parity',
                        'infer_path': str(infer_path), 'replan_id': infer_id})
                    recomputed_full = np.asarray(recomputed_full, dtype=np.float32)
                    if recomputed_full.shape != full_actions.shape or not np.allclose(
                            recomputed_full, full_actions, rtol=1e-4, atol=1e-5):
                        raise RuntimeError(f'Full18 shadow-replay parity failed at {suite}/{episode["task_id"]}/{episode["episode_idx"]}')
                    sample_parity = float(np.max(np.abs(recomputed_full - full_actions)))
                    parity_error_max = sample_parity
                _, variant_actions, timing = accounted_infer(
                    backend, skip, 3, budget=budget, identity={'suite': suite, 'kind': 'candidate',
                    'variant_id': variant_id, 'infer_path': str(infer_path), 'replan_id': infer_id})
                variant_actions = np.asarray(variant_actions, dtype=np.float32)
                if variant_actions.shape != (50, 7) or not np.isfinite(variant_actions).all():
                    raise RuntimeError(f'Variant prediction is not finite 50x7: {variant_id}/{suite}/{infer_id}')
                frame_idx = int(infer['frame_idx'])
                phase_row = telemetry_by_step.get(frame_idx, {})
                row = {
                    'schema': 1, 'variant_id': variant_id, 'condition_name': condition_name,
                    'skip_layers': list(skip), 'suite': suite,
                    'task_id': int(episode['task_id']), 'episode_idx': int(episode['episode_idx']),
                    'full_episode_success': bool(episode['success']), 'full_episode_phase': episode.get('furthest_semantic_phase'),
                    'replan_id': infer_id, 'environment_action_step': frame_idx,
                    'semantic_phase': phase_row.get('semantic_phase', 'unknown'),
                    'full_input_path': str(input_path), 'full_input_sha256': observed_hash,
                    'noise_sha256': source_meta['noise_sha256'], 'checkpoint_sha256': checkpoint_hash,
                    'full_infer_path': str(infer_path), 'full_actions': full_actions.tolist(),
                    'variant_actions': variant_actions.tolist(),
                    'variant_cuda_policy_ms': float(timing['cuda_policy_ms']),
                    'full_recomputed_parity_max_abs_error': sample_parity,
                    'environment_action_source': 'Full18 only; this worker never calls env.step',
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                expected_records += 1
            stream.flush()
            os.fsync(stream.fileno())

    if expected_records == 0:
        raise RuntimeError('No Full replan samples were replayed')
    partial_path.replace(final_path)
    dump_json(validated_path, {
        'complete': True, 'variant_id': variant_id, 'condition_name': condition_name,
        'skip_layers': list(skip), 'suite': suite, 'episode_count': len(episodes),
        'replan_prediction_count': expected_records, 'action_horizon': 50, 'denoise_steps': 3,
        'replan_execute_prefix': 5, 'parity_suite': suite,
        'full_recomputed_parity_max_abs_error': parity_error_max,
        'source_hashes': source_hashes, 'predictions_sha256': sha256(final_path),
        'environment_steps_executed_by_variant': 0,
    })
    print(json.dumps({'variant_id': variant_id, 'suite': suite,
                      'replan_prediction_count': expected_records,
                      'full_parity_max_abs_error': parity_error_max,
                      'validated': str(validated_path)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant-id', required=True)
    parser.add_argument('--suite', required=True, choices=SUITES)
    parser.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    return run(args.variant_id, args.suite, args.gpu_index)


if __name__ == '__main__':
    raise SystemExit(main())
