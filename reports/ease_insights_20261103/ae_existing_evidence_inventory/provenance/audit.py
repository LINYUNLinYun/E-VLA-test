"""CPU-only contract/provenance audit; no success-rate or timing reanalysis."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/root/nfs/embodied/benchmark')
POLICY = ROOT / 'policies/realtime-vla-flash'
EXPERIMENT = POLICY / 'results/experiment9_standard_vlm_pruning_20261004'
CLOSED = EXPERIMENT / 'action_expert_layer_probe/closed_loop'
OUT = ROOT / 'reports/ease_insights_20261103/ae_existing_evidence_inventory/provenance'
SOURCES = [
    'scripts/experiment9/launch_ae_pilot_pair.py',
    'scripts/experiment9/run_action_expert_followup.py',
    'scripts/experiment9/launch_ae_pilot_d3.py',
    'scripts/experiment9/launch_early_ae_full_suite.py',
    'scripts/experiment9/serve_action_layer_screen.py',
    'scripts/experiment9/screen_action_expert_candidates.py',
    'scripts/experiment9/run_standard_vlm_pruning.py',
    'scripts/spec/spec_serve_policy.py',
    'scripts/spec/triton/triton_pi0_runtime.py',
    'scripts/spec/triton/pi0_infer.py',
    'scripts/experiment9/probe_action_expert_layers.py',
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def main() -> None:
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Eight exact historical AE single-layer pilot runs, first and last request of first and last episode per run; no outcomes/timing statistics recalculated.',
        'current_source_hashes_are_not_historical_execution_hashes': True,
        'current_sources': [{'path': rel(POLICY / s), 'sha256': sha(POLICY / s)} for s in SOURCES],
        'runs': [], 'violations': [],
        'coverage': '32 sampled requests from 16 files in eight runs; all 800 episodes or all requests were NOT decoded for this provenance audit.',
        'noise_hash_limit': 'Recorded noise_sha256 is a CPU reconstruction in current _attach_ablation_metadata, not a saved actual post-conversion GPU input tensor. Equality here verifies the reported deterministic rule.',
    }
    all_noise = {}
    for run in sorted(CLOSED.glob('exp9_ae_layer_pilot_SkipAE_L[28]_libero_*_i00_10_a01')):
        layer = int(run.name.split('SkipAE_L')[1].split('_')[0])
        manifest = json.loads((run / 'manifest.json').read_text())
        episodes = json.loads((run / 'episode_log.json').read_text())
        retained = [i for i in range(1, 19) if i != layer]
        counts = Counter()
        signatures = Counter()
        infer_index = []
        keys = set()
        for episode in (episodes[0], episodes[-1]):
            path = run / episode['infer_path']
            raw_bytes = path.read_bytes()
            raw_lines = [line for line in raw_bytes.splitlines() if line.strip()]
            rows = len(raw_lines)
            for sample_line_index in sorted({0, rows - 1}):
                    raw = raw_lines[sample_line_index]
                    row = json.loads(raw)
                    counts['requests'] += 1
                    meta = row['server_experiment']
                    keys.update(meta)
                    episode_idx = row['episode_idx']
                    infer_id = row['infer_id']
                    cache_key = (episode_idx, infer_id)
                    if cache_key not in all_noise:
                        noise = np.random.default_rng(np.random.SeedSequence([12345, episode_idx, infer_id])).standard_normal((50, 32)).astype(np.float32)
                        all_noise[cache_key] = hashlib.sha256(noise.tobytes()).hexdigest()
                    expected = {
                        'run_id': row.get('run_id') == run.name,
                        'suite': row.get('task_suite_name') == manifest['task_suite_name'],
                        'H50x7': len(row['chunk_actions']) == 50 and all(len(a) == 7 for a in row['chunk_actions']),
                        'VLM_full18': meta.get('vlm_layer_pattern') == 'vlm_full18' and meta.get('vlm_layer_indices_1based') == list(range(1, 19)),
                        'VLM_full_blocks18_no_KV_shell': meta.get('vlm_full_block_indices_1based') == list(range(1, 19)) and meta.get('vlm_kv_shell_indices_1based') == [],
                        'AE_skip_exact_layer': meta.get('action_layer_pattern') == f'ae_skip_l{layer}' and meta.get('action_layer_indices_1based') == retained,
                        'D10_times_dt': meta.get('denoise_steps') == 10 and meta.get('denoise_times') == [1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1] and meta.get('flow_dt') == -.1,
                        'noise_seed_rule': meta.get('ablation_noise_seed') == 12345 and meta.get('noise_seed_mode') == 'episode_infer' and meta.get('noise_seed_rule') == 'SeedSequence(global_noise_seed, episode_idx, replan_idx)',
                        'trace_noise_identity': meta.get('trace_episode_idx') == episode_idx and meta.get('trace_infer_id') == infer_id,
                        'reconstructed_reported_noise_sha256': meta.get('noise_sha256') == all_noise[cache_key],
                        'full_round_prefill': row.get('route_type') == 'full' and row.get('did_prefill') == 1 and row.get('is_full_pipeline_round') == 1 and row.get('used_full_fallback') == 1,
                        'no_draft_or_verify': row.get('draft_ms') == 0 and row.get('action_verify_ms') == 0,
                        'R12': row.get('accepted_prefix_len') == 12 and row.get('chunk_exec_len') == 12,
                    }
                    for name, passed in expected.items():
                        if passed:
                            counts[name] += 1
                        else:
                            result['violations'].append({'run': run.name, 'path': rel(path), 'line': sample_line_index + 1, 'check': name})
                    signatures[json.dumps(meta, sort_keys=True)] += 1
            if rows != episode['infer_record_count']:
                result['violations'].append({'run': run.name, 'path': rel(path), 'check': 'infer_record_count', 'actual': rows, 'declared': episode['infer_record_count']})
            infer_index.append({'path': rel(path), 'sha256': hashlib.sha256(raw_bytes).hexdigest(), 'file_records_counted_without_decoding': rows, 'sampled_line_numbers_1based': sorted({1, rows})})
        server = CLOSED / 'logs' / (run.name + '_server.log')
        selected = []
        log_counts = Counter()
        for n, line in enumerate(server.read_text().splitlines(), 1):
            if 'Physical VLM layers' in line or 'Experiment layers:' in line or 'norm stats' in line.lower():
                selected.append({'line': n, 'text': line})
            if 'actions shape=' in line:
                log_counts[line.removeprefix('INFO:root:')] += 1
        result['runs'].append({
            'run': run.name, 'layer_1based': layer,
            'manifest': {'path': rel(run / 'manifest.json'), 'sha256': sha(run / 'manifest.json'), 'fields': manifest},
            'episode_log': {'path': rel(run / 'episode_log.json'), 'sha256': sha(run / 'episode_log.json'), 'episodes': len(episodes)},
            'sampled_request_checks_passed': counts,
            'server_experiment_keys_union': sorted(keys),
            'server_log': {'path': rel(server), 'sha256': sha(server), 'selected_direct_evidence': selected, 'action_shape_log_counts': log_counts},
            'infer_files': infer_index,
        })
    probe = EXPERIMENT / 'action_expert_layer_probe/probe/summary.json'
    p = json.loads(probe.read_text())
    norm = ROOT / 'shared/checkpoints/realtime-vla-flash/converted/base/assets/physical-intelligence/libero/norm_stats.json'
    result['checkpoint_and_norm'] = {
        'historical_offline_probe_summary': {'path': rel(probe), 'sha256': sha(probe), 'checkpoint_sha256': p['checkpoint_sha256'], 'checkpoint_unchanged': p['checkpoint_unchanged']},
        'closed_loop_checkpoint_hash': None,
        'closed_loop_checkpoint_hash_limit': 'Offline probe records the base weight digest, but no closed-loop per-run artifact binds its actual model bytes to this digest.',
        'current_norm': {'path': rel(norm), 'sha256': sha(norm), 'historical_run_hash_binding': False},
        'no_large_checkpoint_hashed_or_loaded': True,
    }
    for name in ['registered_pattern_parity.json', 'pattern_parity_failure.json']:
        p = CLOSED / name
        result[name] = {'path': rel(p), 'sha256': sha(p), 'contents': json.loads(p.read_text())}
    result['sampled_request_records'] = sum(r['sampled_request_checks_passed']['requests'] for r in result['runs'])
    result['historical_complete_argv_record_found'] = False
    result['historical_execution_source_hash_record_found'] = False
    result['sampled_contract_audit_passed'] = len(result['runs']) == 8 and not result['violations']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'runs': len(result['runs']), 'requests': result['sampled_request_records'], 'violations': len(result['violations']), 'sampled_contract_audit_passed': result['sampled_contract_audit_passed']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
