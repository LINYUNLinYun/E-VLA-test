"""Final protocol coverage, denominator and deliverable evidence index."""
import hashlib
import json
from pathlib import Path
import time
from preparations.idea1_v3.io import atomic
from .analyze import OUT,ROOT,records
from .cost_analysis import load_nested
from .manifest import cases


def main():
    state=json.loads((OUT/'campaign_state.json').read_text())
    if state['phase']!='scientific_complete':raise ValueError('Scientific campaign not complete')
    parent=json.loads((OUT/'manifest.json').read_text());bounded=json.loads((OUT/'stages/bounded_followups/manifest.json').read_text());confirmation=json.loads((OUT/'stages/confirm300/manifest.json').read_text())
    configs={c['id']:c for m in [parent,bounded,confirmation] for c in m['conditions']};expanded={c['id'] for c in confirmation['conditions']};groups=records();coverage=[]
    if set(groups)!=set(configs):raise ValueError('Unexpected scientific condition or denominator pollution')
    for cid,cfg in configs.items():
        expected={tuple(c[k] for k in ('suite','task','init')) for c in cases(cfg['model'],expand=cid in expanded)}
        actual=[tuple(r['case'][k] for k in ('suite','task','init')) for r in groups[cid]]
        if set(actual)!=expected or len(actual)!=len(expected):raise ValueError('Final exact case coverage mismatch '+cid)
        if any(not r['valid'] or r['config']!=cfg for r in groups[cid]):raise ValueError('Final configuration/validity mismatch '+cid)
        coverage.append(dict(condition=cid,case_count=len(actual),target=300 if cid in expanded else 80,successes=sum(r['success'] for r in groups[cid]),passed=True))
    total=sum(len(rows) for rows in groups.values())
    if total>6540:raise ValueError('Science hard budget exceeded')
    engineering={p.name:len(load_nested(p)) for p in (OUT/'engineering_cost_rebaseline').iterdir() if p.is_dir() and (p/'episodes').exists()}
    if sum(engineering.values())!=400:raise ValueError('Engineering valid denominator differs from authorized400')
    for path,sha in parent['source_files'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError('Original frozen source drift '+path)
    audit=json.loads((OUT/'final_execution_audit.json').read_text())
    if not audit['passed'] or audit['cases']!=total:raise ValueError('Final execution audit does not cover all science')
    requirements={
        'protocol_and_case_identity':['manifest.json','E2_manifest.json','E2_calibration.json','stages/bounded_followups/manifest.json','stages/confirm300/manifest.json','stages/bounded_followups/freeze_audit.json','stages/confirm300/freeze_audit.json','stages/bounded_followups/first_case_gate_receipt.json','stages/confirm300/first_case_gate_receipt.json'],
        'native_models_checkpoints_and_asset_hashes':['pi0_parent/episodes/libero_spatial/task00/init06/identity.json','xiaomi_full_d2/episodes/libero_spatial/task00/init00/identity.json'],
        'E1_native_D1_grid_quality':['D1_native_euler_probe.json','request_bank/D1_native_euler_arrays.npz','reports/E1_report_zh.md'],
        'E2_strict_world_pose_gate_no_c_and_noise':['E2_gate_audit.json','E2_gate_decisions.jsonl','E2_padding_diagnostic_receipt.json','metadata_revision_r2.json','bounded_probe_history_h1.json','bounded_probe_history_h1_execution.json','probe_prompt_history_receipt.json','E2_no_c_case_mechanism_receipt.json','reports/E2_report_zh.md'],
        'E3_soft_and_hard_controls':['reports/E3_report_zh.md','reports/ablation_pairing.json'],
        'E4_no_gripper_P2_native_boundary_and_periodic':['xiaomi_full_native_probe.json','E4_protocol_audit.json','E4_tau0025_full_exact_receipt.json','E4_tau08_periodic_exact_receipt.json','E4_tau08_tau16_exact_receipt.json','E4_tau08_periodic_all_shared_comparison.json','E4_timing_device_scope_receipt.json','E4_single_case_difference_receipt.json','E4_causal_interpretation_receipt.json','reports/E4_report_zh.md'],
        'matched_cost_and_engineering_history':['cost_gate_pi0_screen.json','cost_gate_final.json','cost_gate_final_profiles.json','engineering_cost_rebaseline/initial_task_numeric_history_matrix.json','engineering_cost_rebaseline/legacy_timing_confound.json'],
        'single_worker_profile_and_auxiliary_bank':['bounded_profile.json','request_bank/pi0_gpu0.json','request_bank/xiaomi_gpu2.json','request_bank/timing_order_receipt.json'],
        'GPU1_lease_and_physical_EGL':['external_gpu_leases.json','gpu1_release_receipt.json','egl_revision.json','egl_parity_gpu0.json','egl_parity_gpu2.json','egl_parity_gpu3.json','final_resource_release.json'],
        'complete_data_statistics_and_actual_XYZ':['reports/episodes.csv','reports/requests.csv','reports/tasks.csv','reports/module_times.csv','reports/cost_by_physical_gpu.csv','reports/matched_confirmation_costs.json','reports/campaign_wall_receipt.json','reports/confirmation_decision.json','reports/execution_status_inventory.json','reports/report_receipt.json','final_execution_audit.json'],
    }
    paths={name:dict(path=str(OUT/name),sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()) for names in requirements.values() for name in names}
    for exp in ['E1','E2','E3','E4']:
        p=OUT/'reports'/f'{exp}_fixed_xyz.png'
        if not p.exists():raise ValueError('Missing fixed-case actual XYZ figure '+exp)
        paths[str(p.relative_to(OUT))]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    atomic(OUT/'execution_evidence_index.json',dict(time=time.time(),status='complete',scientific_valid_cases=total,scientific_hard_budget=6540,
        engineering_valid_cases=400,engineering_conditions=engineering,engineering_science_denominator_increment=0,exact_condition_case_coverage=coverage,
        original_frozen_sources_unchanged=True,requirements=requirements,files=paths,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Official task/init subsets80/300, not the complete50-initial-state LIBERO benchmark.','Selection and clustered bootstrap do not prove population noninferiority; multiple comparisons are unadjusted.','Known Pi first-prompt history and GPU0 cold preparation confounds remain explicit; no unmatched cost fallback.','E4 original tau.8/periodic wall gap is cross-GPU descriptive; action equivalence and measured verifier CUDA-event module cost are separate evidence.']))


if __name__=='__main__':main()
