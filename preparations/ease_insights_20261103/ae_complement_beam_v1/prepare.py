"""Freeze the bounded replay matrix before any GPU output from this cohort."""
import copy
import json
from pathlib import Path
import time
from .common import *


def prepare():
    target = OUTPUT / 'manifest.json'
    if target.exists():
        value = read_manifest(target, verify_sources=True)
        return dict(reused=True, manifest_hash=value['manifest_hash'], manifest_file_sha256=sha(target))
    base_path = ROOT / 'results/ease_insights_20261103/denoise_depth_paths_v1/discovery/manifest.json'
    from preparations.ease_insights_20261103.denoise_depth_paths_v1.common import read_manifest as read_prior
    prior = read_prior(base_path, verify_sources=True)
    folder = ROOT / 'reports/ease_insights_20261103/ae_complement_beam_v1_preparation'
    inventory_path = folder / 'observation_inventory.json'
    require(sha(inventory_path) == '94249d6613b21a9ab4b73c48628c44760a99bd09644f2e7a7e036e37cbe49550', 'Input inventory changed')
    inventory = json.loads(inventory_path.read_text())
    require(inventory['status'] == 'pass' and inventory['observations'] == 110 and inventory['tasks'] == 40, 'Incomplete input inventory')
    require(sha(inventory['input_manifest']['path']) == inventory['input_manifest']['sha256'], 'Input source manifest changed')
    observations = copy.deepcopy(inventory['observation_evidence'])
    for row in observations:
        load_observation(row)
    norm = next(p for p in prior['checkpoints']['pi0']['identity_artifacts'] if p.endswith('/norm_stats.json'))
    sources = dict(prior['source_files'])
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    for p in [inventory_path, folder / 'proposal.json']:
        sources[str(p.relative_to(ROOT))] = sha(p)
    value = dict(revision='ease-AE16-joint-mask-screen-v1', mode='offline_snapshot_replay', created_at=time.time(),
                 output_root=str(OUTPUT), planned_conditions=['joint16_d2'], conditions=[CONDITION],
                 observations=observations, masks=MASKS, noise_seeds=[12345, 23456, 34567], source_files=sources,
                 checkpoints=prior['checkpoints'], normalization=dict(path=norm, sha256=sha(norm)),
                 input_inventory=dict(path=str(inventory_path), sha256=sha(inventory_path)),
                 physical_episodes=0, new_official_initial_states=0, replay_units=110, joint_mask_samples=50490,
                 dense_reference_samples=330, quality_evidence=False, closed_loop_confirmation_allocated=False,
                 input_contract='Archived uint8 agent/wrist images already rotated and resized224x224; direct8D robot_state. Use native adapter preprocessed branch without another rotation or quaternion reconstruction.',
                 comparison='Fresh denseAE18/D2 reference and all153 skip-two AE16/D2 variants on identical full VLM18/vision27 prefix and identical generated noise. No additive single-layer approximation.',
                 error_definition='Mean squared decoded error over first5 actions and all7 command dimensions, divided by max(mean squared same-input dense reference over those values,1e-4). Average seeds within snapshot, snapshots within task, then40 tasks equally.',
                 proxy_rule=dict(overall_task_mean_NMSE_max=.05, max_suite_mean_NMSE=.1,
                                 minimum_opposite_suite_mean_error_gap=.003, minimum_positive_tasks=6, every_task_leave_one_out_positive=True),
                 selection_rule='Only after all110 complete records and both service gates pass. Rank minimum opposite-suite error gap, then sum, then smaller worst overall error, then lexical masks/suites. Select one pair; no inference or rollout quality claim.',
                 initial_only_sensitivity='Separately labeled, cannot replace primary all-phase selected masks.',
                 future_requirement='A proxy-selected pair requires an independently frozen closed-loop discovery, measured native cost and fresh initial-state confirmation before supporting task-dependent path preference.',
                 resource_accounting='Inherited scheduler service new_episodes/completed_after fields count archived replay units only; physical episodes remain zero. Retain existing two GPU leases and16GiB idle placeholders.',
                 historical_normalization_or_runtime_recreated=False,
                 design_proposal=dict(path=str(folder / 'proposal.json'), sha256=sha(folder / 'proposal.json')))
    value['manifest_hash'] = digest(value)
    validate_output(OUTPUT, value)
    atomic_json(target, value)
    read_manifest(target, verify_sources=True)
    return dict(reused=False, manifest_hash=value['manifest_hash'], manifest_file_sha256=sha(target), source_files=len(sources), physical_episodes=0, replay_units=110, GPU_started=False)


if __name__ == '__main__':
    print(json.dumps(prepare()))
