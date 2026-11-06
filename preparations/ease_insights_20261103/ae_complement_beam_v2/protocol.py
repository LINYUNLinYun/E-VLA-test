"""Prospective world test of one offline-selected pair; directions cannot change."""
import json
import time
from .common import *

REPORT = ROOT / 'reports/ease_insights_20261103/ae_complement_beam_v2'


def freeze():
    path = REPORT / 'protocol.json'
    if not path.exists():
        proxy = proxy_evidence()
        independent = ROOT / 'reports/ease_insights_20261103/ae_complement_beam_v1/independent_audit.json'
        audit = json.loads(independent.read_text())
        require(audit['status'] == 'pass' and audit['selected']['masks'] == [[2, 13], [7, 14]], 'Independent proxy reconstruction required')
        value = dict(revision='ease-AE16-joint-mask-world-v2', status='prospective_before_new_pair_world_outcomes', time=time.time(),
            proxy_selection=proxy, independent_proxy_reconstruction=dict(path=str(independent), sha256=sha(independent)),
            conditions=conditions(), preregistered_directions=DIRECTIONS,
            rationale='The complete153-mask fixed-budget offline screen selected [2,13] versus [7,14]. Object/Spatial error preferences are a heuristic; new closed-loop outcomes must establish whether preferences transfer to success.',
            fixed_science=dict(H=50, R=5, Full_D=10, Fast_D=2, Full_AE=18, Fast_AE=16, VLM_layers=18, vision_layers=27,
                fresh_both_cameras=True, refresh_period=2, cycle='FullD10 then two FastD2 requests', training=False,
                residual_compensation='Unused with empty VLM masks; established own-Full timestamps and receipts retained.',
                sampler='Unmodified Pi0Adapter.sample, native eager arithmetic; observed helper dispatch bypasses two AE layers only on FastD2'),
            calibration=dict(episodes=8, independent_quality_cases=0, conditions=['full_d10', 'ae16_d2'], tasks=[0], inits=[45],
                requirement='Native no-bypassD2/D10 raw/decoded parity1e-5; actual ordered layer visits/NFE and full prefix, period/reset; all eight known Full/AE16 controls bitwise reproduce their two original roots.',
                fixed_cost='All four definitions on each service, four fixed suite inputs,6 warmups plus12 measured requests; whole three-request native policy cycles.'),
            discovery=dict(new_conditions=list(NEW_CONDITIONS), new_episodes=160, original_control_cells=160, analytical_cells=320,
                tasks=list(range(10)), inits=[45,46], original_controls=REFERENCES,
                qualification='Both new paths overall>=.75 and>=Full-.15, each completed-service median native fixed-input cost ratio>=1.15, pair costs within10%; both fixed proxy directions gap>=10pp,>=2 positive tasks, every task LOO positive.',
                direction_selection='Exactly path_a>path_b onObject and path_b>path_a onSpatial. No suite/path reselection, no promotion of another screened mask, and no use of known AE16 benchmark as a selectable alternative.'),
            confirmation=dict(episodes=360, inits=[47,48,49], conditions=['full_d10',*NEW_CONDITIONS], directions=DIRECTIONS,
                primary='One-sided exact ten-task signflip including ties; Holm2 for exactly the two fixed directions.',
                bootstrap_replicates=20000, bootstrap_seed=20261109, bootstrap_arithmetic='Integer paired/task difference sums before division; linear percentile quantiles.',
                success='Both differences positive, both task-signflip Holm2p<.05, both whole-task-bootstrap lower bounds>0 and all task LOO positive. Both paths also pass the same quality/native-cost gates.',
                no_reselection=True),
            physical_contract=dict(policy_action_limits=LIMITS, settling_actions=10, environment_seed=7),
            total_new_episode_cap=528,
            independence='init45–46 and original Full/AE16 controls are known discovery cases, not new controls or confirmation. Offline observations use historical initial/phase inputs and are not heldout quality results.',
            no_candidate='Preserve all valid outcomes, keep confirmation locked, do not lower gates; continue the overall goal with a new bounded branch if warranted.',
            limits='Suite-conditioned fixed paths only; no online phase-selection, runtime-routing gain, universal category inference, exclusive-GPU speed, or study-wide5% family-error claim.')
        atomic_json(path, value)
    return dict(path=str(path), sha256=sha(path))
