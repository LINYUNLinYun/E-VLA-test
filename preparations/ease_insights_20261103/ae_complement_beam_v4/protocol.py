"""Freeze the post-hoc pair before joint-cost calibration and untouched-state inference."""
import json
import time
from .common import *

REPORT=ROOT/'reports/ease_insights_20261103/ae_complement_beam_v4'


def freeze():
    target=REPORT/'protocol.json'
    if not target.exists():
        source=ROOT/'reports/ease_insights_20261103/cross_module_completed_audit_v1/audit.json';audit=json.loads(source.read_text())
        require(audit['status']=='completed_source_discovery_audit_passed' and audit['success_discovery_gates_pass'] and audit['new_quality_cases']==0,'Need source-paired post-hoc discovery evidence')
        previous=ROOT/'reports/ease_insights_20261103/ae_complement_beam_v3/completed_summary.json';require(json.loads(previous.read_text())['status']=='completed_no_candidate','Current neighbor search must end without confirmation first')
        value=dict(revision='ease-cross-module-v4',status='prospective_before_joint_costs_and_fresh_outcomes',time=time.time(),
            historical_discovery_evidence=dict(path=str(source),sha256=sha(source)),previous_branch=dict(path=str(previous),sha256=sha(previous)),
            post_hoc_cross_branch_discovery=True,conditions=conditions(),reference_origins=REFERENCES,preregistered_hypotheses=DIRECTIONS,
            discovery='Completed known cases only: Full77/80,VLM_c70/80,AE14D3 68/80. Spatial AE14 exceedsVLM_c by3/20;Long VLM_c exceedsAE14 by6/20. All240 source cases were reconstructed and80 initial inputs/noise/physical identities paired. These are not new replications.',
            budget_scope='VLM12/AE18/D2 and VLM18/AE14/D3 have different retained layers/NFE. Matching is based exclusively on new joint native-policy costs,not layer counts or separate historical medians.',
            calibration=dict(episodes=12,independent_quality_cases=0,conditions=list(CONDITIONS),tasks=[0],inits=[45],noise_repeat=0,
                numerical='Unmodified adapter no-bypassD2/D3/D10 parity1e-5; VLM residuals,actual AE visits/NFE,time nodes,period/reset; all12 source trajectories bitwise identical.',
                joint_cost='All three definitions in each of six completed services; four real bank inputs,6 warmups and12 measured requests each. Median per-service Full/path policy ratios both>=1.15; max/min of the two ratios<=1.10. No gate relaxation after timing.'),
            fresh_confirmation=dict(official_initials=[47,48,49],tasks_per_suite=10,suites=4,conditions=list(CONDITIONS),
                noise_repeats=[0,1],episodes_per_noise_repeat=360,total_episodes=720,unique_physical_case_states_per_algorithm=120,
                explanation='The same three new official states are deliberately evaluated under two fixed policy-noise streams. This is240 rollouts per algorithm,not240 independent physical initial states. All six observations within each task remain one primary cluster.',
                seed_rule='repeat0 uses the original condition-independent seed. repeat1 is the first little-endian32 bits of SHA256(ease-cross-module-v4/noise-repeat1/{original_seed}). Order and algorithm ID never affect noise.',
                primary='Exactly two fixed directions. One-sided exact signflip over ten whole-task sums of six paired outcomes,including zero ties;Holm2.',
                bootstrap='20,000 resamples of ten whole-task integer sums divided by60;seed20261111;linear percentile quantiles. Case resampling is a separately labeled sensitivity.',
                success='Both direction differences positive,primary Holm2p<.05,both whole-task bootstrap lower>0,every task LOOpositive. Both algorithms globally>=.75 and>=Full-.15; each joint native-policy speed ratio>=1.15 and pair costs within10%.',
                no_reselection=True,no_more_rollouts_after_significance_inspection=True),
            known_power_boundary='Historical Spatial differences occur in2/10 tasks andLong in5/10. Strong case-level repetition alone will not override a failed task-cluster primary verdict.',
            fixed_science=dict(H=50,R=5,Full_D=10,period=2,vision_layers=27,cameras=2,fresh_cameras=True,training=False,
                VLM_c='Skip[4,7,10,13,15,17] with the existing residual extrapolation;AE18/D2.',AE14='FullVLM18;skipAE[2,6,7,8] onFastD3;empty VLM mask leaves residual compensation unused.'),
            physical_contract=dict(environment_seed=7,settling_actions=10,policy_action_limits=LIMITS),
            maximum_new_physical_rollouts=732,
            stop='If joint calibration/costs fail,do not collect confirmation. If720 fresh rollouts fail either direction or quality/cost gate,preserve not_reproduced. Do not change masks,suites,seed streams,primary tests or thresholds.',
            limits='Post-hoc historical discovery followed by new preregistered validation;no global adaptive-routing-necessity,phase-switching,closed-loop acceleration or study-wide5% error claim. Dense/AE16 alternatives can still be better fixed routes.')
        atomic_json(target,value)
    return dict(path=str(target),sha256=sha(target))
