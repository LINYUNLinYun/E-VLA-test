"""Prospective fixed attention-versus-FFN paths, before GPU/outcomes."""
import json
import time
from .common import *
REPORT=ROOT/'reports/ease_insights_20261103/camera_paths_v7'


def freeze():
    path=REPORT/'protocol.json'
    if path.exists():return dict(path=str(path),sha256=sha(path))
    prior=ROOT/'reports/ease_insights_20261103/camera_paths_v6/completed_summary.json'
    require(json.loads(prior.read_text())['status']=='complete_no_candidate','Preserve completed image-edge branch first')
    historical=ROOT/'reports/ease_insights_20261103/historical_policy_initials_v1/audit.json'
    value=dict(revision='ease-fixed-VLM-component-paths-v7',status='prospective_before_real_GPU_or_new_outcomes',time=time.time(),conditions=conditions(),
        antecedent_evidence=dict(path=str(prior),sha256=sha(prior)),historical_inventory=dict(path=str(historical),sha256=sha(historical)),
        hypothesis='Attention and FFN hidden updates may support different task categories. Compare fixed same-layer component bypasses with current-observation KV,fresh vision,AE schedule and equal residual compensation.',
        fixed_science=dict(H=50,R=5,vision=[27,27],VLM_KV_layers=18,AE_layers=18,Full_D=10,Fast_D=2,period=2,
            components=dict(COMPONENTS),fixed_original_layers=list(FIXED_LAYERS),
            layer_choice='Reuse spread_c original six skip positions; no new layer-mask search or old module quality-based selection.',
            Full_capture='Selected component at all17 hidden-update layers; retain only six selected FP32 residual tensors per bank; last two Full banks.',
            Fast_patch='At each selected module slot use latest residual plus elapsed/gap times latest-minus-older. First bank uses carry. Attention patch precedes same-layer FFN.',
            current_observation_KV=True,final_layer='Original L18 QKV-only',training=False,camera_feature_cache=False,selector_calls=0),
        numerical=dict(native_and_capture_identity=dict(atol=1e-5,rtol=1e-5),partial_full_compute_restore_reference=dict(atol=1e-5,rtol=1e-5,decoded_gripper_all50_exact=True),
            structural='Production omits selected attention update plus Q or selected FFN; reference runs all18QKV,17attention,17FFN,restores selected updates and Torch-patches; actual helper restoration,NFE,cadence and residual identity required.'),
        calibration=dict(episodes=8,independent_quality_cases=0,conditions=['full_d10','dense_d2'],tasks=[0],inits=[45],
            bridge='All8 known R5 Full/DenseD2 raw/decoded trajectories bitwise identical to original controls.',
            joint_cost='All4 definitions in each of4 completed services;4 real bank inputs,6warmup+12measured requests per definition/input. Median same-service Full/path>=1.15 and pair max/min<=1.10. Capture,patch,vision,decode/transfer included.'),
        discovery=dict(inits=[45,46],all_tasks=40,new_conditions=list(NEW_CONDITIONS),new_rollouts=160,preserved_Full_and_DenseD2=160,analytical_cells=320,
            quality='Each candidate overall>=.75 and >=Full-.15',selection='Fixed pair only; each opposite suite gap>=.10,at least4 positive tasks per direction,every task LOO positive. Rank minimum gap,sum gaps,then lexical suites. Freeze after all320 cells and independent audit.',no_discovery_significance_claim=True),
        confirmation=dict(inits=[10,11,12,13,14],noise_repeat=0,all_tasks=40,conditions=list(CONDITIONS),new_rollouts=800,
            primary='Two discovery-frozen directions;one-sided exact10-task signflip including ties;Holm2. Five physical initials remain inside their task.',bootstrap='20000 whole-task integer-sum resamples/50;seed20261121;linear percentile.',
            success='Both directions positive,Holm2p<.05,both task-CI lower>0,all task LOO positive;both quality>=.75 and>=Full-.15;nativeFull/path>=1.15 and pair within10%.',no_reselection=True,no_added_rollouts_after_significance_inspection=True),
        state_exposure_scope='All official0-49 states have historical R12 exposure. Confirmation10-14 is separate from this family45-46 discovery, not globally unseen states/tasks or training-OOD. No confirmation outcomes may select components,layers or directions.',
        physical_contract=dict(environment_seed=7,settling_actions=10,policy_action_limits=LIMITS),maximum_physical_rollouts_including_repeated_controls=968,
        DenseD2_boundary='Report fixed DenseD2 quality and native cost. If it dominates both candidates do not claim routing benefit. Complementarity and routing benefit are separate questions.',
        budget_boundary='Six attention skips and six FFN skips have different operator costs; equal component counts are not equal FLOPs. Real native cost matching is mandatory before quality.',
        cohort_identifier='camera_paths_v7 is an existing opaque resource-queue identifier; scientific intervention is VLM component bypass, not camera pruning.',
        stop='Reject numeric/control/cost failures before quality. No qualified discovery pair means no confirmation. Failed confirmation remains not_reproduced; no effective-failure reruns or relaxed thresholds.',
        overall_goal_complete_claim=False,limits='Exploration then fixed-pair replication;no study-wide5percent error claim,online phase selector or closed-loop acceleration claim.')
    atomic_json(path,value);return dict(path=str(path),sha256=sha(path))
