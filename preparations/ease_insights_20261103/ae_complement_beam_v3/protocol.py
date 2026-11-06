"""Direct same-budget layer-neighbor discovery after a failed offline proxy pair."""
import json
import time
from .common import *

REPORT=ROOT/'reports/ease_insights_20261103/ae_complement_beam_v3'


def freeze():
    path=REPORT/'protocol.json'
    if not path.exists():
        previous=ROOT/'reports/ease_insights_20261103/ae_complement_beam_v2/completed_summary.json'
        prior=json.loads(previous.read_text())
        if prior['status']!='completed_no_candidate' or prior['condition_counts']['path_a']!=60 or prior['condition_counts']['path_b']!=62:
            raise ValueError('Previous proxy-selected pair must be preserved as negative')
        value=dict(revision='ease-AE14-neighbor-v3',status='prospective_before_new_neighbor_world_outcomes',time=time.time(),
            previous_branch=dict(path=str(previous),sha256=sha(previous)),conditions=conditions(),reference_origins=REFERENCES,
            rationale='The AE14/D2 offline error preferences failed to predict Object/Spatial success and both paths lost Long quality. Use the already usable AE14/D3 anchor68/80 and test four one-layer swaps directly in closed loop, keeping Fast depth/NFE fixed. Do not use offline MSE as task-success evidence.',
            neighborhood=dict(anchor_skips=[2,6,7,8],replacement_layers=[4,9],
                swaps=dict(path_a=dict(restore=6,skip=4),path_b=dict(restore=7,skip=4),path_c=dict(restore=8,skip=4),path_d=dict(restore=8,skip=9)),
                reason='Layers4 and9 are the next omissions in the existing nested AE12 mask [2,4,6,7,8,9]. Restoring one anchor omission while skipping one of these keeps14 layers. This structural choice does not assume task preferences.'),
            fixed_science=dict(H=50,R=5,Full_D=10,Full_AE=18,Fast_D=3,Fast_AE=14,VLM_layers=18,vision_layers=27,
                cameras=2,fresh_both_cameras=True,period=2,cycle='FullD10 then two AE14/D3 requests',training=False,
                sampler='Unmodified original Pi0Adapter.sample uniform EulerD3; no released-native-D3-kernel equivalence claim.',
                compensation='Unused with empty VLM masks; existing own-Full timestamps retained.'),
            calibration=dict(episodes=8,conditions=['full_d10','ae14_base'],tasks=[0],inits=[45],independent_quality_cases=0,
                requirement='Original adapter no-bypass parity1e-5,actual layer visits/NFE/time nodes,full prefix/period/reset; eight known Full/AE14D3 trajectories bitwise identical across two source roots.',
                fixed_cost='All six definitions on each service; four real bank inputs,6 warmups plus12 measured requests per condition, whole three-request policy cycles.'),
            discovery=dict(new_conditions=list(NEW_CONDITIONS),new_episodes=320,original_control_cells=160,analytical_cells=480,inits=[45,46],tasks=list(range(10)),
                candidates='Four new one-swap paths and the unchanged usable AE14/D3 anchor. Full is a quality/cost reference, not a selectable fast path.',
                qualify='Both overall>=.75 and>=Full-.15; each median completed-service Full/path native cost ratio>=1.15, pair costs within10%; opposite distinct-suite success gaps>=10pp,>=2 positive tasks per direction, all task LOO positive.',
                ranking='Maximum minimum directional count gap,then sum gaps,then minimum overall successes,then lexical path/suite ids. Freeze exactly one pair after all320 new and160 original records are validated.',
                reuse='Original Full80 and AE14D3 80 retain actual source identities. Known init45–46 discovery states are not independent confirmation.'),
            confirmation=dict(episodes=360,inits=[47,48,49],conditions='New Full and exactly two frozen paths',
                primary='One-sided exact ten-task signflip including ties; Holm2 for exactly two frozen directions.',
                bootstrap_replicates=20000,bootstrap_seed=20261110,quantile_method='linear',integer_sums_before_division=True,
                success='Both differences positive,task-signflip Holm2p<.05,whole-task bootstrap lower>0,all task LOOpositive; both quality/native-cost gates and10% pair cost gate pass.',no_reselection=True),
            physical_contract=dict(policy_action_limits=LIMITS,settling_actions=10,environment_seed=7),new_episode_cap=688,
            no_candidate='Preserve the bounded negative,keep confirmation locked,never lower scientific gates. This branch ending does not complete the overall goal.',
            limits='Suite-level fixed-path comparison only. No phase detector,online routing gain,exclusive-GPU or closed-loop acceleration claim. Multiple exploratory branches do not imply study-wide5% error control.')
        atomic_json(path,value)
    return dict(path=str(path),sha256=sha(path))
