"""Prospective camera-module hypothesis; freeze before any camera GPU outcome."""
import json
import time
from .common import ROOT,conditions,sha,atomic_json,NEW_CONDITIONS,PAIR_PATHS

REPORT=ROOT/'reports/ease_insights_20261103/camera_paths_v3'


def freeze():
    path=REPORT/'protocol.json'
    if not path.exists():
        prior=ROOT/'reports/ease_insights_20261103/module_tradeoff_v1/completed_summary.json'
        rejected=ROOT/'reports/ease_insights_20261103/camera_paths_v1/calibration_rejection.json'
        validator=ROOT/'reports/ease_insights_20261103/camera_paths_v2/D5_validator_block.json'
        value=dict(revision='ease-camera-paths-v3',status='prospective_before_corrected_camera_GPU_outcome',time=time.time(),
            earlier_numerical_rejection=dict(path=str(rejected),sha256=sha(rejected)),
            earlier_passed_numerics_but_unlaunched_discovery=dict(path=str(validator),sha256=sha(validator)),
            receipt_correction='A separately versioned exact D5/D10 Euler-node validator replaces the old D2/D10-only receipt helper; sampler, camera arithmetic, conditions, tolerance and research gates are unchanged. New GPU gates also check node receipts on every finite-bank policy request.',
            technical_correction='Batch1 reuses unmodified native batch2 two-part projection kernels with seq_len256, preserving BF16 partial-rounding order. The failed generic-batch1 implementation and its records remain unchanged. All same1e-5 gates must pass; no quality episodes from v1 exist.',
            conditions=conditions(),previous_branch=dict(path=str(prior),sha256=sha(prior)),
            hypothesis='External-view and wrist-view freshness may matter differently across task suites; no measured preference assumed.',
            proposal_basis='The completed module branch produced two usable paths but AE16 was no worse in every suite. Camera branches now change the information refreshed, at equal nominal and measured cost.',
            design=dict(Full='D10,27-layer vision for both views,VLM18,AE18 on every Full request',
                Fast='D5,VLM18,AE18; one camera runs native27-layer vision, the other holds raw features from its own most recent Full',
                primary_pair=list(PAIR_PATHS),both_fresh_control='D5 and same refresh schedule, both native camera views recomputed on every Fast; excluded from primary pair selection',
                refresh_period=4,cycle='Full + four Fast requests',H=50,R=5,
                motivation='D5 protects action-generation quality while four Fast requests expose camera-specific freshness over5/10/15/20 executed actions. This is an exploratory design choice, not a previously measured improvement.',
                cache='Periodic paths clone raw pre-projector two-view features at each own Full; hold only the other view; never update cache from Fast, another condition, prompt or episode. The always-Full baseline creates no unused feature cache.',
                implementation='Native num_views argument is unused. Real input/scratch tensors use batch1; original native kernels and the released batch2 projection arithmetic are retained with seq_len256; camera0=agentview,camera1=wrist.'),
            calibration=dict(episodes=4,inits=[45],tasks=[0],suites=['libero_spatial','libero_object','libero_goal','libero_10'],condition='full_d10',
                independent_quality_evidence=False,cases_previously_used=True,
                required='Native unwrapped D5/D10 vs AE18-wrapped parity; actual AE18 visits/NFE; real vision batch1; same-observation raw/prefix/action parity; nontrivial mixed-view oracle raw/prefix/action parity; own-Full cache age and reset; four Full trajectories bitwise bridge preserved Full.',
                atol=1e-5,rtol=1e-5,profile='All four configurations on four original bank prompts,10 warmup and20 measured requests each; whole five-request cycles.'),
            discovery=dict(new_conditions=list(NEW_CONDITIONS),new_episodes=240,inits=[45,46],tasks=list(range(10)),
                preserved_reference_conditions={'full_d10':'original full_d10'},preserved_reference_episodes=80,analytical_cells=320,
                reference_scope='Original Full records/manifests and receipts retain identities. These80 known cases are exploratory controls, not80 new replications. Reuse requires accepted numeric/trajectory/physical-input bridge.',
                pair_eligibility='Only agent_fresh_d5/wrist_fresh_d5; both overall>=.75 and>=Full-.15; each median completed-service native cost ratio>=1.15; measured pair mean-cycle cost within10%; opposite distinct-suite gaps>=.10, >=2 positive tasks and every task leave-one-out positive.',
                ranking='Maximum minimum directional gap, sum gaps, minimum overall successes, then lexical path/suite ids. Freeze directions only after all240 new and80 preserved cells accepted.'),
            confirmation=dict(episodes=360,inits=[47,48,49],conditions=['full_d10',*PAIR_PATHS],
                primary='One-sided exact ten-task sign-flip including ties, Holm2 across exactly two frozen directions',
                bootstrap_replicates=20000,bootstrap_seed=20261107,quantile_method='linear',
                verdict='Both directions positive; each Holm2 p<.05, whole-task bootstrap lower>0 and every task LOO positive; both paths>=.75 and>=Full-.15; each native cost ratio>=1.15 and pair within10%.',
                no_reselection=True),new_episode_cap=604,
            physical_contract=dict(policy_action_limits={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520},settling_actions=10,environment_seed=7,control_hz=20.),
            limits='Suite-conditioned periodic camera-path evidence only; no semantic-category generality, no within-task phase claim, no standalone path-quality or adaptive-choice benefit established. Shared GPU timing is descriptive; multiple exploratory branches do not imply study-wide5% error control.',
            no_candidate='Preserve the bounded negative result; do not launch confirmation, weaken gates or reinterpret the both-fresh control as the prespecified camera pair. Continue the overall goal in a separately specified branch.')
        atomic_json(path,value)
    return dict(path=str(path),sha256=sha(path))


if __name__=='__main__':print(json.dumps(freeze()))
