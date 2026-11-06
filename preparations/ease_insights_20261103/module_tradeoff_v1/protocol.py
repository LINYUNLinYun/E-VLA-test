"""Prospective module experiment and explicit provenance for reused controls."""
import json
from pathlib import Path
import time
from .common import ROOT,PATHS,NEW_CONDITIONS,conditions,sha,atomic_json

REPORT=ROOT/'reports/ease_insights_20261103/module_tradeoff_v1'


def freeze():
    path=REPORT/'protocol.json'
    if not path.exists():
        previous=ROOT/'reports/ease_insights_20261103/vlm_diversity_v2/completed_summary.json'
        ae=ROOT/'reports/ease_insights_20261103/ae_existing_evidence_inventory/offline_layer_affinity/analysis.json'
        value=dict(revision='ease-module-tradeoff-v1',status='prospective_before_new_module_GPU_outcomes',time=time.time(),
            conditions=conditions(),previous_branch=dict(path=str(previous),sha256=sha(previous)),
            proposal_basis=dict(VLM_c='The only qualified path in the completed broad-mask discovery,70/80; explicitly selected using known previous outcomes.',
                AE_layers='Nested two/four/six skips from the historical global low first5-command NMSE ranking; single-layer proxy only, not assumed composable or task-specific.',
                AE_evidence=dict(path=str(ae),sha256=sha(ae)),hybrid='Three VLM skips from the old Jac mask plus three low global-error AE skips; no measured task preference assumed.'),
            calibration=dict(episodes=4,inits=[45],tasks=[0],suites=['libero_spatial','libero_object','libero_goal','libero_10'],
                condition='full_d10',cases_previously_used=True,independent_quality_evidence=False,
                required='Actual native no-AE-bypass raw/decoded D10 and D2 parity on four bank inputs; direct AE visits and actualNFE; all four Full trajectories must bridge preserved Full actions/outcomes before any quality-reference reuse.',
                profile='All six fixed configurations measured on the original four-suite input bank,6 warmup and12 measured rounds each; descriptive native policy cost only'),
            discovery=dict(new_conditions=list(NEW_CONDITIONS),new_episodes=320,inits=[45,46],tasks=list(range(10)),
                reused_conditions={'full_d10':'old full_d10','vlm_c':'old spread_c'},reused_episodes=160,analytical_case_cells=480,
                reference_scope='Prior original records/manifests remain unchanged and retain source identities. This is exploratory reuse of known discovery controls, never a new160-case replication. Only reuse after a source/numerical/physical-input bridge passes.',
                pair_eligibility='Both overall>=.75 and >=Full-.15; each median completed-service native cost ratio>=1.15; pair measured mean-cycle cost within10%; opposite distinct-suite gaps>=.10, >=2 positive tasks and every task leave-one-out positive.',
                ranking='Maximum minimum directional gap, sum gaps, minimum overall successes, then lexical path/suite ids; freeze one pair only after all320 new and160 preserved cases accepted'),
            confirmation=dict(episodes=360,inits=[47,48,49],conditions='Full and exactly two frozen paths',
                primary='One-sided exact ten-task sign-flip including ties, Holm2 across exactly two frozen directions',
                bootstrap_replicates=20000,bootstrap_seed=20261106,quantile_method='linear',
                verdict='Both frozen directions positive; primary Holm2 p<.05; whole-task bootstrap lower>0; every task leave-one-out positive; both paths>=.75 and>=Full-.15; measured pair cost within10% and each ratio>=1.15',
                no_reselection=True),new_episode_cap=684,
            fixed_science=dict(H=50,R=5,Full_D=10,Fast_D=2,period=2,Full_AE_layers=18,vision_layers=27,
                Full_refresh='FullVLM18/AE18 every third request',VLM_compensation='Own-path residual extrapolation; empty VLM mask still has periodic timestamps',
                policy_action_limits={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520},settling_actions=10,environment_seed=7),
            cost_warning='VLM/AE layer counts do not represent equal computation. Pair matching uses measured completed-policy cost, not layer counts or FLOP estimates.',
            limits='Suite-conditioned effects only; no universal semantic-category or same-task phase claim, no exclusive-GPU timing guarantee; many exploratory branches do not provide a study-wide5% error guarantee.',
            no_candidate='Do not launch confirmation or weaken gates; preserve this bounded result and continue the overall goal separately')
        atomic_json(path,value)
    return dict(path=str(path),sha256=sha(path))


if __name__=='__main__':print(json.dumps(freeze()))
