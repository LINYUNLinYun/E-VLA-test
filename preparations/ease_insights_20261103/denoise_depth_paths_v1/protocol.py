"""Prospective AE depth/time experiment; no new joint outcome used for design."""
import json
import time
from .common import ROOT,conditions,sha,atomic_json,NEW_CONDITIONS,REFERENCES

REPORT=ROOT/'reports/ease_insights_20261103/denoise_depth_paths_v1'


def freeze():
    path=REPORT/'protocol.json'
    if not path.exists():
        previous=ROOT/'reports/ease_insights_20261103/camera_paths_v3/completed_summary.json'
        proposal=ROOT/'reports/ease_insights_20261103/denoise_depth_paths_v1_preparation/proposal.json'
        value=dict(revision='ease-denoise-depth-v1',status='prospective_before_new_depth_time_GPU_outcomes',time=time.time(),
            conditions=conditions(),previous_branch=dict(path=str(previous),sha256=sha(previous)),proposal=dict(path=str(proposal),sha256=sha(proposal)),
            motivation='With fixedD2 the prior AE16/14/12 paths yielded74/64/24 successes out of80. This branch tests whether more Euler updates change the task preference of shallower action experts. Quality gains or complementarity are not assumed.',
            fixed_science=dict(H=50,R=5,Full_D=10,Full_AE_layers=18,vision_layers=27,cameras=2,VLM_layers=18,
                fresh_both_cameras=True,period=2,cycle='FullD10 then two configured Fast requests',training=False,
                residual_compensation='Unused because every VLM mask is empty. Periodic own-Full timestamps preserve the existing scheduling/receipt contract.',
                sampler='Unchanged Pi0Adapter.sample; original-adapter uniform EulerD2/D3/D4 and released fusedD10. Original sampler/kernel sources remain unchanged.'),
            calibration=dict(episodes=8,conditions=['full_d10','ae16_d2'],tasks=[0],inits=[45],suites=4,cases_previously_used=True,independent_quality_cases=0,
                requirement='Original unwrapped adapter AE18 versus observed wrapper on D2/D3/D4/D10, raw/decoded1e-5; actual AE visits/NFE and exact Euler nodes; full VLM prefix/period/reset; reproduce all eight known Full/AE16 action/raw trajectories bitwise across both original roots.',
                cost='Six conditions,four old-bank suite inputs,6 warmup and12 measured requests each; completed three-request cycle cost, not layer-count equivalence'),
            reference_origins=REFERENCES,
            discovery=dict(new_conditions=list(NEW_CONDITIONS),new_episodes=320,reference_episodes=160,analytical_cells=480,inits=[45,46],tasks=list(range(10)),
                reuse='Original Full80 and AE16D2 80 have different directories/manifests. Preserve every source identity and alias only in derived analysis; these160 cells are not new replications.',
                selection='After all320 new and160 original cells accepted: both overall>=.75 and>=Full-.15; each completed-service median fixed-input cost ratio>=1.15; pair cost within10%; opposite distinct-suite gaps>=.10,>=2 positive tasks,each task LOO positive.',
                ranking='Maximum minimum directional count gap,sum gaps,minimum overall successes,then lexical path/suite ids. Freeze one pair only; discovery p-values remain descriptive.'),
            confirmation=dict(episodes=360,inits=[47,48,49],conditions='Full and exactly two frozen selected paths',
                primary='One-sided exact ten-task signflip including ties; Holm2 across exactly two frozen directions',bootstrap_replicates=20000,bootstrap_seed=20261108,quantile_method='linear',
                bootstrap_arithmetic='Integer paired/task difference sums before division, preserving the exact zero boundary',
                verdict='Both directions positive,primary Holm2p<.05,whole-task bootstrap lower>0,each task LOOpositive; both paths overall>=.75 and>=Full-.15; each cost ratio>=1.15 and pair within10%.',no_reselection=True),
            physical_contract=dict(policy_action_limits={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520},settling_actions=10,environment_seed=7),
            new_episode_cap=688,limits='Suite-conditioned periodic paths only; no universal semantic/phase/runtime-selection claim, no exclusive-GPU or closed-loop acceleration claim. Logical layer visits are not FLOPs or latency. Multiple exploratory branches do not imply study-wide5% error control.',
            no_candidate='Preserve all valid outcomes and do not lower gates or run confirmation. The bounded search ending does not complete the overall goal.')
        atomic_json(path,value)
    return dict(path=str(path),sha256=sha(path))


if __name__=='__main__':print(json.dumps(freeze()))
