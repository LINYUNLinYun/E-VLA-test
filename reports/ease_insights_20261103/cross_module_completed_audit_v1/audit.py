"""Post-hoc completed-branch discovery only; no current partial outcomes are read."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
SOURCES={
 'full_d10':('vlm_diversity_v2','full_d10','a4a996b03059874ceb10a21e1099985508051d7ae0392af944e739911cd1494b'),
 'vlm_c':('vlm_diversity_v2','spread_c','a4a996b03059874ceb10a21e1099985508051d7ae0392af944e739911cd1494b'),
 'ae14_d3':('denoise_depth_paths_v1','ae14_d3','df34a6e280492733ee57862368cd41fdbd54cb114ae385feb06d0fc6bb076ddf')}


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def main():
    tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Original case validator changed')
    spec=importlib.util.spec_from_file_location('cross_module_completed_validator',tool);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    counts={cid:{s:[0]*10 for s in SUITES} for cid in SOURCES};paired={};evidence=[];algorithms={};source_manifests=[];first_manifest=None
    for alias,(family,cid,pin) in SOURCES.items():
        origin=ROOT/'results/ease_insights_20261103'/family/'discovery';require(sha(origin/'manifest.json')==pin,'Original raw manifest changed')
        cm=importlib.import_module('preparations.ease_insights_20261103.'+family+'.common');w=importlib.import_module('preparations.ease_insights_20261103.'+family+'.worker');m=cm.read_manifest(origin/'manifest.json',verify_sources=True)
        if first_manifest is None:first_manifest=m
        else:
            require(m['checkpoints']==first_manifest['checkpoints'] and m['noise']==first_manifest['noise'],'Checkpoint/noise contracts differ')
            for field in ['H','R','refresh_D','settling_actions','environment_seed','policy_action_limits','training']:
                require(m['science'][field]==first_manifest['science'][field],'Physical/science contract differs: '+field)
        config=next(c for c in m['conditions'] if c['id']==cid);algorithms[alias]=dict(config,source_condition=cid,source_manifest_hash=m['manifest_hash'],source_root=str(origin))
        source_manifests.append(dict(alias=alias,path=str(origin/'manifest.json'),sha256=pin,manifest_hash=m['manifest_hash']))
        for s in SUITES:
            for task in range(10):
                for initial in [45,46]:
                    scene=dict(suite=s,task=task,init=initial);validator.case(cm,w,origin,m,cid,scene);p=w.episode_dir(origin,cid,scene)/'episode_receipt.json';r=read(p)
                    require(r['condition']==cid and r['manifest_hash']==m['manifest_hash'] and type(r['success']) is bool,'Outcome source identity differs')
                    key=(s,task,initial);gate=r['initial_gate'];identity={k:gate[k] for k in ['input_hash','seed','physical_assets']}
                    if key in paired:require(identity==paired[key],'Original physical first input or noise differs')
                    else:paired[key]=identity
                    counts[alias][s][task]+=int(r['success']);evidence.append(dict(path=str(p),sha256=sha(p),analysis_alias=alias,source_condition=cid,source_manifest_hash=m['manifest_hash']))
    total={c:sum(sum(v) for v in by_suite.values()) for c,by_suite in counts.items()}
    qualified={c:total[c]>=60 and total['full_d10']-total[c]<=12 for c in ['vlm_c','ae14_d3']}
    directions=[]
    for suite,better,worse in [('libero_spatial','ae14_d3','vlm_c'),('libero_10','vlm_c','ae14_d3')]:
        delta=[a-b for a,b in zip(counts[better][suite],counts[worse][suite])];n=sum(delta)
        directions.append(dict(suite=suite,better=better,worse=worse,difference_count=n,difference=n/20,task_difference_counts=delta,
            positive_tasks=sum(v>0 for v in delta),every_task_LOO_positive=all(n-v>0 for v in delta),discovery_direction_pass=n>=2 and sum(v>0 for v in delta)>=2 and all(n-v>0 for v in delta)))
    cost_sources=[];ratios={}
    for alias,family in [('vlm_c','module_tradeoff_v1'),('ae14_d3','denoise_depth_paths_v1')]:
        p=ROOT/'results/ease_insights_20261103'/family/'discovery/analysis/analysis.json';r=read(p);require(r['complete'] is True,'Need complete historical cost analysis')
        ratios[alias]=r['selection']['condition_eligibility'][alias]['median_fixed_input_speed_ratio'];cost_sources.append(dict(alias=alias,path=str(p),sha256=sha(p)))
    value=dict(status='completed_source_discovery_audit_passed',time=time.time(),new_quality_cases=0,source_cases=240,paired_initials=len(paired),
        algorithms=algorithms,source_manifests=source_manifests,total_successes=total,suite_successes={c:{s:sum(v) for s,v in by_suite.items()} for c,by_suite in counts.items()},
        task_success_counts=counts,quality_qualification=qualified,directions=directions,
        success_discovery_gates_pass=all(qualified.values()) and all(d['discovery_direction_pass'] for d in directions),
        preliminary_separate_service_cost_ratios=ratios,preliminary_pair_cost_factor=max(ratios.values())/min(ratios.values()),cost_sources=cost_sources,
        joint_same_service_cost_gate_passed=False,cost_scope='Separate completed historical service groups; normalizedFull ratios suggest a candidate only. A new joint fixed-input cost gate is required before acceptance.',
        same_layer_count_or_NFE_claim=False,post_hoc_cross_branch_selection=True,independent_validation=False,insight2_complete=False,
        current_AE14_partial_outcomes_read=False,current_frozen_AE14_experiment_changed=False,original_scientific_records_modified=False,evidence=evidence,
        source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        next='Finish the current320-case neighbor search. Preserve its own decision; if an independently versioned cross-module follow-up is warranted, freeze masks/directions, pass joint numerical/known-control/cost gates, and validate on untouched official states before claiming reproducibility.')
    output=Path(__file__).with_name('audit.json');output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k not in ['evidence','task_success_counts','algorithms']}))


if __name__=='__main__':main()
