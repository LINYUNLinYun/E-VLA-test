"""Read-only CPU audit and metadata-selected draft diagnostic plan. No model/env imports."""
import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT/'results/idea1_v3_2_e4_supplement/draft_diagnostics_20261023'
HEADS = {'libero_spatial':'draft_spatial','libero_object':'draft_object',
         'libero_goal':'draft_goal','libero_10':'draft_10'}
CATEGORIES = ('Objects Layout','Camera Viewpoints','Robot Initial States','Language Instructions',
              'Light Conditions','Background Textures','Sensor Noise')
FOCUS = ('Robot Initial States','Language Instructions','Sensor Noise','Background Textures')


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with Path(path).open('w') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)


def csvout(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def legacy_language(name):
    """Execute only the repository's pure filename-language function."""
    path=ROOT/'LIBERO/libero/libero/benchmark/__init__.py'
    node=next(n for n in ast.parse(path.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='grab_language_from_filename')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[node.name](name+'.bddl')


def audit(output=DEFAULT_OUT):
    manifest_path=ROOT/'results/idea1_v3_2_e4_supplement/libero_plus_half_20261022T1216Z/manifest.json'
    ledger_path=ROOT/'reports/idea1_libero_plus_half_final_20261022T2124Z/audit_ledger.json'
    history_path=ROOT/'policies/realtime-vla-flash/results/idea1_v3/e4_expand_init7_10_20261019T1454Z/summaries/with_history/trajectory_index.csv'
    language_path=ROOT/'results/idea1_v3_2_e4_supplement/native_preflight/plus_languages_v1/receipt.json'
    manifest=read(manifest_path);ledger=read(ledger_path)
    jobs={(j['group'],j['source_index']):j for j in manifest['jobs']}
    records=defaultdict(dict)
    for entry in ledger['selected_completed']:
        key=(entry['group'],entry['source_index']);record=read(ROOT/entry['path'])
        assert record['success']==entry['success'] and record['mode']==entry['mode']
        assert record['case']==jobs[key]['case']
        records[key][entry['mode']]=record
    assert len(records)==195 and sum(map(len,records.values()))==585
    languages={(r['mother_suite'],r['name']):r for r in read(language_path)['cases']}
    rows=[]
    for key,by_mode in sorted(records.items()):
        case=jobs[key]['case'];head=case['matching_draft'];draft=by_mode[head]
        language=languages[(case['mother_suite'],case['name'])]
        row=dict(group=key[0],source_index=key[1],head=head,suite=case['mother_suite'],
                 task=case['mother_task'],name=case['name'],category=case['category'],
                 difficulty=case['difficulty_level'],init_index=0,
                 full_success=int(by_mode['full_d10']['success']),
                 v14_success=int(by_mode['v14_d10']['success']),draft_success=int(draft['success']),
                 draft_steps=draft['steps'],legacy_instruction=legacy_language(case['mother_task']),
                 mother_bddl_instruction=language['mother_instruction'],actual_instruction=draft['native_info']['instruction'],
                 language_changed_from_legacy=legacy_language(case['mother_task'])!=draft['native_info']['instruction'],
                 language_changed_from_mother_bddl=not language['equal_to_original_mother'],
                 initial_vector_sha256=draft['native_info']['initial_vector_sha256'],
                 effective_bddl=language['effective_bddl'],gpu_uuid=draft['gpu_uuid'])
        rows.append(row)
    aggregates=[]
    for head in HEADS.values():
        for category in CATEGORIES+('Original unperturbed sanity',):
            levels=range(1,6) if category in CATEGORIES else [None]
            for level in levels:
                group=[r for r in rows if (r['head'],r['category'],r['difficulty'])==(head,category,level)]
                totals=dict(head=head,category=category,difficulty=level,n=len(group))
                for mode in ('full','v14','draft'):
                    totals[mode+'_success']=sum(r[mode+'_success'] for r in group)
                for control in ('full','v14'):
                    for a,b,label in ((0,0,'both_fail'),(0,1,'draft_only'),(1,0,'control_only'),(1,1,'both_success')):
                        totals[control+'_'+label]=sum((r[control+'_success'],r['draft_success'])==(a,b) for r in group)
                aggregates.append(totals)
    history=list(csv.DictReader(history_path.open()));assert len(history)==200
    historical=[]
    for suite,head in HEADS.items():
        all_old=[r for r in history if r['suite']==suite]
        original=[r for r in rows if r['suite']==suite and r['category']=='Original unperturbed sanity']
        task_indices={jobs[(r['group'],r['source_index'])]['case']['task_index'] for r in original}
        matched=[r for r in all_old if int(r['task']) in task_indices]
        historical.append(dict(head=head,old_n=len(all_old),old_success=sum(r['success']=='True' for r in all_old),
            new_original_n=len(original),new_original_success=sum(r['draft_success'] for r in original),
            old_same_task_subset_n=len(matched),old_same_task_subset_success=sum(r['success']=='True' for r in matched),
            new_task_indices=sorted(task_indices),old_init_indices=sorted({int(r['init']) for r in all_old}),
            new_init_indices=[0],paired_same_init=False))
    plan=[]
    for head in HEADS.values():
        for category in FOCUS:
            candidates=sorted((r for r in rows if r['head']==head and r['category']==category),key=lambda r:r['source_index'])
            assert candidates
            selected=candidates[0]
            plan.append(dict(head=head,category=category,source_index=selected['source_index'],
                case=jobs[(selected['group'],selected['source_index'])]['case'],
                legacy_instruction=selected['legacy_instruction'],mother_bddl_instruction=selected['mother_bddl_instruction'],
                actual_instruction=selected['actual_instruction'],
                arms=['actual','mother_bddl','legacy_filename'] if category=='Language Instructions' else ['actual','counterfactual'],
                readiness='same_observation_prompt_only' if category=='Language Instructions' else
                    'same_sim_state_clean_agentview_available' if category=='Sensor Noise' else 'requires_physical_identity_gate'))
    protocol=dict(
        interpretation='Counts are observational; changing tasks/init/settle/prompt together cannot establish perturbation causality.',
        old=dict(init_indices=[6,7,8,9,10],seed=7,settle_steps=10,settle_action=[0]*6+[-1],
                 step_caps=dict(libero_spatial=220,libero_object=280,libero_goal=300,libero_10=520),
                 instruction='benchmark Task.language derived from filename',H=50,R=5,
                 history='E4 branch calls prepare(encode=False) and learned on current observation only; ack bookkeeping does not enter learned input'),
        new=dict(init_index=0,seed='case hash seed',settle_steps=5,settle_action=[0]*7,step_cap=600,
                 instruction='effective BDDL :language',H=50,R=5,
                 history='TransferPolicy learned branch uses current observation only; no trajectory history input'),
        shared=dict(head='explicit mother-suite mapping; same TritonBackend.learned implementation',
                    image='native_observation once; 256x256 two views; same released input/output transforms',
                    weights='shared released base and suite-specific converted draft artifact paths; startup gate checks exact old route; artifact path identity alone is not a historical tensor checksum proof'),
        limits=['Old200 vs new20 are not paired episodes, even when task indices overlap.',
                'Step caps are longer in new protocol; this does not explain failures before historical caps without trajectories.',
                'D2 waits for GPU3. Draft pairs may use GPU0 only after all four heads pass recorded GPU3 real-input raw+decoded bitwise-equality startup checks (strict1e-5 also reported); otherwise wait for GPU3. All new paired arms remain on one device. No inference was run by this CPU audit.',
                'Robot variants require checking actual class init_qpos and problem _load_model base placement. Whether geometry/base changes is unproven here. Swapping an entire init vector is not a valid robot-only intervention.',
                'Background intervention must preserve geometry, object poses, qpos/qvel and contacts; a mother-task reload is insufficient.'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    csvout(output/'case_audit.csv',rows);csvout(output/'head_category_difficulty.csv',aggregates)
    summary=[]
    for head in HEADS.values():
        for category in CATEGORIES+('Original unperturbed sanity',):
            group=[r for r in aggregates if r['head']==head and r['category']==category]
            summary.append(dict(head=head,category=category,**{key:sum(r[key] for r in group)
                for key in group[0] if key not in ('head','category','difficulty')}))
    csvout(output/'head_category.csv',summary)
    dump(output/'historical_comparison.json',historical);dump(output/'protocol_audit.json',protocol)
    dump(output/'paired_plan.json',dict(selection_rule='For each head and focus category, minimum source_index, ignoring success.',
        cases=plan,count=len(plan),video=False,gpu=3,gpu_uuid=next(iter(rows))['gpu_uuid'],
        execution='Not launched; parent must schedule on GPU3 with same loaded model and pair order ABBA.',
        observables=['decoded action H50x7 and first R5','raw action H50x32','gripper sign/threshold crossing',
                     'normalized state L2/RMS','projected visual prefix L2/RMS','language embedding L2/RMS'],
        claim_gate='Action sensitivity is not a closed-loop success cause. Only validated single-factor closed-loop pairs support that claim.'))
    dump(output/'audit_receipt.json',dict(status='complete',gpu_used=False,model_constructed=False,simulator_constructed=False,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest_path,ledger_path,history_path,language_path,Path(__file__))},
        case_count=len(rows),condition_count=585,historical_count=len(history),plan_count=len(plan)))
    return historical


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    print(json.dumps(audit(parser.parse_args().output),indent=2))
