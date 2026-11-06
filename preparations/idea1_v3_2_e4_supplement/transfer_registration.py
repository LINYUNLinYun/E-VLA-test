"""Independent corrected-B scientific registration; metadata-only CPU CLI.

This is neither a Plus390-case manifest nor permission to dispatch quality RPC.
The source LIBERO model protocol is a reference, not the B experiment identity.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
MODES=('full_d10','v14_d10','draft_goal','draft_object','draft_spatial','draft_10')
GOOGLE_TASKS=('google_robot_pick_coke_can','google_robot_move_near')
SCOPE='corrected_B_transfer'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_google_initializations():
    """Static native script/reset extraction, no simulator import/construction."""
    scripts=ROOT/'SimplerEnv/scripts'
    coke_script=scripts/'octo_pick_coke_can_visual_matching.sh'
    move_script=scripts/'octo_move_near_visual_matching.sh'
    envs=ROOT/'SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes'
    coke_source=envs/'grasp_single_in_scene.py';move_source=envs/'move_near_in_scene.py'
    text=coke_script.read_text()
    def axis(flag):
        match=re.search(re.escape(flag)+r'\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)',text)
        if not match:raise ValueError('Official Coke grid flag missing: '+flag)
        import numpy as np
        low,high,count=float(match[1]),float(match[2]),int(match[3])
        return np.linspace(low,high,count).tolist()
    xs,ys=axis('--obj-init-x'),axis('--obj-init-y')
    array=re.search(r'coke_can_options_arr=\((.*?)\)',text,re.S)
    orientations=re.findall(r'"(\w+)=True"',array[1] if array else '')
    if len(xs)!=5 or len(ys)!=5 or orientations!=['lr_switch','upright','laid_vertically']:
        raise ValueError('Official Coke25XY×3orientation contract changed')
    coke=[]
    for orientation in orientations[:2]:
        for x in xs:
            for y in ys:
                options=dict(obj_init_options=dict(init_xy=[x,y],orientation=orientation))
                coke.append(dict(reset_options=options,geometry_key=digest([x,y,orientation]),
                    official_xy=[x,y],official_orientation=orientation))
    tree=ast.parse(move_source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MoveNearGoogleBakedTexInSceneEnvV1')
    setup=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_setup_obj_configs')
    literals={}
    for node in setup.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Attribute) and target.attr in ('triplets','_xy_config_per_triplet'):
                    literals[target.attr]=ast.literal_eval(node.value)
    triplets=literals['triplets'];layouts=literals['_xy_config_per_triplet']
    if len(triplets)!=5 or len(layouts)!=2 or any(len(t)!=3 for t in triplets+layouts):
        raise ValueError('Native MoveNear5triplets×2layouts changed')
    if '--obj-episode-range 0 60' not in move_script.read_text():
        raise ValueError('Official MoveNear60episode enumeration changed')
    pairs=[(i,j) for i in range(3) for j in range(3) if i!=j]
    move=[]
    for triplet_index,triplet in enumerate(triplets):
        for layout_index,layout in enumerate(layouts):
            # Five roles per layout, jointly all six roles per triplet. This
            # selects official episode IDs, not invented workspace positions.
            roles=range(5) if layout_index==0 else range(1,6)
            for role in roles:
                episode=triplet_index*12+layout_index*6+role
                move.append(dict(reset_options=dict(obj_init_options=dict(episode_id=episode)),
                    geometry_key=digest([triplet,layout]),official_episode_id=episode,
                    triplet_index=triplet_index,layout_index=layout_index,object_ids=list(triplet),
                    object_xy=layout,source_object_index=pairs[role][0],target_object_index=pairs[role][1]))
    paths=[coke_script,move_script,coke_source,move_source,ROOT/'SimplerEnv/simpler_env/evaluation/maniskill2_evaluator.py',
           ROOT/'SimplerEnv/simpler_env/evaluation/argparse.py',ROOT/'SimplerEnv/simpler_env/__init__.py']
    audit=dict(status='static_native_options_audited_no_GPU',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        coke=dict(selected_cases=50,unique_requested_object_poses=50,grid_xy=[xs,ys],
            selected_orientations=orientations[:2],available_orientations=orientations,
            rule='Official25XY×first2namedorientations; not all75poses or allURDFvariants',
            seed_only_v1_evidence='Native _initialize_actors does uniformXY sampling; seed changes geometry, but explicitoptions avoid nominal duplicates'),
        move_near=dict(selected_cases=50,official_available_cases=60,unique_object_geometries=10,
            cases_per_triplet=10,cases_per_layout_per_triplet=5,distinct_source_target_roles=6,
            rule='For each5triplets: layout0roles0..4,layout1roles1..5; allsixroles jointly,50officialIDs not50uniquegeometry',
            seed_only_v1_evidence='Native reset randomly selects episode_id from60; repeats are possible; source/target roles share objectgeometries'),
        robot_init='Native prepackaged reset overrides robotinit to official fixed pose; do not claim an externalrobotgrid is applied',
        visual_randomization='Native seeded URDFvisual variation retained; record actualversion and objectposes during authorized run',
        runtime_measured_initial_poses=False)
    return {GOOGLE_TASKS[0]:coke,GOOGLE_TASKS[1]:move},audit


def check_registration(registration,verify_sources=False):
    payload={k:v for k,v in registration.items() if k!='registration_hash'}
    if digest(payload)!=registration.get('registration_hash'):
        raise ValueError('Independent B registration digest changed')
    if registration.get('scope')!=SCOPE or registration.get('family')!='libero_transfer':
        raise ValueError('Not a corrected B LIBERO transfer registration; Plus/native manifests are forbidden')
    if tuple(registration.get('modes',()))!=MODES or registration.get('H')!=50 or registration.get('R')!=5:
        raise ValueError('All six frozen source conditions H50/R5 required')
    if registration.get('denoising')!={m:(None if m.startswith('draft_') else 10) for m in MODES}:
        raise ValueError('Full/V14 D10; learned drafts D=N/A, not diffusion denoising')
    cases=registration['cases'];keys=[c['case_id'] for c in cases]
    if len(keys)!=300 or len(set(keys))!=300:
        raise ValueError('Expected300 unique expanded cases, never old8/5/5 or Plus390')
    calvin=[c for c in cases if c['environment']=='calvin']
    if ([c['sequence']['index'] for c in calvin]!=list(range(100)) or
            any(len(c['sequence']['tasks'])!=5 or c.get('max_steps_per_subtask')!=360 or c.get('native_hz')!=30 for c in calvin)):
        raise ValueError('Canonical100 five-subtask chains required')
    for task in GOOGLE_TASKS:
        selected=[c for c in cases if c.get('task')==task and c['environment']=='simpler']
        if [c['seed'] for c in selected]!=list(range(50)) or any(c.get('max_steps')!=80 or c.get('native_hz')!=3 for c in selected):
            raise ValueError('Each fixed Google task requires50 registered reset seeds')
    stress=[c for c in cases if c['environment']=='pusht']
    if ([c['seed'] for c in stress]!=list(range(100)) or
            any(not c.get('stress_projection') or c.get('max_steps')!=300 or c.get('native_hz')!=10 for c in stress)):
        raise ValueError('PushT100 native reset seeds must remain labelled projectionstress')
    if registration.get('condition_count')!=1800:
        raise ValueError('All300cases×6conditions required')
    if registration.get('schema')==2:
        google=[c for c in cases if c['environment']=='simpler']
        coke=[c for c in google if c['task']==GOOGLE_TASKS[0]]
        move=[c for c in google if c['task']==GOOGLE_TASKS[1]]
        if len({c['initialization']['geometry_key'] for c in coke})!=50:
            raise ValueError('Coke requires50explicit official objectpose cases')
        if len({c['initialization']['geometry_key'] for c in move})!=10:
            raise ValueError('MoveNear must disclose10objectgeometries, not invent50')
        ids=[c['reset_options']['obj_init_options']['episode_id'] for c in move]
        if len(set(ids))!=50 or min(ids)<0 or max(ids)>=60:
            raise ValueError('MoveNear requires50distinct officialepisode IDs')
    if verify_sources:
        for name,expected in registration['source_files'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Registered implementation changed: '+name)
        for item in registration['input_files'].values():
            if sha(item['path'])!=item['sha256']:raise ValueError('Registered input asset changed: '+item['path'])
    return registration


def load_registration(path,verify_sources=False):
    return check_registration(json.loads(Path(path).read_text()),verify_sources)


def build_registration(sequence_file,source_reference,policy_protocol):
    sequence_file,source_reference,policy_protocol=map(lambda p:Path(p).resolve(),(sequence_file,source_reference,policy_protocol))
    canonical_payload=json.loads(sequence_file.read_text());canonical=canonical_payload['sequences']
    if len(canonical)!=1000 or [c['index'] for c in canonical]!=list(range(1000)):
        raise ValueError('First generate the entire official canonical1000, then take100prefix')
    if canonical_payload.get('sequence_sha256')!=digest(canonical):
        raise ValueError('Canonical sequence payload hash mismatch')
    model=json.loads(policy_protocol.read_text())
    model_hash=digest({k:v for k,v in model.items() if k!='manifest_hash'})
    if model.get('manifest_hash')!=model_hash:
        raise ValueError('Referenced LIBERO model protocol digest differs')
    if model.get('family')!='libero_transfer':
        raise ValueError('Reference must name released LIBERO family, never target-trained native positive controls')
    from .corrected_transfer_envs import source_native_reference
    source_native_reference(json.loads(source_reference.read_text()))
    initializations,initialization_audit=official_google_initializations()
    cases=[]
    for sequence in canonical[:100]:
        cases.append(dict(case_id='calvin_seq%04d'%sequence['index'],environment='calvin',
            sequence=sequence,max_steps_per_subtask=360,native_hz=30,
            scientific_unit='canonical five-task state-carry chain',video=sequence['index']==0))
    for task in GOOGLE_TASKS:
        for seed,initialization in enumerate(initializations[task]):
            cases.append(dict(case_id='%s_init%03d'%(task,seed),environment='simpler',task=task,
                seed=seed,reset_options=initialization['reset_options'],initialization=initialization,max_steps=80,native_hz=3,
                scientific_unit='official explicit objectpose case' if task==GOOGLE_TASKS[0] else 'official explicit task-layout episode case',
                initial_distinctness='50requestedobjectposes' if task==GOOGLE_TASKS[0] else '50task-layoutcases sharing10objectgeometries; not50uniquegeometry',video=seed==0))
    for seed in range(100):
        cases.append(dict(case_id='pusht_stress_seed%03d'%seed,environment='pusht',seed=seed,
            instruction='push the T-shaped block onto the matching target',max_steps=300,native_hz=10,
            scientific_unit='native PushT reset seed with declared LIBERO7-toXY projection',stress_projection=True,video=seed==0))
    names=['transfer_registration.py','evaluate_transfer.py','corrected_transfer_envs.py','policy_native.py','native_envs.py']
    paths=[Path(__file__).with_name(name) for name in names]
    paths.extend(ROOT/p for p in ['preparations/idea1_v3_2/transfer_envs.py','preparations/idea1_v3_2/transfer_client.py',
                                'preparations/idea1_v2_1/runtime_client.py'])
    paths.extend(ROOT/p for p in initialization_audit['source_sha256'])
    registration=dict(schema=2,scope=SCOPE,family='libero_transfer',created_utc=datetime.now(timezone.utc).isoformat(),
        H=50,R=5,denoising={m:(None if m.startswith('draft_') else 10) for m in MODES},modes=list(MODES),
        case_count=300,condition_count=1800,cases=cases,google_initialization_audit=initialization_audit,
        input_files={name:dict(path=str(path),sha256=sha(path)) for name,path in
            [('canonical_sequences',sequence_file),('source_reference',source_reference),('model_protocol_reference',policy_protocol)]},
        policy_reference=dict(manifest_hash=model_hash,qualification='Same released LIBERO model identity reference only; not Plus permission or B results'),
        source_files={str(p.relative_to(ROOT)):sha(p) for p in paths},
        source_action='retain native sourceOSCclip[-1,1], .05m/.5rad physicaldelta; CALVINnativeabsolute, Googleee_align, PushTstressXY',
        state='episode-continuous quaternion lift anchored factualsourceq; no principal-rotvec branch replacement',
        reset='once per CALVINsequence; carry environment/lift through subtasks; model reset per subtask; otherenv per case',
        seed_rule='sha256([corrected-B-transfer-v1,case_id,subtask_index,replan_index])[:8]; mode excluded',
        video_rule='preregistered CALVINseq0, Googleeachinit0, PushTseed0, allsix modes; initial+everyactualstep; fullffmpegdecode+framecount',
        validity='Only matching complete case/mode is resumed; preserve technicalfailure attempts, never policy0; completion requires all1800valid conditions',
        launch_authorization='not granted by this CPU registration; require explicit B authorization and service B capability before anyquality RPC',
        aggregation='CALVIN chainmetrics; eachGoogletask SR separately; PushTstress separately; never pooledcapacityscore; Full0 impliesinconclusive',
        native_positive_controls=False)
    registration['registration_hash']=digest(registration)
    return check_registration(registration)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-file',required=True);p.add_argument('--source-reference',required=True)
    p.add_argument('--model-protocol-reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--write',action='store_true',help='Write CPU registration only; no service/env/model construction')
    a=p.parse_args();registration=build_registration(a.sequence_file,a.source_reference,a.model_protocol_reference)
    if a.write:
        target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x') as stream:json.dump(registration,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(registration_hash=registration['registration_hash'],case_count=300,condition_count=1800,
                         written=a.write,gpu_used=False,quality_requests=0)))


if __name__=='__main__':main()
