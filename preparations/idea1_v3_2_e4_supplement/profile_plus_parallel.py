"""Warmup-only strict interleaving gate and fixed two-environment profile.

Does not modify the frozen child/service or count probes as scientific episodes.
The existing sequential scientific queue remains live during this profile.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from .native_envs import ROOT,digest,sha,write_json
from .evaluation import protocol_identity,verify_service
from .libero_plus_env import isolated_child_environment
from preparations.idea1_v3_2.transfer_client import rpc


def warmup_rpc(port,request):
    request=dict(request)
    if request.get('op')=='predict': request['category']='warmup'
    return rpc(port,request)


def gate(a,protocol,output):
    captures=[]
    for index in (0,10):
        case=protocol['case_selection']['original_baselines'][index]
        path=Path(a.root)/'original40'/('case%04d'%index)/'full_d10/attempt000/initial_native_observation.npz'
        with np.load(str(path),allow_pickle=False) as data:
            observation={key:data[key].copy() for key in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
            instruction=str(data['instruction'])
        captures.append(dict(index=index,case=case,observation=observation,instruction=instruction,path=str(path),sha256=sha(path)))
    references={};checks=[];requests=[]
    def predict(which,mode,label):
        capture=captures[which]
        response=warmup_rpc(a.port,dict(op='predict',family='libero_transfer',environment='libero',mother_suite=capture['case']['mother_suite'],
            mode=mode,observation=capture['observation'],instruction=capture['instruction'],seed=7,category='warmup'))
        if response.get('manifest_hash')!=protocol['manifest_hash']: raise ValueError('Wrong service identity')
        arrays={key:np.asarray(response[key],dtype=np.float32) for key in ('actions','raw_actions') if key in response}
        if 'actions' not in arrays or any(not np.isfinite(value).all() for value in arrays.values()): raise ValueError('Nonfinite/missing prediction')
        requests.append(dict(label=label,index=capture['index'],mode=mode,category='warmup',
            output_hashes={key:__import__('hashlib').sha256(value.tobytes()).hexdigest() for key,value in arrays.items()}))
        return arrays
    for which in (0,1):
        for mode in ('full_d10','v14_d10',captures[which]['case']['matching_draft']):
            references[(which,mode)]=predict(which,mode,'reference')
    # Each repeat of A follows another input/route and a real reset request.
    for mode in ('full_d10','v14_d10',captures[0]['case']['matching_draft']):
        for which in (0,1,0):
            warmup_rpc(a.port,dict(op='reset'))
            selected=mode if not mode.startswith('draft_') else captures[which]['case']['matching_draft']
            candidate=predict(which,selected,'A-B-A_after_reset')
            reference=references[(which,selected)]
            comparison={key:dict(bitwise_equal=bool(np.array_equal(value,candidate[key])),
                strict_1e5=bool(np.allclose(value,candidate[key],atol=1e-5,rtol=1e-5)),
                max_abs=float(np.max(np.abs(value-candidate[key])))) for key,value in reference.items()}
            checks.append(dict(index=captures[which]['index'],mode=selected,comparison=comparison))
    result=dict(status='complete',passed=all(row['comparison'][key]['bitwise_equal'] for row in checks for key in row['comparison']),
        category='warmup',manifest_hash=protocol['manifest_hash'],checks=checks,requests=requests,
        captures=[{key:value for key,value in capture.items() if key not in ('observation','case')} for capture in captures],
        scope='Same actual captured native inputs+explicitnoise7; Full/V14/matched drafts with reset and A-B-A interleaving; sequential service')
    write_json(output/'interleave_gate.json',result);return result


def profile_episode(a,protocol):
    from . import run_plus
    original=run_plus.rpc;run_plus.rpc=warmup_rpc
    try:
        case=protocol['case_selection']['original_baselines'][a.child_index]
        args=SimpleNamespace(output=a.output,gpu=3,gpu_uuid=a.gpu_uuid,port=a.port,deadline_unix=a.deadline_unix,video=False)
        result=run_plus.episode(args,case,'full_d10',protocol)
        result.update(category='warmup',scientific_denominator=False,profile_source_sha256=sha(__file__))
        write_json(Path(a.output)/'result.json',result)
        return result
    finally: run_plus.rpc=original


def cohort(a,protocol,output,concurrent):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();active=[];rows=[];memory=[]
    cases=protocol['case_selection']['original_baselines']
    def launch(index):
        directory=output/('case%04d'%index);env=isolated_child_environment(cases[index],output/'config',3,a.gpu_uuid)
        command=[str(ROOT/'policies/libero-eval-v21/.venv/bin/python'),'-m',__spec__.name,'--root',a.root,'--output',str(directory),
            '--port',str(a.port),'--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--child-index',str(index),'--launch']
        log=(output/('case%04d.log'%index)).open('w');process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        active.append((index,process,log));rows.append(dict(index=index,pid=process.pid,command=command))
    try:
        for index in (0,10):
            launch(index)
            if not concurrent:
                while active[-1][1].poll() is None:
                    if time.time()>=a.deadline_unix: raise TimeoutError('Profile10minute deadline')
                    time.sleep(.1)
        while any(process.poll() is None for _,process,_ in active):
            if time.time()>=a.deadline_unix: raise TimeoutError('Profile10minute deadline')
            memory.append(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            time.sleep(.25)
    finally:
        for index,process,log in active:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill();process.wait()
            process.wait();log.close()
            next(row for row in rows if row['index']==index).update(exit_code=process.returncode,reaped=True)
    wall=time.perf_counter()-started;results=[]
    for row in rows:
        path=output/('case%04d'%row['index'])/'result.json'
        value=json.loads(path.read_text()) if path.exists() else dict(status='technical_failure',success=None)
        results.append(dict(index=row['index'],status=value['status'],success=value.get('success'),steps=value.get('steps',0),requests=value.get('requests',0),wall_s=value.get('wall_s'),result=str(path)))
    result=dict(status='complete',concurrent=concurrent,category='warmup',scientific_denominator=False,wall_s=wall,
        episodes_per_s=len(results)/wall,actions_per_s=sum(value['steps'] for value in results)/wall,
        passed=all(value['status']=='complete' for value in results) and all(row['exit_code']==0 for row in rows),
        results=results,processes=rows,gpu_memory_mib_samples=memory,
        background_interference='Original scientific sequential queue remains active on same service/GPU; profile does not claim isolated hardware throughput')
    write_json(output/'receipt.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=19143)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--child-index',type=int)
    p.add_argument('--launch',action='store_true');a=p.parse_args();protocol=protocol_identity(Path(a.root)/'manifest.json')
    if not a.launch: print(json.dumps(vars(a)));return
    if a.child_index is not None:
        value=profile_episode(a,protocol);raise SystemExit(0 if value['status']=='complete' else 1)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);verify_service(a.port,protocol['manifest_hash'],'libero_transfer')
    a.deadline_unix=min(a.deadline_unix,time.time()+600)
    receipt=dict(status='running',pid=os.getpid(),started_unix=time.time(),deadline_unix=a.deadline_unix,
        manifest_hash=protocol['manifest_hash'],source_sha256=sha(__file__),scientific_denominator=False,rounds=[])
    write_json(output/'receipt.json',receipt);receipt['gate']=gate(a,protocol,output)
    if not receipt['gate']['passed']:
        receipt.update(status='complete',deploy_parallel=False,reason='strict_interleaving_gate_failed');write_json(output/'receipt.json',receipt);return
    for repeat,order in enumerate(((False,True),(True,False))):
        by_kind={}
        for concurrent in order:
            label='parallel2' if concurrent else 'serial'
            by_kind[label]=cohort(a,protocol,output/('repeat%d_%s'%(repeat,label)),concurrent)
            write_json(output/'receipt.json',receipt)
        serial,parallel=by_kind['serial'],by_kind['parallel2']
        consistent=all((x['success'],x['steps'])==(y['success'],y['steps']) for x,y in zip(serial['results'],parallel['results']))
        receipt['rounds'].append(dict(repeat=repeat,serial=serial,parallel2=parallel,
            episode_speedup=serial['wall_s']/parallel['wall_s'],actions_speedup=parallel['actions_per_s']/serial['actions_per_s'],
            outcome_steps_equal=consistent))
        write_json(output/'receipt.json',receipt)
    receipt.update(status='complete',completed_unix=time.time(),deploy_parallel=all(row['serial']['passed'] and row['parallel2']['passed'] and
        row['episode_speedup']>=1.2 and row['actions_speedup']>=1.2 and row['outcome_steps_equal'] for row in receipt['rounds']),
        decision_rule='Strict bitwise same-input interleaving; both order-balanced rounds >=20% episode AND action throughput gain; exact profile outcomes+steps; no child technicalfailure/OOM. Otherwise keep scientific sequential queue.')
    write_json(output/'receipt.json',receipt);print(json.dumps(dict(deploy_parallel=receipt['deploy_parallel'],speedups=[row['episode_speedup'] for row in receipt['rounds']])))


if __name__=='__main__': main()
