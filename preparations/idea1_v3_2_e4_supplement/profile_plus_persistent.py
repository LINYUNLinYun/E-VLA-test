"""Bounded warmup-only persistent original-LIBERO worker cost diagnostic.

Original and Plus imports never share a worker. Frozen episode() is unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from .native_envs import ROOT,sha,write_json
from .evaluation import protocol_identity
from .libero_plus_env import isolated_child_environment


def worker(a,protocol):
    started=time.perf_counter()
    import libero.libero as package
    from libero.libero.envs.env_wrapper import ControlEnv
    from . import run_plus
    from .profile_plus_parallel import warmup_rpc
    run_plus.rpc=warmup_rpc
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'ready.json',dict(pid=os.getpid(),actual_import=str(package.__file__),cold_native_import_s=time.perf_counter()-started,category='warmup'))
    for line in __import__('sys').stdin:
        request=json.loads(line)
        if request.get('op')=='shutdown': return
        index=request['index'];args=SimpleNamespace(output=request['output'],gpu=3,gpu_uuid=a.gpu_uuid,port=a.port,deadline_unix=a.deadline_unix,video=False)
        result=run_plus.episode(args,protocol['case_selection']['original_baselines'][index],'full_d10',protocol)
        result.update(category='warmup',scientific_denominator=False,persistent_worker_pid=os.getpid())
        write_json(Path(request['output'])/'result.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=19143)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--fresh-profile',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    protocol=protocol_identity(Path(a.root)/'manifest.json')
    if not a.launch: print(json.dumps(vars(a)));return
    if a.worker: worker(a,protocol);return
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);processes=[];receipt=dict(status='running',pid=os.getpid(),source_sha256=sha(__file__),category='warmup',scientific_denominator=False,cohorts=[])
    write_json(output/'receipt.json',receipt)
    try:
        for index in range(2):
            directory=output/('worker%d'%index);case=protocol['case_selection']['original_baselines'][0]
            env=isolated_child_environment(case,output/'config',3,a.gpu_uuid)
            command=[str(ROOT/'policies/libero-eval-v21/.venv/bin/python'),'-m',__spec__.name,'--root',a.root,'--output',str(directory),'--port',str(a.port),
                '--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--fresh-profile',a.fresh_profile,'--worker','--launch']
            log=(output/('worker%d.log'%index)).open('w');tick=time.perf_counter()
            process=subprocess.Popen(command,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
            processes.append((process,log,directory,tick))
        for process,log,directory,tick in processes:
            while not (directory/'ready.json').exists():
                if process.poll() is not None or time.time()>=a.deadline_unix: raise RuntimeError('Persistent worker import failed/deadline')
                time.sleep(.1)
            row=json.loads((directory/'ready.json').read_text());row['spawn_to_ready_s']=time.perf_counter()-tick
            receipt.setdefault('workers',[]).append(row)
        # Do not overlap a second profiling renderer cohort with the existing
        # fresh-parallel profile. CPU imports above are separately disclosed.
        while True:
            fresh=json.loads(Path(a.fresh_profile).read_text())
            if fresh.get('status')=='complete': break
            if time.time()>=a.deadline_unix: raise TimeoutError('No persistent profile budget after fresh profile')
            time.sleep(.2)
        receipt['fresh_profile']=fresh
        for concurrent in (False,True):
            label='persistent2' if concurrent else 'persistent_serial';start=time.perf_counter();requests=[]
            for offset,index in enumerate((0,10)):
                directory=output/label/('case%04d'%index);process=processes[offset if concurrent else 0][0]
                process.stdin.write(json.dumps(dict(index=index,output=str(directory)))+'\n');process.stdin.flush();requests.append((index,directory))
                if not concurrent:
                    while not (directory/'result.json').exists() or json.loads((directory/'result.json').read_text()).get('status')=='running':
                        if time.time()>=a.deadline_unix: raise TimeoutError('Persistent profile deadline')
                        time.sleep(.1)
            while any(not (directory/'result.json').exists() or json.loads((directory/'result.json').read_text()).get('status')=='running' for _,directory in requests):
                if time.time()>=a.deadline_unix: raise TimeoutError('Persistent profile deadline')
                time.sleep(.1)
            rows=[dict(index=index,**{key:json.loads((directory/'result.json').read_text()).get(key) for key in ('status','success','steps','requests','wall_s')}) for index,directory in requests]
            receipt['cohorts'].append(dict(kind=label,wall_s=time.perf_counter()-start,results=rows))
            write_json(output/'receipt.json',receipt)
        serial,parallel=receipt['cohorts'];consistent=all((x['success'],x['steps'])==(y['success'],y['steps']) for x,y in zip(serial['results'],parallel['results']))
        receipt.update(status='complete',episode_speedup=serial['wall_s']/parallel['wall_s'],outcome_steps_equal=consistent,
            deployment_evidence='One persistent paired round is exploratory; not independently sufficient for robust20percent deployment gate',
            memory_after=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    except Exception:
        import traceback
        receipt.update(status='incomplete_or_technical_failure',error=traceback.format_exc(),deploy_parallel=False)
    finally:
        for process,log,directory,tick in processes:
            if process.poll() is None:
                try: process.stdin.write('{"op":"shutdown"}\n');process.stdin.flush();process.wait(timeout=5)
                except Exception:
                    process.terminate()
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: process.kill();process.wait()
            process.wait();log.close()
        receipt['all_workers_reaped']=True;write_json(output/'receipt.json',receipt)
    print(json.dumps({key:receipt.get(key) for key in ('status','episode_speedup','outcome_steps_equal')}))


if __name__=='__main__': main()
