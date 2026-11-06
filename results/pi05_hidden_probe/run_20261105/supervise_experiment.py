"""Persistently execute calibration, freezing, independent tests, and analysis.

GPU allocation follows the user's ten-second rule and never terminates another
process. The cache validation has its own already-running GPU waiter. Failed
technical attempts retain their logs and completed result rows; semantic failures
are ordinary observations and are never retried for a more favorable answer.
"""
from __future__ import annotations
import json,os,subprocess,time
from pathlib import Path
import yaml
from experiments.pi05_hidden_probe.common import load_config,write_json,read_jsonl
from experiments.pi05_hidden_probe.workflow import freeze_config

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
PYTHON=ROOT/'policies/openpi/.venv/bin/python'
STATE=BASE/'experiment_supervisor.json'


def read(path):return json.loads(Path(path).read_text())


def gpu_snapshot():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [{'index':int(p[0]),'used_mib':int(p[1]),'utilization':int(p[2])}
            for line in output.splitlines() for p in [[v.strip() for v in line.split(',')]]]


def environment(gpu):
    return {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(ROOT),
            'OPENPI_DATA_HOME':str(ROOT/'shared/checkpoints/openpi'),'XLA_PYTHON_CLIENT_PREALLOCATE':'false',
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}


def checkpoint_complete(output):
    path=Path(output)/'completion.json'
    return path.exists() and read(path).get('completed') is True


def main():
    validation_path=BASE/'cached_generation_validation.json'
    while not validation_path.exists():
        job=read(BASE/'cached_validation_job.json')
        write_json(STATE,{'stage':'waiting_for_cache_validation','validation_job':job,'updated_unix':time.time()})
        if job['state']=='finished':
            log=(BASE/'cached_validation.log').read_text(errors='replace')
            if 'OutOfMemoryError' in log or 'CUDA out of memory' in log:
                child=subprocess.Popen([str(PYTHON),'-u',str(BASE/'wait_cached_validation.py')],cwd=ROOT,env=environment(''),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                print('GPU contention interrupted validation; started a new ten-second waiter',child.pid,flush=True)
                time.sleep(2)
            else:raise RuntimeError('Cache validation stopped before producing a numerical result; inspect cached_validation.log')
        time.sleep(10)
    validation=read(validation_path)
    cached=bool(validation['passed'])
    mode='patched_prefill_kv_cache' if cached else 'reference_full_recomputation'
    print('Frozen decode implementation',mode,flush=True)
    plans={}
    for channel in ['image','text']:
        path=BASE/(channel+'_calibration.yaml')
        config=yaml.safe_load(path.read_text())
        config['selection_prediction']='score_prediction'
        if cached:config['target_model']['decode_implementation']=mode
        else:config['target_model'].pop('decode_implementation',None)
        path.write_text(yaml.safe_dump(config,sort_keys=False))
        plans[channel]={'phase':'calibration','config':str(path),'output':config['output_dir'],'attempts':0}
    running={};completed=[]
    while any(p['phase']!='done' for p in plans.values()):
        for channel,entry in list(running.items()):
            process=entry['process'];code=process.poll()
            if code is None:continue
            entry['log'].close();del running[channel]
            plan=plans[channel]
            if code!=0:
                text=Path(entry['log_path']).read_text(errors='replace')
                if ('OutOfMemoryError' in text or 'CUDA out of memory' in text or 'RESOURCE_EXHAUSTED' in text) and plan['attempts']<4:
                    print('Technical GPU contention, preserving rows and waiting:',channel,plan['phase'],flush=True)
                    continue
                write_json(STATE,{'stage':'technical_error','channel':channel,'phase':plan['phase'],'exit_code':code,'log':entry['log_path'],'completed':completed})
                raise RuntimeError('Experiment worker failed; inspect '+entry['log_path'])
            if not checkpoint_complete(plan['output']):raise RuntimeError('Worker exited without a completed result matrix')
            completed.append({'channel':channel,'phase':plan['phase'],'gpu':entry['gpu'],'exit_code':code,'output':plan['output']})
            if plan['phase']=='calibration':
                config=load_config(plan['config']);result_path=Path(plan['output'])/'probe_results.jsonl'
                frozen_path=BASE/channel/'frozen_probe_config.yaml'
                frozen=freeze_config(config,read_jsonl(result_path),frozen_path,input_path=result_path)
                plan.update(phase='evaluation',config=str(frozen_path),output=frozen['output_dir'],attempts=0)
                print('Calibration complete; frozen Target layers:',channel,frozen['calibration_provenance']['selection'],flush=True)
            else:
                analysis=Path(plan['output'])/'analysis'
                log_path=BASE/channel/'analysis.log'
                with log_path.open('w') as log:
                    result=subprocess.run([str(PYTHON),'-m','experiments.pi05_hidden_probe.analyze_results','--input',str(Path(plan['output'])/'probe_results.jsonl'),
                        '--output-dir',str(analysis),'--bootstrap-samples','1000','--seed','42'],cwd=ROOT,env=environment(''),stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError('Analysis failed: '+str(log_path))
                plan.update(phase='done',analysis=str(analysis))
                print('Independent test and analysis complete',channel,flush=True)
        snapshot=gpu_snapshot()
        used={e['gpu'] for e in running.values()}
        idle=[g['index'] for g in snapshot if g['index'] in [2,3] and g['index'] not in used and g['used_mib']<1000 and g['utilization']<2]
        for channel,plan in plans.items():
            if channel in running or plan['phase']=='done' or not idle:continue
            gpu=idle.pop(0)
            plan['attempts']+=1
            log_path=BASE/channel/(plan['phase']+f"_attempt{plan['attempts']:02d}.log")
            log_path.parent.mkdir(exist_ok=True)
            if cached:argv=[str(PYTHON),'-u',str(BASE/'run_cached_probe.py'),'--config',plan['config']]
            else:argv=[str(PYTHON),'-u','-m','experiments.pi05_hidden_probe.run_patch_probe','--config',plan['config']]
            if (Path(plan['output'])/'probe_results.jsonl').exists():argv.append('--resume')
            log=log_path.open('a')
            process=subprocess.Popen(argv,cwd=ROOT,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
            running[channel]={'process':process,'gpu':gpu,'log':log,'log_path':str(log_path)}
            print('Started immediately',channel,plan['phase'],'gpu',gpu,'pid',process.pid,flush=True)
        write_json(STATE,{'stage':'running','decode_implementation':mode,'plans':plans,
            'running':[{'channel':c,'phase':plans[c]['phase'],'gpu':r['gpu'],'pid':r['process'].pid,'log':r['log_path']} for c,r in running.items()],
            'gpu_snapshot':snapshot,'poll_seconds':10,'completed':completed,'updated_unix':time.time()})
        if any(p['phase']!='done' for p in plans.values()):time.sleep(10)
    write_json(STATE,{'stage':'all_runs_and_analysis_complete','decode_implementation':mode,'plans':plans,'completed':completed,'updated_unix':time.time()})
    print('All runs complete; primary-agent completion audit and interpretation remain',flush=True)


if __name__=='__main__':main()
