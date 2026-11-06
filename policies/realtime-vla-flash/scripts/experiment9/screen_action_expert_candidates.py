"""Screen only probe-approved Action Expert skips on official init0-9 across four suites."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT))
import run_standard_vlm_pruning as runner

OUTPUT=runner.RESULTS/'action_expert_layer_probe/closed_loop'


def complete(path,layer,suite):
    try:
        rows=json.loads((path/'episode_log.json').read_text())
        manifest=json.loads((path/'manifest.json').read_text())
        keys=[(r['task_id'],r['episode_idx']) for r in rows]
        if len(keys)!=100 or set(keys)!={(t,i) for t in range(10) for i in range(10)} or manifest['task_suite_name']!=suite:
            return False
        for row in rows:
            paths=[path/row[f] for f in ('video_path','trace_path','telemetry_path','infer_path')]
            if any(not p.is_file() or p.stat().st_size==0 for p in paths):
                return False
            telemetry=[json.loads(s) for s in paths[2].read_text().splitlines() if s]
            if len(telemetry)!=row['executed_action_count']+1 or row['frame_count']!=len(telemetry):
                return False
            inference=[json.loads(s) for s in paths[3].read_text().splitlines() if s]
            if not inference or len(inference)!=row['infer_record_count']:
                return False
            for infer in inference:
                meta=infer['server_experiment']
                if meta['vlm_layer_pattern']!='vlm_full18' or meta['action_layer_pattern']!=f'ae_skip_l{layer}' or meta['denoise_steps']!=10 or meta['ablation_noise_seed']!=12345 or list(meta['action_layer_indices_1based'])!=[i for i in range(1,19) if i!=layer] or not meta.get('noise_sha256'):
                    return False
        return True
    except (OSError,ValueError,KeyError):
        return False


def pattern_parity(layers):
    """Confirm registered CUDA-graph skip equals the saved eager skip on one sample per phase."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    import pickle
    import numpy as np
    import torch
    from scripts.spec.triton import pi0_infer
    from scripts.experiment1.run_intermediate_action_probe import _load_prompt_embeddings
    from scripts.spec.spec_serve_policy import _build_policy_components
    from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed
    folder=runner.RESULTS/'action_expert_layer_probe'
    source=json.loads((folder/'observations/manifest.json').read_text())
    selected=[];seen=set()
    for filename in source['observations']:
        path=Path(filename)
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata_json'].item()))
        if meta['semantic_phase'] not in seen:
            selected.append((path,meta));seen.add(meta['semantic_phase'])
    _,transform,output_transform=_build_policy_components(config_name='pi0_libero',checkpoint_dir=runner.BASE_DIR)
    embeddings=_load_prompt_embeddings(runner.BASE_DIR.parent,[m for _,m in selected])
    if os.environ.get('AE_PROBE_WARM_DYNAMIC')=='1':
        all_meta=[]
        for filename in source['observations']:
            with np.load(filename,allow_pickle=False) as data:
                all_meta.append(json.loads(str(data['metadata_json'].item())))
        all_embeddings=_load_prompt_embeddings(runner.BASE_DIR.parent,all_meta)
        lengths=sorted({len(v) for v in all_embeddings.values()})[:2]
        for length in lengths:
            seq=512+length
            q=torch.zeros((seq*8,256),dtype=torch.bfloat16,device='cuda')
            cache=torch.zeros((2,seq+51,256),dtype=torch.bfloat16,device='cuda')
            pi0_infer.AttnSingleKey(q,cache[0,:seq],cache[1,:seq],1/(256**.5))
        torch.cuda.synchronize()
        print(f'Diagnostic prefix compilation warmup lengths={lengths}',flush=True)
    with (runner.BASE_DIR/'base_weights.pkl').open('rb') as stream:
        base=pickle.load(stream)
    checks=[]
    for layer in layers:
        pattern=f'ae_skip_l{layer}'
        pi0_infer._ACTION_LAYER_PATTERNS[pattern]=tuple(i for i in range(18) if i!=layer-1)
        for path,meta in selected:
            checkpoint=dict(base);checkpoint['language_embeds']=embeddings[meta['suite'],meta['task_description']]
            runtime=pi0_infer.Pi0Inference(checkpoint,num_views=2,chunk_size=50,ablation_layer_pattern=pattern)
            with np.load(path,allow_pickle=False) as data:
                transformed=transform({'observation/image':data['agent_image'],'observation/wrist_image':data['wrist_image'],'observation/state':data['robot_state'],'prompt':str(data['prompt'].item())})
            with np.load(folder/'probe/samples'/f"{meta['sample_id']}_seed12345.npz",allow_pickle=False) as probe:
                images,state,noise=prepare_triton_inputs_from_transformed(transformed=transformed,device='cuda',action_horizon=50,action_dim=32,noise=probe['noise'])
                raw=runtime.forward(images,state,noise).float().cpu().numpy().copy()
                command=np.asarray(output_transform({'state':np.asarray(transformed['state']).copy(),'actions':raw})['actions'],dtype=np.float32)
                equal=np.array_equal(command,probe['skip_actions'][layer-1])
                maximum=float(np.max(np.abs(command-probe['skip_actions'][layer-1])))
                if not equal:
                    from scripts.experiment9.probe_action_expert_layers import decoder_forward
                    eager,_,_=decoder_forward(runtime,images,state,noise,skip_layer=layer-1)
                    eager_command=np.asarray(output_transform({'state':np.asarray(transformed['state']).copy(),'actions':eager.float().cpu().numpy().copy()})['actions'],dtype=np.float32)
                    full,_,_=decoder_forward(runtime,images,state,noise)
                    full_command=np.asarray(output_transform({'state':np.asarray(transformed['state']).copy(),'actions':full.float().cpu().numpy().copy()})['actions'],dtype=np.float32)
                    details={'layer':layer,'sample_id':meta['sample_id'],'graph_vs_saved_skip_max_abs':maximum,'graph_vs_same_runtime_eager_max_abs':float(np.max(np.abs(command-eager_command))),'fresh_eager_skip_vs_saved_max_abs':float(np.max(np.abs(eager_command-probe['skip_actions'][layer-1]))),'fresh_full_vs_saved_full_max_abs':float(np.max(np.abs(full_command-probe['baseline_actions'])))}
                    print(json.dumps(details),flush=True)
                    (OUTPUT/'pattern_parity_failure.json').write_text(json.dumps(details,indent=2)+'\n')
                    raise RuntimeError(f'Closed-loop pattern disagrees with probe: {details}')
                checks.append({'layer':layer,'sample_id':meta['sample_id'],'bitwise_equal':True,'max_abs_error':maximum})
            del runtime,checkpoint;gc.collect();torch.cuda.empty_cache()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'registered_pattern_parity.json').write_text(json.dumps(checks,indent=2)+'\n')


def job(layer,suite,gpu):
    prefix=f'exp9_ae_layer_pilot_SkipAE_L{layer}_{suite}_i00_10_a'
    for path in sorted(OUTPUT.glob(prefix+'*'),reverse=True):
        if complete(path,layer,suite):
            return path
    attempts=[int(p.name[len(prefix):]) for p in OUTPUT.glob(prefix+'*') if p.name[len(prefix):].isdigit()]
    attempt=max(attempts,default=0)+1
    name=f'{prefix}{attempt:02d}';path=OUTPUT/name
    port=19000+gpu*100+(os.getpid()%80)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',LIBERO_CONFIG_PATH=str(ROOT/'.libero'),OPENPI_DATA_HOME=str(runner.OPENPI_DATA),PYTHONPATH=f'{runner.LIBERO_ROOT}:{ROOT / "src"}:{ROOT}')
    logs=OUTPUT/'logs';logs.mkdir(exist_ok=True)
    server_command=[runner.SERVER_PYTHON,HERE/'serve_action_layer_screen.py','--skip-action-layer',layer,'--config','pi0_libero','--base-triton-path',runner.BASE_DIR,'--draft-triton-path',runner.gpu_cache(gpu),'--backend','triton','--force-full-each-round','--denoise-steps','10','--ablation-noise-seed','12345','--max-exec-steps','12','--port',port]
    with (logs/f'{name}_server.log').open('w') as stream:
        server=subprocess.Popen([str(p) for p in server_command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+1800
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Screening server exited: {name}')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError(name)
                time.sleep(2)
        command=[runner.CLIENT_PYTHON,ROOT/'scripts/spec/spec_client_libero.py','--host','127.0.0.1','--port',port,'--task-suite-name',suite,'--task','0-9','--num-trials-per-task','10','--episode-index-start','0','--episode-index-end','10','--num-steps-wait','10','--replan-steps','12','--seed','7','--record-semantic-telemetry','--video-out-path',OUTPUT,'--run-name',name]
        runner.log_event({'event':'action_layer_screen_start','layer':layer,'suite':suite,'gpu':gpu,'run_dir':str(path)})
        with (logs/f'{name}_client.log').open('w') as stream:
            subprocess.run([str(p) for p in command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if not complete(path,layer,suite):raise RuntimeError(f'Incomplete screening data: {name}')
        runner.log_event({'event':'action_layer_screen_complete','layer':layer,'suite':suite,'gpu':gpu,'run_dir':str(path)})
        return path
    finally:
        if server.poll() is None:
            server.terminate()
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=15)


def main():
    summary=json.loads((runner.RESULTS/'action_expert_layer_probe/probe/summary.json').read_text())
    layers=summary['closed_loop_candidates']
    if not summary['complete'] or summary['task_count']!=40 or summary['noise_seeds']!=[12345,23456,34567] or len(layers)>2 or any(not any(r['layer']==l and r['eligible'] for r in summary['ranking']) for l in layers):
        raise RuntimeError('Probe candidate eligibility not verified')
    if not layers:return 0
    pattern_parity(layers)
    if os.environ.get('AE_PARITY_ONLY')=='1':
        return 0
    pending=[(layer,suite) for suite in ('libero_10','libero_goal','libero_spatial','libero_object') for layer in layers]
    active={};paths=[];next_poll=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        while pending or active:
            if pending and time.monotonic()>=next_poll:
                occupied={v[2] for v in active.values()}
                free=[g for g in runner.free_gpus(runner.load_config()) if g not in occupied]
                for gpu in free:
                    if not pending:break
                    layer,suite=pending.pop(0);active[pool.submit(job,layer,suite,gpu)]=(layer,suite,gpu)
                next_poll=time.monotonic()+300
            if active:
                done,_=wait(active,timeout=60,return_when=FIRST_COMPLETED)
                for future in done:
                    _,_,gpu=active.pop(future);paths.append(future.result())
                    if pending:
                        layer,suite=pending.pop(0);active[pool.submit(job,layer,suite,gpu)]=(layer,suite,gpu)
            elif pending:time.sleep(60)
    rows=[]
    for path in paths:
        records=json.loads((path/'episode_log.json').read_text())
        rows.append({'run_dir':str(path),'suite':records[0]['task_suite_name'],'successes':sum(r['success'] for r in records),'episodes':len(records)})
    (OUTPUT/'screening_summary.json').write_text(json.dumps({'complete':True,'layers':layers,'episodes':sum(r['episodes'] for r in rows),'results':rows},indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
