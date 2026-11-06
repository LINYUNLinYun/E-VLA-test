"""Bounded native renderer/actual noise-operator gate, no model loading."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from .native_envs import sha,write_json
from .libero_plus_env import NativeLiberoPlusEnv,isolated_child_environment,PLUS,CATEGORIES


def probe(case,output,gpu,uuid):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',case=case,pid=os.getpid(),model_used=False)
    env=None
    try:
        from libero.libero.envs import env_wrapper as wrapper
        calls=[]
        # Instrument only calls, not outputs or native noise parameters.
        for name in ('motion_blur','gaussian_blur','zoom_blur','fog','glass_blur'):
            original=getattr(wrapper,name)
            def audited(image,*args,_function=original,_name=name,**kwargs):
                before=np.asarray(image).copy();after=_function(image,*args,**kwargs)
                calls.append(dict(operator=_name,severity=kwargs.get('severity',args[0] if args else 1),
                    input_sha256=__import__('hashlib').sha256(before.tobytes()).hexdigest(),
                    output_sha256=__import__('hashlib').sha256(np.asarray(after).tobytes()).hexdigest(),
                    mean_abs_change=float(np.mean(np.abs(before.astype(float)-np.asarray(after).astype(float))))))
                return after
            setattr(wrapper,name,audited)
        env=NativeLiberoPlusEnv(case,render=True,gpu=gpu,gpu_uuid=uuid)
        env.reset();before=env.actual()
        from OpenGL import GL
        gl_context={name:GL.glGetString(token).decode() for name,token in
            (('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION))}
        namespace=[line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('Pid:','NSpid:'))]
        audits=[]
        for action in (np.zeros(7),np.array([.1,0,0,0,0,0,0]),np.zeros(7)):
            _,audit=env.step_native(action);audits.append(audit)
        for key in ('agentview_image','robot0_eye_in_hand_image'):
            image=np.asarray(env.obs[key]);assert image.shape==(256,256,3) and image.dtype==np.uint8
        np.savez_compressed(str(output/'actual_observation.npz'),**{k:env.obs[k] for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')},instruction=env.info['instruction'])
        result.update(status='complete',passed=True,native_info=env.info,actual_before=before,actual_after=env.actual(),
            steps=audits,noise=int(env.env.noise),noise_calls=calls,
            observation_sha256=sha(output/'actual_observation.npz'),gpu_uuid=uuid,
            native_render_passed=True,gl_context=gl_context,namespace_identity=namespace,
            binding_environment={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','MUJOCO_GL','MUJOCO_EGL_DEVICE_ID')},
            allocation_compute_pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],text=True),
            renderer='native MuJoCo EGL',scope='Native RGB/init/control/oracle and actual wrapper operator calls; no model quality evaluation')
        if int(env.env.noise) and not calls: raise RuntimeError('Native noise case did not invoke noise operator')
    except Exception:
        result.update(status='technical_failure',passed=False,error=traceback.format_exc())
    finally:
        if env is not None: env.close()
        write_json(output/'receipt.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu',type=int,default=3);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--child-case',type=int)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not a.launch: print(json.dumps(vars(a)));return
    payload=json.loads(Path(a.selection).read_text());selected=payload['selected']
    if a.child_case is not None:
        result=probe(selected[a.child_case],a.output,a.gpu,a.gpu_uuid);raise SystemExit(0 if result['passed'] else 1)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    # One genuine selected instance per category plus every noise operator in
    # the selected subset. Case choice is independent of any policy outcome.
    indices=[]
    for category in CATEGORIES:
        indices.append(next(i for i,c in enumerate(selected) if c['category']==category))
    operators={}
    for i,c in enumerate(selected):
        if c['category']=='Sensor Noise':
            noise=int(c['name'].split('_noise_')[-1])
            operators.setdefault((noise-1)//10,i)
    indices=list(dict.fromkeys(indices+list(operators.values())))
    receipt=dict(status='running',selection_sha256=payload['selection_sha256'],indices=indices,
        gpu_uuid=a.gpu_uuid,pid=os.getpid(),results=[])
    from preparations.idea1_v3_2.transfer_envs import gpu_renderer_binding
    receipt['initial_idle_binding']=gpu_renderer_binding(a.gpu,a.gpu_uuid,require_idle=True)
    write_json(output/'receipt.json',receipt)
    for index in indices:
        if time.time()>=a.deadline_unix: break
        directory=output/('case%03d'%index)
        env=isolated_child_environment(selected[index],output/'config',a.gpu,a.gpu_uuid)
        command=[str(PLUS/'.venv/bin/python'),'-m',__spec__.name,'--selection',a.selection,'--output',str(directory),
            '--gpu',str(a.gpu),'--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--child-case',str(index),'--launch']
        with (output/('case%03d.log'%index)).open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            try: code=process.wait(timeout=max(1,a.deadline_unix-time.time()))
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill();process.wait()
                code=124
        row=dict(index=index,pid=process.pid,exit_code=code,receipt=str(directory/'receipt.json'))
        if (directory/'receipt.json').exists(): row['passed']=json.loads((directory/'receipt.json').read_text()).get('passed',False)
        receipt['results'].append(row);write_json(output/'receipt.json',receipt)
    receipt.update(status='complete',passed=len(receipt['results'])==len(indices) and all(r.get('passed') for r in receipt['results']),
        all_children_reaped=True,completed_unix=time.time())
    write_json(output/'receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__': main()
