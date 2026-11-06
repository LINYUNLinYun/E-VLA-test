"""Fresh-physics native CALVIN condition evaluation with pre-quality pair gates.

Reuses the frozen v2 five-task rollout loop, not its warm environment lifecycle.
Each reference and condition is a new CPU physics context. No old v2 result is
reused. A prepared one-shot proxy prevents a second warm reset inside v2.
"""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from .native_envs import ROOT,digest,sha,write_json,chain_metrics
from .native_reset import FreshResetCalvinEnv,compare_observations
from . import evaluation_v2 as rollout
from .evaluation import protocol_identity,verify_service,native_preflight,request_seed,check_deadline


class PreparedSequence:
    """One already-reset fresh environment, handed to the frozen rollout once."""
    def __init__(self,env,initial_condition):
        self._env=env;self._initial_hash=digest(initial_condition);self._consumed=False

    def reset_sequence(self,initial_condition):
        if self._consumed or digest(initial_condition)!=self._initial_hash:
            raise ValueError('Prepared sequence may be consumed once with its exact official initial condition')
        self._consumed=True
        return self._env.obs

    def __getattr__(self,name): return getattr(self._env,name)


def capture(env,sequence):
    seed=request_seed('calvin_abc_d_native',sequence['index'],0,0)
    return dict(robot_obs=np.asarray(env.obs['robot_obs']).copy(),scene_obs=np.asarray(env.obs['scene_obs']).copy(),
        base_rgb=np.asarray(env.obs['rgb_obs']['rgb_static']).copy(),wrist_rgb=np.asarray(env.obs['rgb_obs']['rgb_gripper']).copy(),
        noise=np.random.default_rng(seed).standard_normal((5,32),dtype=np.float32))


def save_capture(path,observation):
    np.savez_compressed(str(path),**observation)


def load_capture(path):
    with np.load(str(path),allow_pickle=False) as data:
        return {key:data[key].copy() for key in ('robot_obs','scene_obs','base_rgb','wrist_rgb','noise')}


def reference_for_sequence(output,sequence,plan,env_factory=FreshResetCalvinEnv):
    directory=Path(output)/('seq%04d'%sequence['index'])/'paired_reference'
    directory.mkdir(parents=True,exist_ok=True);metadata=directory/'receipt.json';path=directory/'reference.npz'
    if metadata.exists():
        receipt=json.loads(metadata.read_text())
        if receipt.get('plan_hash')!=plan['plan_hash'] or receipt.get('sequence_hash')!=digest(sequence) or receipt.get('capture_sha256')!=sha(path):
            raise ValueError('Immutable runtime reference identity mismatch')
        return load_capture(path),receipt
    env=None
    try:
        env=env_factory();env.reset_sequence(sequence['initial_condition']);observation=capture(env,sequence)
        save_capture(path,observation)
        receipt=dict(status='complete',plan_hash=plan['plan_hash'],sequence_hash=digest(sequence),capture_path=str(path),capture_sha256=sha(path),
            native_environment='FreshResetCalvinEnv new independent physics reference; closed before qualitycondition',
            first_request_seed=request_seed('calvin_abc_d_native',sequence['index'],0,0),noise_shape=[5,32],
            reference_is_quality_episode=False,reference_model_calls=0,source_sha256=sha(__file__))
        write_json(metadata,receipt);return observation,receipt
    finally:
        if env is not None: env.close()


def run_fresh_case(args,sequence,mode,plan,directory,reference,reference_receipt,env_factory=FreshResetCalvinEnv):
    directory=Path(directory);gate_directory=directory.parent/('prequality_'+directory.name)
    gate_directory.mkdir(parents=True,exist_ok=False);env=None
    failure=dict(status='technical_failure',success=None,plan_hash=plan['plan_hash'],manifest_hash=plan['policy_manifest_hash'],
        sequence_index=sequence['index'],sequence_hash=digest(sequence),mode=mode,completed_subtasks=0,requests=0)
    started=time.perf_counter()
    try:
        check_deadline(args.deadline_unix)
        env=env_factory();env.reset_sequence(sequence['initial_condition']);actual=capture(env,sequence)
        save_capture(gate_directory/'actual_initial.npz',actual)
        cpu_comparison=compare_observations(reference,actual)
        gate=dict(status='complete',passed=cpu_comparison['passed'],before_any_quality_request=True,
            plan_hash=plan['plan_hash'],manifest_hash=plan['policy_manifest_hash'],mode=mode,sequence_index=sequence['index'],
            cpu_reference=reference_receipt,comparison_to_cpu_reference=cpu_comparison,
            actual_capture_path=str(gate_directory/'actual_initial.npz'),actual_capture_sha256=sha(gate_directory/'actual_initial.npz'),
            first_request_seed=request_seed('calvin_abc_d_native',sequence['index'],0,0),
            reset_contract='New independent physics persequencecondition; actual controller storedsign synchronized before official reset; native motor width reset and physicsstep unchanged')
        full_path=directory.parents[1]/'paired_reference/full_actual_reference.npz'
        if full_path.exists():
            pair=compare_observations(load_capture(full_path),actual)
            gate.update(comparison_to_full_actual=pair,passed=gate['passed'] and pair['passed'])
        if mode=='full_d10' and gate['passed'] and not full_path.exists():
            save_capture(full_path,actual)
            write_json(full_path.with_suffix('.json'),dict(plan_hash=plan['plan_hash'],sequence_hash=digest(sequence),
                capture_sha256=sha(full_path),source_gate=str(gate_directory/'gate.json'),saved_before_first_quality_request=True))
        write_json(gate_directory/'gate.json',gate)
        if not gate['passed']:
            directory.mkdir(parents=True,exist_ok=False)
            result=dict(failure,status='invalid_initial_condition',runtime_initial_gate=gate,wall_s=time.perf_counter()-started,
                invalid_reason='Pre-quality actual initial robot15/scene24/RGB/noise pairing gate failed; no quality request sent')
            write_json(directory/'result.json',result);return result
        result=rollout.run_case(args,PreparedSequence(env,sequence['initial_condition']),sequence,mode,plan,directory)
        result.update(driver_version=3,runtime_initial_gate=gate,fresh_independent_environment=True,
            lifecycle='Fresh only before condition; same environment and scene carried across all five subtasks; closed after result')
        write_json(directory/'result.json',result);return result
    except Exception:
        directory.mkdir(parents=True,exist_ok=True)
        result=dict(failure,error=traceback.format_exc(),wall_s=time.perf_counter()-started)
        write_json(directory/'result.json',result);return result
    finally:
        if env is not None: env.close()


def result_matches(value,plan,sequence,mode):
    return (rollout.result_matches(value,plan,sequence,mode) and value.get('driver_version')==3
        and value.get('fresh_independent_environment') is True and value.get('runtime_initial_gate',{}).get('passed') is True
        and value['runtime_initial_gate'].get('before_any_quality_request') is True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--sequence-file',required=True)
    p.add_argument('--sequence-count',type=int,default=100);p.add_argument('--modes',nargs='+',default=['full_d10','v14_d10'],choices=['full_d10','v14_d10'])
    p.add_argument('--max-new-sequences',type=int,default=10);p.add_argument('--preflight-input',required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--video',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args();a.gripper_semantics='action_sign'
    canonical=json.loads(Path(a.sequence_file).read_text())['sequences'];protocol=protocol_identity(a.protocol)
    if len(canonical)!=1000 or not 1<=a.sequence_count<=1000 or a.max_new_sequences<1 or a.modes!=['full_d10','v14_d10']:
        raise ValueError('Fixed canonical prefix and paired Full/V14 order required')
    sources=[Path(__file__),Path(__file__).with_name('native_reset.py'),Path(__file__).with_name('native_exact.py'),
        Path(__file__).with_name('native_envs.py'),Path(__file__).with_name('evaluation.py'),Path(__file__).with_name('evaluation_v2.py')]
    plan=dict(schema=3,policy_manifest_hash=protocol['manifest_hash'],canonical_sequence_file_sha256=sha(a.sequence_file),
        sequences=canonical[:a.sequence_count],modes=a.modes,scientific_unit='official native five-task sequence',
        state_semantics='training stored action_sign',action_semantics='native first6 passthrough; gripper threshold only',
        environment_lifecycle='Independent fresh physics reference and percondition; scene carry within5taskchain',
        initial_pair_gate='Before everycondition qualityRPC: state abs1e-12 rtol0; RGB/noise bitwise, compareCPUreference andFullactual whenavailable',
        prior_v2_results_reused=False,source_files={str(path.relative_to(ROOT)):sha(path) for path in sources})
    plan['plan_hash']=digest(plan)
    if not a.launch:
        print(json.dumps(dict(dry_run=True,plan_hash=plan['plan_hash'],driver_version=3,target_sequences=a.sequence_count,per_launch_new_sequences=a.max_new_sequences)));return
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=plan: raise ValueError('Existing immutable v3 plan mismatch')
    write_json(output/'plan.json',plan);verify_service(a.port,protocol['manifest_hash'],'calvin_abc_d_native')
    write_json(output/('policy_preflight_%d.json'%int(time.time())),native_preflight(a,protocol['manifest_hash']))
    results=[];new_sequences=0
    for sequence in plan['sequences']:
        missing=[]
        for mode in a.modes:
            base=output/('seq%04d'%sequence['index'])/mode;valid=[]
            for path in base.glob('attempt*/result.json'):
                value=json.loads(path.read_text())
                if result_matches(value,plan,sequence,mode): valid.append(value)
            if len(valid)>1: raise ValueError('Duplicate complete v3 scientificcase')
            if valid: results.append(valid[0])
            else: missing.append((mode,base))
        if not missing: continue
        if new_sequences>=a.max_new_sequences: break
        new_sequences+=1;check_deadline(a.deadline_unix)
        reference,reference_receipt=reference_for_sequence(output,sequence,plan)
        for mode,base in missing:
            indices=[int(path.name[7:]) for path in base.glob('attempt*') if path.is_dir() and path.name[7:].isdigit()]
            directory=base/('attempt%03d'%(max(indices,default=-1)+1))
            result=run_fresh_case(a,sequence,mode,plan,directory,reference,reference_receipt);results.append(result)
            write_json(output/'progress.json',dict(status='running',driver_version=3,plan_hash=plan['plan_hash'],results=results))
            if result['status']!='complete': raise RuntimeError('v3 invalid/technicalattempt preserved, not a policyzero; supervisor must stop')
    write_json(output/'progress.json',dict(status='complete' if len(results)==a.sequence_count*2 else 'shard_complete',driver_version=3,
        plan_hash=plan['plan_hash'],target_sequences=a.sequence_count,completed_cases=len(results),new_sequences=new_sequences,results=results,
        metrics={mode:chain_metrics([row['completed_subtasks'] for row in results if row['mode']==mode]) for mode in a.modes}))


if __name__=='__main__': main()
