"""Actual CPU native scene reset gate, with opposite prior command histories."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .native_reset import FreshResetCalvinEnv,compare_observations
from .native_envs import ROOT,sha,write_json,official_calvin_initial_state
from .evaluation import request_seed


def run(sequence_file,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    sequences=json.loads(Path(sequence_file).read_text())['sequences']
    receipt=dict(status='running',cpu_only=True,gpu_used=False,model_calls=0,sequence_file=str(sequence_file),
        sequence_sha256=sha(sequence_file),started_unix=time.time(),comparisons=[],histories=[],
        contract='Each target condition gets a NEW physics context. Actual stored controller sign synchronized before native reset; no observation-only substitution; native motor/joint reset and one physics step retained.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),Path(__file__).with_name('native_reset.py'),
            ROOT/'calvin/calvin_env/calvin_env/robot/robot.py',ROOT/'calvin/calvin_env/calvin_env/envs/play_table_env.py')})
    write_json(output/'receipt.json',receipt)
    for index in (0,1):
        captures=[]
        for history,sign in ((2,-1),(3,1)):
            predecessor=FreshResetCalvinEnv()
            try:
                predecessor.reset_sequence(sequences[history]['initial_condition'])
                for _ in range(6):predecessor.env.step(np.r_[np.zeros(6),sign])
                prior=dict(case=history,requested_command=sign,actual_stored_command=float(predecessor.env.get_obs()['robot_obs'][-1]))
            finally:predecessor.close()
            env=FreshResetCalvinEnv()
            try:
                # Deliberately contaminate the NEW controller scalar too; reset
                # must repair the actual controller, not only a returned dict.
                env.env.robot.gripper_action=sign
                env.reset_sequence(sequences[index]['initial_condition'])
                robot_target,_=official_calvin_initial_state(sequences[index]['initial_condition'])
                seed=request_seed('calvin_abc_d_native',index,0,0)
                capture=dict(robot_obs=np.asarray(env.obs['robot_obs']).copy(),scene_obs=np.asarray(env.obs['scene_obs']).copy(),
                    base_rgb=np.asarray(env.obs['rgb_obs']['rgb_static']).copy(),wrist_rgb=np.asarray(env.obs['rgb_obs']['rgb_gripper']).copy(),
                    noise=np.random.default_rng(seed).standard_normal((5,32),dtype=np.float32))
                path=output/('target%d_history%d_sign%d.npz'%(index,history,sign))
                np.savez_compressed(str(path),**capture)
                row=dict(target=index,predecessor=prior,new_context=True,initial_poisoned_sign=sign,
                    target_stored_sign=float(robot_target[-1]),actual_stored_sign=float(env.env.robot.gripper_action),
                    actual_obs_stored_sign=float(capture['robot_obs'][-1]),noise_seed=seed,path=str(path),sha256=sha(path))
                receipt['histories'].append(row);captures.append(capture)
            finally:env.close()
            write_json(output/'receipt.json',receipt)
        receipt['comparisons'].append(dict(target_sequence=index,**compare_observations(*captures)))
        write_json(output/'receipt.json',receipt)
    receipt.update(status='complete',passed=all(x['passed'] for x in receipt['comparisons']),completed_unix=time.time())
    write_json(output/'receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence-file',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=run(a.sequence_file,a.output);print(json.dumps(dict(status=result['status'],passed=result['passed'])))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
