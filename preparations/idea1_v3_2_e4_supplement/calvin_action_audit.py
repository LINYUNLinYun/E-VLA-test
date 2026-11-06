"""CPU native-controller passthrough check plus immutable pilot clipping audit."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
from .native_envs import ROOT,sha,write_json,NativeCalvinEnv,calvin_native_action
from .native_exact import exact_calvin_action


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    robot=ROOT/'calvin/calvin_env/calvin_env/robot/robot.py';tree=ast.parse(robot.read_text())
    methods={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    relative=methods['relative_to_absolute'];calls=[ast.unparse(n) if hasattr(ast,'unparse') else getattr(n.func,'attr','') for n in ast.walk(relative) if isinstance(n,ast.Call)]
    if any('clip' in c for c in calls): raise RuntimeError('Official controller changed: audit native clipping again')
    rows=[];candidate=None
    for mode in ('full_d10','v14_d10'):
        for path in sorted(Path(a.pilot).glob('*/'+mode+'/infer.jsonl')):
            values=np.concatenate([np.asarray(json.loads(line)['actions'],dtype=float) for line in path.read_text().splitlines()])
            outside=np.abs(values[:,:6])>1
            rows.append(dict(sequence=path.parent.parent.name,mode=mode,predicted_actions=len(values),outside_components=int(outside.sum()),total_components=outside.size,
                max_abs_decoded=float(np.max(np.abs(values[:,:6]))),scope='All predicted channels, not only executed R5; this does not reconstruct a counterfactual policy rollout'))
            if candidate is None and outside.any(): candidate=values[np.flatnonzero(outside.any(axis=1))[0]].copy()
    env=NativeCalvinEnv();probes=[]
    try:
        for name,action in [('within_unit',np.array([.2,-.3,.1,.2,-.1,.1,1.])),('actual_pilot_outside_unit',candidate)]:
            outputs={}
            for variant,converter in [('old_clipped',lambda x:calvin_native_action(x,'relative')),('exact_passthrough',exact_calvin_action)]:
                with np.load(str(ROOT/'results/idea1_v3_2_e4_supplement/native_preflight/calvin_captured_formalD0/native_input.npz')) as captured:
                    env.reset_state(captured['robot_obs'],captured['scene_obs'])
                before_pos=np.array(env.env.robot.target_pos).copy();before_orn=np.array(env.env.robot.target_orn).copy()
                command,audit=converter(action)
                env.obs,_,_,_=env.env.step(command)
                outputs[variant]=dict(command=audit['native_action'],target_pos_delta=(np.array(env.env.robot.target_pos)-before_pos).tolist(),
                    target_euler_delta=(np.array(env.env.robot.target_orn)-before_orn).tolist(),actual_after=env.actual())
            probes.append(dict(name=name,raw_action=action.tolist(),variants=outputs))
    finally: env.close()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'receipt.json',dict(status='complete',official_robot_sha256=sha(robot),relative_controller_has_clip=False,
        native_relative_source_line=relative.lineno,pilot_added_clip=True,pilot_action_audit=rows,native_counterfactual_command_probes=probes,
        model_used=False,gpu_used=False,scope='CPU controller/action-interface diagnostic; old pilot is clipped and not relabeled exact-native; no new quality episodes'))


if __name__=='__main__': main()
