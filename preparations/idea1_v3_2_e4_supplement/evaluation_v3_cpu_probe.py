"""Actual CPU physics test of v3 lifecycle/gate; no model/service/quality calls."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from . import evaluation_v3 as driver
from .native_envs import digest,sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequences',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    sequences=json.loads(Path(a.sequences).read_text())['sequences'][:2]
    plan=dict(plan_hash=digest(['v3_CPU_prequality_only',sha(driver.__file__),sha(a.sequences)]),policy_manifest_hash='CPU_NO_MODEL_NO_QUALITY')
    old=driver.rollout.run_case;calls=[]
    def no_quality(args,env,sequence,mode,plan,directory):
        gate=Path(directory).parent/('prequality_'+Path(directory).name)/'gate.json'
        if not json.loads(gate.read_text())['passed']: raise RuntimeError('Failedgate entered rollout')
        env.reset_sequence(sequence['initial_condition'])  # prepared handoff; must not reset physical context again
        calls.append(dict(sequence=sequence['index'],mode=mode))
        return dict(status='cpu_gate_probe_complete',success=None,requests=0,model_used=False,completed_subtasks=0)
    driver.rollout.run_case=no_quality
    rows=[]
    try:
        for sequence in sequences:
            reference,receipt=driver.reference_for_sequence(output,sequence,plan)
            for mode in ('full_d10','v14_d10'):
                # The probe stub never invokes the policy/RPC or steps a task.
                directory=output/('seq%04d'%sequence['index'])/mode/'attempt000'
                value=driver.run_fresh_case(SimpleNamespace(deadline_unix=0),sequence,mode,plan,directory,reference,receipt)
                rows.append(dict(sequence=sequence['index'],mode=mode,status=value['status'],gate=value.get('runtime_initial_gate'),requests=value.get('requests')))
    finally: driver.rollout.run_case=old
    result=dict(status='complete',passed=len(calls)==4 and all(row['status']=='cpu_gate_probe_complete' and row['gate']['passed'] and row['requests']==0 for row in rows),
        model_used=False,gpu_used=False,quality_requests=0,source_sha256={str(Path(driver.__file__)):sha(driver.__file__),str(Path(__file__)):sha(__file__)},
        scope='Real fresh CPU physics references+allfourcondition runtimegates through v3 PreparedSequence path; rollout stub makes no RPC or tasksteps',rows=rows)
    write_json(output/'receipt.json',result);print(json.dumps(dict(status=result['status'],passed=result['passed'],conditions=len(rows),quality_requests=0)))


if __name__=='__main__': main()
