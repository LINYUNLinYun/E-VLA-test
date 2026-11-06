"""Fresh-physics reset formal100; preserves and invalidates prior warm-reset run."""
import argparse
import json
from pathlib import Path
from preparations.idea1_v3_2.transfer_protocol import digest,sha
from .policy_calvin_manifest import freeze,ROOT


def build(deadline):
    result=freeze(deadline);result.pop('manifest_hash')
    names=('policy_calvin_manifest_v3.py','policy_calvin_server.py','policy_native_watch_v3.py',
        'evaluation_v3.py','evaluation_v3_cpu_probe.py','test_evaluation_v3.py','native_reset.py','native_reset_probe.py','test_native_reset.py')
    for name in names:
        path=Path(__file__).with_name(name);result['source_files'][str(path.relative_to(ROOT))]=sha(path)
    gate=ROOT/'results/idea1_v3_2_e4_supplement/native_preflight/native_reset_fresh_v1/receipt.json'
    evidence=json.loads(gate.read_text())
    if evidence.get('passed') is not True or not all(row['passed'] for row in evidence['comparisons']):
        raise RuntimeError('Actual native fresh-reset gate must pass before registration')
    result['receipt_files'][str(gate)]=sha(gate)
    for row in evidence['histories']:
        if sha(row['path'])!=row['sha256']:raise ValueError('Reset actual capture changed')
        result['receipt_files'][row['path']]=row['sha256']
    invalid=ROOT/'results/idea1_v3_2_e4_supplement/policy_audit/native100_reset_invalid_pairing_v1.json'
    result['receipt_files'][str(invalid)]=sha(invalid)
    runtime=ROOT/'results/idea1_v3_2_e4_supplement/native_preflight/evaluation_v3_runtime_gate_v1/receipt.json'
    runtime_gate=json.loads(runtime.read_text())
    if runtime_gate.get('passed') is not True or runtime_gate.get('quality_requests')!=0:
        raise RuntimeError('Actual fresh driver runtime gate required, without quality calls')
    for filename,expected in runtime_gate['source_sha256'].items():
        if sha(filename)!=expected:raise ValueError('Runtime gate source changed: '+filename)
    for path in runtime.parent.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.npz'):
            result['receipt_files'][str(path)]=sha(path)
    result.update(schema=3,kind='formal_native_CALVIN_ABC_to_D_fixed100_fresh_physics',
        reset_contract='NEW independent native physics environment for EVERY sequence-condition; before native reset restore actual controller.gripper_action from official target robot_state[-1], preserve original width joint/motor reset + one physics step, refresh actual obs. Never reset within five-task chain.',
        initial_pair_gate='Before any quality for each condition, compare actual robot15/scene24 abs<=1e-12 rtol0; static/wrist RGB and first H5 noise bitwise; strict fail/technical error, never policy zero.',
        previous_formal_quality_reusable=False,previous_formal_reason='Warm reset omitted stored gripper command and retained physical scene-history effects; preserve prior results only as invalid-pairing evidence, restart complete fixed prefix0..99.',
        reset_actual_gate=dict(path=str(gate),sha256=sha(gate),passed=True,scope='CPU actual native simulation, targets0/1 opposite actual command+case predecessors; states, RGB, H5 noise bitwise; not model parity or quality.'),
        driver_actual_gate=dict(path=str(runtime),sha256=sha(runtime),passed=True,quality_requests=0),
        allocation_safety='Scan only root-authorized non-owned candidates every>=30s; immediate empty UUID recheck, child preconstructor and final preCUDA guard; no GPU launch from registration.',
        server_entrypoint='preparations.idea1_v3_2_e4_supplement.policy_calvin_server',
        evaluator_entrypoint='preparations.idea1_v3_2_e4_supplement.evaluation_v3',
        watcher_entrypoint='preparations.idea1_v3_2_e4_supplement.policy_native_watch_v3')
    result['manifest_hash']=digest(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--deadline-unix',type=float,required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    result=build(a.deadline_unix);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),manifest_hash=result['manifest_hash'],cpu_only=True,model_launched=False)))


if __name__=='__main__':main()
