"""CPU quaternion geometry versus the released noncanonical native state representation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from .draft_diagnostic_audit import ROOT,DEFAULT_OUT,dump,read


def native_axisangle(quaternion):
    path=ROOT/'preparations/idea1_v2_1/runtime_models.py'
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='quat_axisangle')
    namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['quat_axisangle'](quaternion)


def state_difference(first,second):
    qa,qb=(np.asarray(o['robot0_eef_quat'],np.float64) for o in (first,second))
    qa=qa/np.linalg.norm(qa);qb=qb/np.linalg.norm(qb)
    dot=float(np.clip(np.dot(qa,qb),-1,1));geodesic=float(2*np.arccos(abs(dot)))
    aa,ab=native_axisangle(qa),native_axisangle(qb)
    ca,cb=(native_axisangle(q if q[3]>=0 else -q) for q in (qa,qb))
    delta=float(np.linalg.norm(ab-aa));canonical=float(np.linalg.norm(cb-ca))
    return dict(quaternion_first_xyzw=qa.tolist(),quaternion_second_xyzw=qb.tolist(),quaternion_signed_dot=dot,
        physical_rotation_geodesic_radians=geodesic,native_axisangle_first=aa.tolist(),native_axisangle_second=ab.tolist(),
        native_axisangle_delta_l2=delta,canonical_axisangle_delta_l2=canonical,
        native_minus_physical_delta=delta-geodesic,
        near_2pi_branch_jump_candidate=bool(geodesic<0.5 and delta>5.5),
        eef_translation_l2_m=float(np.linalg.norm(np.asarray(second['robot0_eef_pos'])-np.asarray(first['robot0_eef_pos']))),
        gripper_qpos_delta_l2=float(np.linalg.norm(np.asarray(second['robot0_gripper_qpos'])-np.asarray(first['robot0_gripper_qpos']))),
        interpretation='Geodesic compares physical orientations modulo q/-q. Native axis-angle is the actual released input convention; a large numeric difference is evidence of representation sensitivity, not proof of failure causality.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=DEFAULT_OUT);a=p.parse_args();rows=[]
    for path in sorted((a.directory/'physical_input_pairs').glob('case*/actual_observation.npz')):
        with np.load(path,allow_pickle=False) as archive:first={k:archive[k].copy() for k in archive.files}
        with np.load(path.with_name('counterfactual_observation.npz'),allow_pickle=False) as archive:second={k:archive[k].copy() for k in archive.files}
        rows.append(dict(case_directory=str(path.parent),**state_difference(first,second)))
    dump(a.directory/'state_representation_audit.json',dict(cases=rows,count=len(rows)))
    observational=[]
    entries=read(ROOT/'reports/idea1_libero_plus_half_final_20261022T2124Z/audit_ledger.json')['selected_completed']
    originals={}
    for entry in entries:
        if not entry['mode'].startswith('draft_'):continue
        result=read(ROOT/entry['path'])
        if result['case']['environment']=='libero_original':
            originals[(result['case']['mother_suite'],result['case']['mother_task'])]=(entry,result)
    for entry in entries:
        if not entry['mode'].startswith('draft_'):continue
        result=read(ROOT/entry['path']);case=result['case']
        if case['category']!='Robot Initial States':continue
        matched=originals.get((case['mother_suite'],case['mother_task']))
        if not matched:continue
        original_entry,_=matched
        source=(ROOT/entry['path']).parent/'initial_native_observation.npz'
        reference=(ROOT/original_entry['path']).parent/'initial_native_observation.npz'
        with np.load(source,allow_pickle=False) as archive:first={k:archive[k].copy() for k in archive.files if k!='instruction'}
        with np.load(reference,allow_pickle=False) as archive:second={k:archive[k].copy() for k in archive.files if k!='instruction'}
        observational.append(dict(head=entry['mode'],plus_source_index=entry['source_index'],original_source_index=original_entry['source_index'],
            mother_task=case['mother_task'],plus_success=result['success'],
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            **state_difference(first,second)))
    dump(a.directory/'existing_robot_state_observational.json',dict(count=len(observational),cases=observational,
        limitation='Existing Plus robot cases versus available original mother tasks; scene/seed and full physical identity are not controlled. These are representation diagnostics and correlations, not single-factor causal pairs.'))
    print(json.dumps(dict(count=len(rows),branch_jump_candidates=sum(r['near_2pi_branch_jump_candidate'] for r in rows))))


if __name__=='__main__':main()
