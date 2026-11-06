"""Independent CPU check of same-frame capture artifacts; never imports a simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .draft_diagnostic_audit import DEFAULT_OUT,read,dump


def audit(directory):
    directory=Path(directory);rows=[]
    paths=list((directory/'capture_v2/restore8').glob('case*/result.json'))
    paths+=list((directory/'capture_v2/physical_input_pairs').glob('case*/receipt.json'))
    paths+=list((directory/'capture_v3/physical_input_pairs').glob('case*/receipt.json'))
    paths+=list((directory/'capture_v4/physical_input_pairs').glob('case*/receipt.json'))
    paths+=list((directory/'capture_v5/physical_input_pairs').glob('case*/receipt.json'))
    for path in paths:
        result=read(path)
        if result.get('status')!='complete' or result['case']['category']!='Sensor Noise':continue
        is_restore=path.name=='result.json'
        metadata=result.get('noise_capture') if is_restore else result['intervention']['capture']
        counterpath=path.with_name('intervention_observation.npz' if is_restore else 'counterfactual_observation.npz')
        actualpath=path.with_name('actual_observation.npz')
        with np.load(actualpath,allow_pickle=False) as actual,np.load(counterpath,allow_pickle=False) as counter:
            same={key:bool(np.array_equal(actual[key],counter[key])) for key in actual.files if key!='agentview_image'}
            clean_hash=hashlib.sha256(np.ascontiguousarray(counter['agentview_image']).tobytes()).hexdigest()
            image_differs=not np.array_equal(actual['agentview_image'],counter['agentview_image'])
        clean_matches_last=clean_hash==metadata['calls'][-1]['input']['sha256']
        physics=result['physical_bitwise_equal'] if is_restore else result['physical_equal']
        row=dict(head=result['case']['head'],source_index=result['case']['source_index'],source_type='restore' if is_restore else 'input',
            operator=metadata['operator'],severity=metadata['severity'],operator_calls=metadata['call_count'],
            expected_reset_plus5_settle_calls=metadata['call_count']==6,
            every_severity_matches=all(call['severity']==metadata['severity'] for call in metadata['calls']),
            counter_agentview_matches_final_operator_input=clean_matches_last,non_agentview_fields_bitwise_equal=same,
            agentview_changed=image_differs,physical_bitwise_equal=all(physics.values()),source_path=str(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        row['passed']=row['expected_reset_plus5_settle_calls'] and row['every_severity_matches'] and clean_matches_last and all(same.values()) and row['physical_bitwise_equal']
        rows.append(row)
    result=dict(status='complete_for_available_artifacts',count=len(rows),cases=rows,all_passed=all(r['passed'] for r in rows),
        frame_provenance='Native reset: one official noisy reset call, set_init_state does no corruption, then five official settle steps. Final captured operator input therefore belongs to final actual settle image. Counterfactual changes only agentview; all other model inputs and physical arrays are checked bitwise.',
        limitation='No re-render or observation refresh is used for noise intervention. Operator output is still produced by original official function; actual-arm reproduction against old GPU3 artifacts is a separate gate in restore results.')
    dump(directory/'capture_frame_audit.json',result)
    return dict(count=len(rows),all_passed=result['all_passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=DEFAULT_OUT)
    print(json.dumps(audit(p.parse_args().directory)))
