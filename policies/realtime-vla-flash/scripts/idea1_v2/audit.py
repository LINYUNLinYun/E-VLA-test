"""Recheck actual artifacts; failed tasks are valid outcomes, missing data are not."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import SUITES, sha, write_json, rows, NEW
from .analysis import completed
from .runner import ALT, PHASE, validate

def run(root):
    missing=[];conditions={};expected={(s,t,i) for s in SUITES for t in range(10) for i in range(5)}
    for name,condition,mode in (('E2',ALT,'alternating'),('E3',PHASE,'phase')):
        errors=[]
        try:current=completed(root,condition)
        except Exception as exc:current={};errors.append(str(exc))
        for (s,t,i),row in current.items():
            try:
                actual=validate(Path(row['directory']),mode,condition,s,t,i)
                for field in ('success','actual_actions','actual_predictions','full_rounds','full_actions'):
                    if actual[field]!=row[field]:raise ValueError('Validated summary changed: '+field)
            except Exception as exc:errors.append('%s/%s/%s: %s'%(s,t,i,exc))
        valid=len(current) if not errors else None
        conditions[name]={'status':'complete' if set(current)==expected and not errors else 'incomplete',
                          'recorded_valid':len(current),'revalidated_valid':valid,'successes':sum(bool(r['success']) for r in current.values()),
                          'missing_keys':[list(k) for k in sorted(expected-set(current))],'errors':errors}
    e4path=root/'kalman/completion.json';e4=json.loads(e4path.read_text()) if e4path.exists() else {'status':'missing'}
    try:
        if (e4.get('points'),e4.get('calibration'),e4.get('evaluation'),e4.get('model_calls'),e4.get('environment_actions'))!=(6979,634,6345,0,0):raise ValueError('E4 coverage/counters mismatch')
        parameter=json.loads((root/'protocol/kalman_parameters.json').read_text());digest=parameter.pop('sha256')
        if hashlib.sha256(json.dumps(parameter,sort_keys=True).encode()).hexdigest()!=digest or digest!=e4['parameter_sha256']:raise ValueError('Kalman parameter hash mismatch')
        records=rows(root/'kalman/records.jsonl')
        if len(records)!=6979 or sum(r['split']=='calibration' for r in records)!=634 or sum(r['split']=='evaluation' for r in records)!=6345:raise ValueError('E4 metadata count/split mismatch')
        if any(r['parameter_sha256']!=digest or r['history_count']!=r['step'] for r in records):raise ValueError('E4 parameter/history provenance mismatch')
        import csv
        for path,count in (('kalman/calibration_scores.csv',25),('analysis/kalman_offsets.csv',400),('analysis/kalman_prefixes.csv',112)):
            with (root/path).open() as stream:actual_count=len(list(csv.DictReader(stream)))
            if actual_count!=count:raise ValueError('E4 table coverage mismatch: '+path)
        with np.load(root/'kalman/predictions.npz',allow_pickle=False) as data:
            if data['raw_mean'].shape!=(6979,50,3) or data['covariance'].shape!=(6979,50,3,3):raise ValueError('E4 array shape mismatch')
            if not all(np.isfinite(data[k]).all() for k in data.files):raise ValueError('Nonfinite E4 arrays')
        e4['status']='complete'
    except Exception as exc:e4=dict(e4,status='incomplete',error=str(exc))
    gate=conditions['E2']['status']=='complete' and conditions['E2']['successes']>=160 and e4['status']=='complete'
    deferred=root/'E3_deferred.json'
    if deferred.exists() and not gate and not conditions['E3']['recorded_valid']:
        if conditions['E2']['status']=='complete' and e4['status']=='complete':conditions['E3']['status']='deferred_by_preregistered_gate'
    try:
        smoke=json.loads((root/'smoke/gates.json').read_text())
        frozen=json.loads((root/'protocol/execution_manifest.json').read_text())
        if not smoke.get('passed') or smoke['code_hashes']!=frozen['code_hashes']:raise ValueError('Smoke gate/hash mismatch')
        from .__main__ import manifest
        if frozen['code_hashes']!=manifest(Path(frozen['source']))['code_hashes']:raise ValueError('Code differs from frozen provenance')
        sources=json.loads((root/'protocol/source_manifest.json').read_text())
        for path,digest in sources['files'].items():
            if sha(path)!=digest:raise ValueError('Source hash changed: '+path)
    except Exception as exc:missing.append('GPU smoke/baseline parity: '+str(exc))
    required=['analysis/success.csv','analysis/latency.csv','analysis/success_bootstrap.csv',
              'analysis/failure_events.csv','analysis/phase_counts.csv','analysis/kalman_offsets.csv',
              'analysis/kalman_prefixes.csv','analysis/kalman_strata.csv','media/cases.json','media/event_clips.json',
              'media/index.md','media/viewer.md','media/gallery.html','report.md','figures/kalman_offsets_50.png','figures/kalman_offsets_12.png']
    missing += [p for p in required if not (root/p).is_file()]
    ledger=NEW/'status/work_ledger.json'
    if not ledger.exists():missing.append('campaign work ledger')
    else:
        entries=json.loads(ledger.read_text())['entries']
        for condition in (ALT,PHASE):
            try:current=completed(root,condition)
            except Exception:continue # Already included in condition errors above.
            for (s,t,i),row in current.items():
                key='%s/%s/%02d/%02d'%(condition,s,t,i)
                if not any(e['key']==key and e['status']=='valid' and e.get('directory')==row['directory'] for e in entries):missing.append('ledger mismatch: '+key)
    result=dict(conditions,E4=e4,missing=missing,plugin_playback_test='not_run_manual_editor_check',
                experiments_complete=conditions['E2']['status']=='complete' and e4['status']=='complete' and
                conditions['E3']['status'] in ('complete','deferred_by_preregistered_gate') and not missing,
                failed_tasks_are_valid=True,diagnostics_are_not_success_gates=True)
    write_json(root/'completion_audit.json',result)
    write_json(root/'missing_or_deferred.json',{'missing':missing,'E2':conditions['E2'],'E3':conditions['E3'],
                                             'postponed':['V15','other alternation ratios','restoration causality','Xiaomi','online Kalman gating'],
                                             'manual_checks':['Markdown Preview Enhanced actual playback']})
    return result
