"""Check completed analysis artifacts against fixed selections and episode splits."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')

def read(path):return json.loads(path.read_text())
def csv_rows(name):return list(csv.DictReader((OUT/name).open()))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

samples=[]
for suite in SUITES:
    for task in range(10):
        for init in range(3):
            episode=ROOT/'episodes'/suite/f'task{task:02d}'/f'init{init:02d}'
            identity=read(episode/'identity.json')
            service=f"gpu{identity['service']['cuda_visible_devices']}_pid{identity['service']['pid']}"
            for item in read(episode/'selection.json')['samples']:
                samples.append(dict(item,suite=suite,task=task,init=init,service=service,
                                    episode=f'{suite}/{task}/{init}'))
assert len(samples)==2326
assert sum('uniform' in s['sources'] for s in samples)==1440

def validate_probe_rows(rows,mode):
    for row in rows:
        chosen=[s for s in samples if s['suite']==row['suite']]
        if row['target']=='phase':chosen=[s for s in chosen if s['phase']!='unknown']
        if mode=='linear' and row['subset']=='uniform12':
            chosen=[s for s in chosen if 'uniform' in s['sources']]
        if mode=='conditional':
            key='task' if row['condition']=='within_task' else 'phase'
            chosen=[s for s in chosen if str(s[key])==row['condition_value']]
        if mode=='backend':chosen=[s for s in chosen if s['service']==row['service_group']]
        train=[s for s in chosen if s['init']<2];test=[s for s in chosen if s['init']==2]
        assert (int(row['n_train']),int(row['n_test']))==(len(train),len(test))
        target=row['target']
        train_labels={s[target] for s in train};test_labels={s[target] for s in test}
        assert int(row['train_classes'])==len(train_labels)
        assert int(row['test_classes'])==len(test_labels)
        estimable=len(train_labels)>=2 and bool(test)
        assert bool(row['balanced_accuracy'])==estimable
        for metric in ('accuracy','balanced_accuracy','majority_accuracy','majority_balanced_accuracy'):
            if row[metric]:assert 0<=float(row[metric])<=1
        if estimable:
            unseen=sum(s[target] not in train_labels for s in test)/len(test)
            assert float(row['test_unseen_class_fraction'])==unseen

linear=csv_rows('linear_probes.csv');conditional=csv_rows('conditional_probes.csv')
backend=csv_rows('backend_control_probes.csv');clusters=csv_rows('clustering.csv')
assert (len(linear),len(conditional),len(backend),len(clusters))==(1568,2016,784,1656)
for rows,mode in ((linear,'linear'),(conditional,'conditional'),(backend,'backend')):
    validate_probe_rows(rows,mode)
assert Counter(r['status'] for r in clusters)=={
    'estimated':1644,'not_estimable:k_exceeds_distinct_training_inputs':12}
conditions=Counter()
for row in clusters:
    chosen=[s for s in samples if s['suite']==row['suite']]
    assert int(row['n_train'])==sum(s['init']<2 for s in chosen)
    assert int(row['n_test'])==sum(s['init']==2 for s in chosen)
    assert int(row['k']) in (5,10,15) and int(row['seed']) in (0,1,2)
    if row['status']!='estimated':assert int(row['k'])>int(row['distinct_train_inputs'])
    else:
        assert 1<=int(row['fitted_clusters'])<=int(row['k'])
        for key in ('task_ari','phase_ari','task_nmi','phase_nmi'):
            if row[key]:assert np.isfinite(float(row[key]))
    conditions[row['feature'],row['layer'],row['suite']]+=1
assert len(conditions)==184 and set(conditions.values())=={9}

projection_path=OUT/'pca_projections.json'
projections=read(projection_path)
assert len(projections)==184
expected_samples={s:(set((x['episode'],x['request']) for x in samples if x['suite']==s)) for s in SUITES}
lookup={(s['episode'],s['request']):s for s in samples}
keys=set()
for row in projections:
    key=(row['feature'],str(row['layer']),row['suite'])
    assert key in conditions and key not in keys;keys.add(key)
    identities=list(zip(row['episode'],row['request']))
    assert len(identities)==len(set(identities)) and set(identities)==expected_samples[row['suite']]
    assert np.asarray(row['coordinates']).shape==(len(identities),4)
    assert np.isfinite(row['coordinates']).all() and np.isfinite(row['variance']).all()
    for i,identity in enumerate(identities):
        original=lookup[identity]
        for field in ('task','phase','sources','phase_reason'):
            assert row[field][i]==original[field]
    assert row['cluster_spec']['k']==10 and row['cluster_spec']['seed']==0
    assert len(row['cluster'])==len(identities)
    assert all(0<=value<10 for value in row['cluster'])

pngs=list((OUT/'figures').glob('*.png'))
assert len(pngs)==68
assert Counter(path.name.split('_')[0] for path in pngs)==dict(pca=46,residual=8,probe=10,conditional=4)
for path in pngs:
    with Image.open(path) as image:
        image.load();assert min(image.size)>100
render=read(OUT/'pca_render_complete.json')
assert render['passed'] and render['figures']==46
assert render['projection_sha256']==sha(projection_path)
conditional_render=read(OUT/'conditional_render_complete.json')
assert conditional_render['passed'] and conditional_render['figures']==4
assert conditional_render['source_csv_sha256']==sha(OUT/'conditional_probes.csv')

page=(OUT/'pca_explorer.html').read_text()
script=page[page.rfind('<script>')+len('<script>'):page.rfind('</script>')]
assert script.startswith('const data=')
embedded=json.loads(script[len('const data='):script.index(';const el=')])
assert embedded==projections
for value in ('0,1','0,2','0,3','1,2','1,3','2,3','0,1,2'):
    assert f'value="{value}"' in page
script_path=ROOT/'diagnostics/final_pca_script.js';script_path.write_text(script)
subprocess.run(['node','--check',str(script_path)],check=True,capture_output=True,text=True)
for target in re.findall(r'\]\(([^)]+)\)',(OUT/'report_zh.md').read_text()):
    assert (OUT/target).exists(),target

receipts={name:read(ROOT/name) for name in (
    'collection_audit.json','controller_scale_audit.json','pilot_gate.json',
    'features_complete.json','feature_audit.json')}
assert all(value['passed'] for value in receipts.values())
assert receipts['feature_audit.json']['verified']==2326
assert receipts['feature_audit.json']['pending']==0
assert receipts['feature_audit.json']['max_raw_abs']==receipts['feature_audit.json']['max_decoded_abs']==0
analysis=read(OUT/'analysis_complete.json')
assert analysis['passed'] and analysis['episodes']==120 and analysis['samples']==2326
assert read(ROOT/'status.json')['status']=='complete'
owned_pids=(557914,563934,568453,572112,572113,623556)
assert all(not Path(f'/proc/{pid}').exists() for pid in owned_pids)
source=Path.cwd()/'preparations/idea1_visualization'
(ROOT/'source_identity_final.json').write_text(json.dumps({str(path.relative_to(Path.cwd())):sha(path) for path in source.glob('*.py')},indent=2)+'\n')
receipt=dict(passed=True,episodes=120,selected_requests=2326,uniform12_samples=1440,
    analysis_rows=dict(linear=len(linear),conditional=len(conditional),backend=len(backend),clustering=len(clusters)),
    projections=len(projections),decoded_pngs=len(pngs),interactive_payload_matches=True,
    interactive_script_syntax_passed=True,owned_processes_released=list(owned_pids),
    collection_and_feature_receipts_passed=True,
    excluded_scope=['Large hidden payloads not re-read/rehashed','Trace phase accuracy is not ground truth','HTML not checked in a browser'],
    analysis_source_sha256=analysis['source_sha256'],
    current_analysis_source_sha256=sha(source/'analyze.py'),
    post_analysis_change='Complete static phase legends and distinguish conditional baseline/missing colors; frozen coordinates/CSV rerendered without refitting',
    report_sha256=sha(OUT/'report_zh.md'),checker_sha256=sha(Path(__file__)))
(ROOT/'final_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
