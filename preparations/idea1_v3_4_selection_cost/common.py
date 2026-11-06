"""Frozen pilot: eight predetermined cases, the four existing V12/V14 routes."""
import csv
import hashlib
import json
import time
from pathlib import Path
from preparations.idea1_v3.io import atomic, digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/idea1_v3_4_selection_cost'
PLAN = ROOT/'docs/experiments/idea1/idea1_v3_4_selection_cost_plan.md'
BASE = ROOT/'results/idea1_v3_4_verified_paths'
MODULE = 'preparations.idea1_v3_4_selection_cost'
SUITES = ('libero_spatial','libero_object','libero_goal','libero_10')
VARIANTS = ('baseline','stride2','stride3','combination')


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ck(case):
    return f"{case['suite']}_task{case['task']:02d}_init{case['init']:02d}"


def directory(condition, case):
    return OUT/condition/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def conditions():
    old = json.loads((BASE/'manifest.json').read_text())
    result = []
    for original in old['conditions']:
        if original['mode'] != 'verified':
            continue
        for variant in VARIANTS:
            result.append(dict(original, id=original['id']+'_'+variant,
                base_id=original['id'], variant=variant, n=8, phase='pilot8',
                selection_stride={'stride2':2,'stride3':3}.get(variant,1),
                aggregation='signed_sum_rms' if variant=='combination' else 'layer_rms_sum',
                vjp_batch_size=1, cuda_graph=False))
    assert len(result)==16
    return result


def freeze():
    from preparations.idea1_v3_4_E2_comp_error_multivjp.model import projection_weights
    from preparations.idea1_v3_4_E2_adjacency_200.common import combinations
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        return json.loads((OUT/'manifest.json').read_text())
    cases=[dict(suite=s,task=t,init=19) for t in (0,2) for s in SUITES]
    source=json.loads((BASE/'manifest.json').read_text())['source_files']
    files={p:sha(ROOT/p) for p in source}
    files.update({str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    references={str((BASE/'manifest.json').relative_to(ROOT)):sha(BASE/'manifest.json'),
        'results/idea1_v3_4_E2_comp_error_cost/manifest.json':sha(ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json')}
    if (OUT/'startup_recovery.json').exists():
        references[str((OUT/'startup_recovery.json').relative_to(ROOT))]=sha(OUT/'startup_recovery.json')
    manifest=dict(conditions=conditions(),cases=cases,source_files=files,
        references=references,plan_sha256=sha(PLAN),projection_seed=340610,
        projection_weights=projection_weights(4).tolist(),
        legal_combinations={str(k):len(combinations(k,'spaced')) for k in (4,6)},
        max_gpus=1,actors=1,new_episodes=128,scope='Exploratory pilot, not the official full benchmark',
        gradient_atol=1e-4,gradient_rtol=1e-3,action_atol=1e-5,action_rtol=1e-5,
        batch_equivalence_requires_identical_masks=True,warmups_per_method_per_input=3,
        measurements_per_method_per_input=10,graph_disabled=True,time=time.time())
    manifest['manifest_hash']=digest(manifest)
    atomic(OUT/'manifest.json',manifest)
    return manifest


def verify_frozen(manifest):
    assert sha(PLAN)==manifest['plan_sha256']
    for filename,expected in manifest['source_files'].items():
        assert sha(ROOT/filename)==expected,filename
    for filename,expected in manifest['references'].items():
        assert sha(ROOT/filename)==expected,filename


def progress():
    m=json.loads((OUT/'manifest.json').read_text())
    rows={}
    for cfg in m['conditions']:
        complete=[]
        for case in m['cases']:
            path=directory(cfg['id'],case)/'result.json'
            if path.exists() and (path.parent/'complete.json').exists():
                complete.append(json.loads(path.read_text()))
        rows[cfg['id']]=dict(completed=len(complete),target=len(m['cases']),
            success=sum(int(r['success']) for r in complete))
    return rows
