"""Final requirement audit from case-level execution and stored raw artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.kv_reuse_probe.probe import action_metrics
from preparations.idea1_v2_2.common import seed
from preparations.idea1_v3.policy import verification_seed
from preparations.idea1_v3_1.common import LIMITS
from .manifest import ROOT,OUT


def main():
    m=json.loads((OUT/'manifest.json').read_text()); checks=[]; failures=[]
    def check(name,passed,**detail):
        item=dict(name=name,passed=bool(passed),**detail); checks.append(item)
        if not passed: failures.append(item)
    frozen=dict(m); frozen.pop('manifest_hash')
    check('manifest_digest',digest(frozen)==m['manifest_hash'])
    check('frozen_scientific_sources',all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected for name,expected in m['source_files'].items()))
    check('plan_frozen',hashlib.sha256((ROOT/'docs/experiments/idea1/idea1_v3_4_experiment_plan.md').read_bytes()).hexdigest()==m['plan_sha256'])
    check('source_idea_document_current',hashlib.sha256((ROOT/'idea1.md').read_bytes()).hexdigest()==m['idea1_sha256'])
    checkpoints=set(); n=0; e3samples=0; invalid=[]; initial_identity={}; service_ownership={}
    for cfg in m['conditions']:
        expected={(c['suite'],c['task'],c['init']) for c in m['cases'][cfg['id']]}
        found=set(); errors=[]
        for d in (OUT/cfg['id']/'episodes').glob('*/task*/init*'):
            if not d.is_dir(): continue
            if '.invalid.' in d.name:
                invalid.append(str(d)); continue
            if not (d/'result.json').exists(): continue
            try:
                r=json.loads((d/'result.json').read_text()); ident=json.loads((d/'identity.json').read_text())
                case=tuple(r['case'][k] for k in ('suite','task','init'))
                assert case not in found and case in expected
                found.add(case)
                assert r['valid'] and isinstance(r['success'],bool) and r['config']==cfg
                assert r['manifest_hash']==m['manifest_hash'] and (d/'complete.json').exists()
                assert r['termination']==('success' if r['success'] else 'timeout')
                assert 0<r['actions']<=LIMITS[case[0]]
                if not r['success']: assert r['actions']==LIMITS[case[0]]
                assert ident['case']==r['case'] and ident['settling']==10 and ident['env_seed']==7
                checkpoints.add(ident['protocol']['checkpoint_sha256'])
                rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
                telemetry=[json.loads(line) for line in (d/'telemetry.jsonl').read_text().splitlines()]
                assert len(telemetry)==r['actions']+1 and len(rows)==r['predictions']
                initial_identity.setdefault(case,set()).add((ident['init_vector_sha256'],ident['bddl_sha256'],rows[0]['input_hash'],ident['prompt']))
                service_ownership.setdefault(case,set()).add(ident['protocol']['server_pid'])
                with np.load(d/'arrays.npz') as arrays:
                    step=0; last_full=None; groupcount={'zero':0,'nonzero':0}
                    for i,row in enumerate(rows):
                        assert row['step']==step and row['replan']==i
                        assert row['seed']==seed(*case,i)
                        if cfg['id']=='E3_parent': assert row['verification_seed']==verification_seed(row['seed'])
                        executed=row['executed_length']; assert 0<executed<=row['execute_length']<=5
                        actions=arrays[f'actions_{i}']; raw=arrays[f'raw_{i}']
                        assert actions.shape==(50,7) and raw.shape==(50,32)
                        assert np.isfinite(actions).all() and np.isfinite(raw).all()
                        for j in range(executed):
                            tr=telemetry[step+j+1]
                            assert tr['step']==step+j+1 and np.array_equal(np.asarray(tr['action'],np.float32),actions[j])
                            assert np.shape(tr['eef_pos'])==(3,) and np.shape(tr['eef_quat'])==(4,)
                        route=row['route']
                        def check_nodes(nodes,steps):
                            assert len(nodes)==steps+1
                            assert np.allclose(nodes,[1-i/steps for i in range(steps+1)],atol=1e-8,rtol=0)
                        if cfg['id']=='E3_parent':
                            if route['path']=='full': check_nodes(row['timing']['full_nodes'],10)
                            if 'accepted_length' in route: check_nodes(row['timing']['draft_nodes'],2)
                        else: check_nodes(row['timing']['nodes'],cfg.get('full_steps',10) if route['path']=='full' else 2)
                        if cfg['id']!='E3_parent':
                            if cfg['mode']=='full': assert route['path']=='full'
                            if cfg['mode']=='pure': assert route['path']=='v14'
                            if cfg['mode']=='periodic':
                                assert (route['path']=='full')==(i%3==0)
                                if route['path']!='full':
                                    assert row['full_cache_age']==dict(rounds=i-last_full[0],env_steps=step-last_full[1])
                                    assert row['full_cache_age']['env_steps'] in (5,10)
                                    assert (route['path']=='hybrid')==('cutoff' in cfg)
                        else:
                            if 'accepted_length' in route:
                                group='zero' if route['accepted_length']==0 else 'nonzero'; limit=4 if group=='zero' else 2
                                selected=groupcount[group]<limit; groupcount[group]+=1
                                assert row['shadow_selection']['selected']==selected
                                if selected:
                                    sd=json.loads((d/f'shadow_{i:04d}.json').read_text())
                                    assert sd['group']==group and sd['step']==step and sd['seed']==row['seed']
                                    assert (d/f'shadow_{i:04d}.npz').exists()
                                    with np.load(d/f'shadow_{i:04d}.npz') as shadow_arrays:
                                        names={item['candidate'] for item in sd['metrics']}
                                        expected_names={'full_d10','full_d2','v14_d2','v14_d10','draft_full_q01','draft_full_q03','draft_v14_q03'}
                                        if sd['previous_available']: expected_names|={'remaining_full_q03','remaining_v14_q03'}
                                        assert names==expected_names
                                        actual_name='full_d10' if route['path']=='full' else 'v14_d2'
                                        assert np.array_equal(shadow_arrays[actual_name+'_raw'],raw)
                                        assert np.array_equal(shadow_arrays[actual_name],actions)
                                        for item in sd['metrics']:
                                            name=item['candidate']; a=shadow_arrays[name]; raw_shadow=shadow_arrays[name+'_raw']
                                            assert a.shape==(50,7) and raw_shadow.shape==(50,32)
                                            assert np.isfinite(a).all() and np.isfinite(raw_shadow).all()
                                            raw_rmse=float(np.sqrt(np.mean((raw_shadow[:5,:7]-shadow_arrays['full_d10_raw'][:5,:7])**2)))
                                            assert abs(raw_rmse-item['raw_first5_rmse'])<1e-5
                                            calculated=action_metrics(shadow_arrays['full_d10'],a)
                                            for recorded,metric in zip(item['action_metrics'],calculated):
                                                for field,value in metric.items():
                                                    if field in ('horizon','component') or value=='':
                                                        assert recorded[field]==value
                                                    else: assert abs(float(recorded[field])-float(value))<1e-5
                                            if '_q0' in name:
                                                q=.1 if name.endswith('q01') else .3
                                                assert np.allclose(item['cost']['nodes'],[q,q/2,0],atol=1e-8,rtol=0)
                                                assert item['cost']['NFE']==2
                                    e3samples+=1
                        if route['path']=='full': last_full=(i,step)
                        step+=executed
                    assert step==r['actions']
                    assert r['counts']['full']==sum(row['route']['path']=='full' for row in rows)
                    assert r['counts']['verify']==sum('accepted_length' in row['route'] for row in rows)
                    assert r['counts']['draft']==sum(row['route']['path']!='full' or 'accepted_length' in row['route'] for row in rows)
                    assert abs(r['policy_wall_s']-sum(row['timing']['policy_wall_ms'] for row in rows)/1000)<1e-7
                n+=1
            except Exception as exc: errors.append(dict(directory=str(d),error=repr(exc)))
        check(cfg['id']+'_case_keys',found==expected,completed=len(found),expected=len(expected))
        check(cfg['id']+'_execution_telemetry',not errors,errors=errors)
    check('single_checkpoint',len(checkpoints)==1,checkpoints=sorted(checkpoints))
    check('paired_case_initial_identity',all(len(values)==1 for values in initial_identity.values()),cases=len(initial_identity))
    check('paired_case_same_persistent_service',all(len(values)==1 for values in service_ownership.values()))
    for gpu in (0,2,3):
        wd=OUT/'workers'/f'gpu{gpu}'
        for name in ('gate','cross_service','qk_receipt'):
            path=wd/(name+'.json')
            result=json.loads(path.read_text()) if path.exists() else {}
            check(f'gpu{gpu}_{name}',result.get('passed',False))
        check(f'gpu{gpu}_finished',(wd/'finished.json').exists())
    pairs=0; actionmax=0.; mechanism_errors=[]
    for shard,gpu in enumerate((0,2,3)):
        path=OUT/'workers'/f'gpu{gpu}'/'mechanism_selection.json'
        if not path.exists():
            check(f'mechanism_selection_shard{shard}',False); continue
        selection=json.loads(path.read_text())
        for i,pair in enumerate(selection):
            d=OUT/'mechanism'/f'shard{shard}'/f'pair{i:03d}'
            try:
                assert (d/'complete.json').exists() and json.loads((d/'pair.json').read_text())==pair
                assert pair['current']['step']-pair['previous']['step']==5
                for side in ('current','previous'):
                    assert hashlib.sha256(Path(pair[side]['path']).read_bytes()).hexdigest()==pair[side]['sha256']
                import csv
                with (d/'action_metrics.csv').open() as f: recorded=list(csv.DictReader(f))
                with (d/'layer_metrics.csv').open() as f: layer_rows=list(csv.DictReader(f))
                assert len(layer_rows)==216
                assert len({(r['candidate'],r['kv'],r['region'],r['layer']) for r in layer_rows})==216
                for layer_row in layer_rows:
                    cosine,relative,ratio=[float(layer_row[k]) for k in ('cosine','relative_l2','rms_ratio')]
                    assert np.isfinite([cosine,relative,ratio]).all() and abs(cosine)<=1.00001 and relative>=0 and ratio>0
                    if layer_row['candidate']=='v14' and int(layer_row['layer'])<=2: assert relative==0
                with np.load(d/'actions.npz') as arrays:
                    for name in arrays.files:
                        assert np.isfinite(arrays[name]).all()
                        if name.endswith('_raw'): assert arrays[name].shape==(50,32)
                        else:
                            assert arrays[name].shape==(50,7)
                            if name.startswith('full_'): continue
                            candidate,denoise=name.rsplit('_D',1)
                            calculated=action_metrics(arrays['full_D'+denoise],arrays[name])
                            for metric in calculated:
                                got=next(r for r in recorded if r['candidate']==candidate and r['D']==denoise and r['component']==metric['component'] and int(r['horizon'])==metric['horizon'])
                                for field,value in metric.items():
                                    if field in ('horizon','component') or value=='': continue
                                    delta=abs(float(got[field])-float(value)); actionmax=max(actionmax,delta)
                                    assert delta<1e-5
                pairs+=1
            except Exception as exc: mechanism_errors.append(dict(directory=str(d),error=repr(exc)))
    check('mechanism_160_pairs',pairs==160,completed=pairs,expected=160)
    check('mechanism_raw_recomputation',not mechanism_errors,max_abs_metric_difference=actionmax,errors=mechanism_errors)
    supplement=OUT/'E3_noise_control'; noise_errors=[]; reproduced=0; noise_max=0.
    noise_audit=json.loads((supplement/'audit.json').read_text()) if (supplement/'audit.json').exists() else {}
    check('E3_noise_control_metric_audit',noise_audit.get('passed',False))
    if (supplement/'manifest.json').exists():
        nm=json.loads((supplement/'manifest.json').read_text())
        check('E3_noise_control_frozen_sources',all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected for name,expected in nm['sources'].items()))
        check('E3_noise_control_frozen_plan',hashlib.sha256(Path(nm['plan']).read_bytes()).hexdigest()==nm['plan_sha256'])
        for item in nm['selected']:
            dest=supplement/'samples'/f"sample{item['id']:03d}"
            try:
                assert (dest/'complete.json').exists() and item['seed']!=item['refine_seed']
                assert json.loads((dest/'reference_parity.json').read_text())['passed']
                with np.load(dest/'actions.npz') as got,np.load(item['shadow_arrays']) as original:
                    for new,old in (('full_d10_e0','full_d10'),('full_d2_e0','full_d2'),('draft_q03_e1','draft_full_q03'),('remaining_q03_e1','remaining_full_q03')):
                        if old not in original.files: continue
                        for suffix in ('','_raw'):
                            diff=float(np.abs(got[new+suffix]-original[old+suffix]).max()); noise_max=max(noise_max,diff)
                            assert np.allclose(got[new+suffix],original[old+suffix],atol=1e-5,rtol=1e-5)
                    metrics=json.loads((dest/'metrics.json').read_text())['metrics']
                    for metric in metrics:
                        raw_rmse=float(np.sqrt(np.mean((got[metric['candidate']+'_raw'][:5,:7]-got['full_d10_'+metric['reference']+'_raw'][:5,:7])**2)))
                        assert abs(raw_rmse-metric['raw_first5_rmse'])<1e-5
                reproduced+=1
            except Exception as exc: noise_errors.append(dict(id=item['id'],error=repr(exc)))
    check('E3_noise_control_reproduces_original_207',reproduced==207 and not noise_errors,reproduced=reproduced,max_abs_difference=noise_max,errors=noise_errors)
    status=dict(passed=not failures,time=__import__('time').time(),episodes=n,E3_selected=e3samples,checks=checks,failures=failures,
        technical_invalid_directories=invalid,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic(OUT/'final_audit.json',status)
    print(json.dumps(dict(passed=status['passed'],episodes=n,E3_selected=e3samples,mechanism_pairs=pairs,failed_checks=len(failures))))
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
