"""Predeclared bounded followups and promotion decisions, without GPU launches."""
import json
import hashlib
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from .analyze import OUT,records,summary
from .manifest import cases
from .costs import receipt as matched_cost_receipt,eligible as cost_eligible,rank as cost_rank


def freeze(name,conditions,expand=False,decision=None):
    parent=json.loads((OUT/'manifest.json').read_text())
    m=dict(stage=name,conditions=conditions,cases={model:cases(model,expand) for model in ('pi0','xiaomi')},
        parent_manifest_hash=parent['manifest_hash'],source_files=parent['source_files'],plan_sha256=parent['plan_sha256'],decision=decision)
    names=['selection.py','costs.py','cost_analysis.py','stage.py','xiaomi_cost.py','campaign.py','analyze.py','annotate.py','rollout_bound.py','dispatch_bound.py','egl_mapping.py']
    m['stage_tool_sources']={str(Path(__file__).with_name(n)):hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    snapshot={}
    groups=records()
    sources=list(parent['conditions'])
    previous=OUT/'stages/bounded_followups/manifest.json'
    if name!='bounded_followups' and previous.exists(): sources+=json.loads(previous.read_text())['conditions']
    for cfg in sources:
        expected={tuple(c[k] for k in ('suite','task','init')) for c in cases(cfg['model'])}
        selected={tuple(r['case'][k] for k in ('suite','task','init')):r for r in groups[cfg['id']] if tuple(r['case'][k] for k in ('suite','task','init')) in expected}
        if set(selected)!=expected: raise ValueError('Selection lacks exact shared80 case coverage: '+cfg['id'])
        entries=[]
        for key,row in sorted(selected.items()):
            if row['config']!=cfg: raise ValueError('Selection config mismatch: '+cfg['id'])
            path=Path(row['_directory'])/'result.json'
            entries.append(dict(case=list(key),result_path=str(path),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),manifest_hash=row['manifest_hash']))
        snapshot[cfg['id']]=dict(config=cfg,case_count=80,cases=entries,case_keys_sha256=digest(sorted(expected)))
    m['selection_inputs']=snapshot
    revision=OUT/'metadata_revision_r2.json'
    if revision.exists(): m['metadata_revision_receipt']=dict(path=str(revision),sha256=hashlib.sha256(revision.read_bytes()).hexdigest())
    for key,filename in (('E2_scientific_manifest','E2_manifest.json'),('EGL_engineering_revision','egl_revision.json')):
        p=OUT/filename
        if p.exists(): m[key]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    plan=Path(__file__).resolve().parents[2]/'docs/experiments/idea1/idea1_v3_3_experiment_plan.md'
    m['parent_plan_sha256']=parent['plan_sha256'];m['plan_sha256']=hashlib.sha256(plan.read_bytes()).hexdigest()
    cost=OUT/('cost_gate_pi0_screen.json' if name=='bounded_followups' else 'cost_gate_final.json')
    if cost.exists():m['matched_cost_gate_receipt']=dict(path=str(cost),sha256=hashlib.sha256(cost.read_bytes()).hexdigest())
    m['manifest_hash']=digest(m)
    path=OUT/'stages'/name/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=m: raise ValueError('Frozen stage differs')
    else: atomic(path,m)
    return path


def first80():
    gate=OUT/'cost_selection_gate.json'
    if gate.exists() and json.loads(gate.read_text()).get('pending') and not json.loads(gate.read_text()).get('pi0_screen_ready'):return None
    cost=matched_cost_receipt('pi0_screen')
    if cost is None:return None
    groups=records(); parent=json.loads((OUT/'manifest.json').read_text()); configs={c['id']:c for c in parent['conditions']}
    stats={cid:summary(rows) for cid,rows in groups.items()}
    def ready(ids): return all(stats.get(i,{}).get('n')==80 for i in ids)
    extras=[]; decisions={}
    e2=[c for c in configs.values() if c['experiment']=='E2' and not c.get('alternate')]
    if ready([c['id'] for c in e2]+['pi0_parent']):
        base=stats['pi0_parent']
        eligible=[c for c in e2 if stats[c['id']]['successes']>=base['successes']-2 and cost_eligible(c['id'],cost)]
        best=min(eligible,key=lambda c:(cost_rank(c['id'],cost),-stats[c['id']]['successes'],c['angle'],c['pose'])) if eligible else min(e2,key=lambda c:(-stats[c['id']]['successes'],cost_rank(c['id'],cost),c['angle'],c['pose']))
        extras.append(dict(best,id=best['id']+'_no_c',pose=None)); decisions['E2_no_c_parent']=best['id']
    e4=[c for c in configs.values() if c['experiment']=='E4' and c.get('mode')!='periodic']
    if ready([c['id'] for c in e4]):
        frontier=[]
        for c in e4:
            s=stats[c['id']]
            dominated=any(stats[o['id']]['successes']>=s['successes'] and stats[o['id']]['ms_per_action']<=s['ms_per_action'] and (stats[o['id']]['successes']>s['successes'] or stats[o['id']]['ms_per_action']<s['ms_per_action']) for o in e4 if o!=c)
            if not dominated: frontier.append(c['tau'])
        for boundary,new in ((.025,.0125),(.8,1.6)):
            if boundary in frontier: extras.append(dict(id=f'E4_tau{new}',experiment='E4',model='xiaomi',tau=new))
        decisions['E4_first_grid_frontier']=frontier
    if len(decisions)<2: return None
    return freeze('bounded_followups',extras,decision=decisions)


def promote():
    gate=OUT/'cost_selection_gate.json'
    if gate.exists() and json.loads(gate.read_text()).get('pending'):return None
    cost=matched_cost_receipt()
    if cost is None:return None
    groups=records(); parent=json.loads((OUT/'manifest.json').read_text()); configs=parent['conditions']
    extra=OUT/'stages/bounded_followups/manifest.json'
    if extra.exists(): configs+=json.loads(extra.read_text())['conditions']
    screen_keys={model:{tuple(c[k] for k in ('suite','task','init')) for c in cases(model)} for model in ('pi0','xiaomi')}
    stats={cid:summary([r for r in rows if tuple(r['case'][k] for k in ('suite','task','init')) in screen_keys[r['config']['model']]]) for cid,rows in groups.items()}
    if any(stats.get(c['id'],{}).get('n')!=80 for c in configs): return None
    lookup={c['id']:c for c in configs}; selected=[]; decisions={}
    base=stats['pi0_parent']
    for exp in ('E1','E2','E3'):
        candidates=[c for c in configs if c['experiment']==exp and c.get('mode')!='pure' and not c.get('alternate') and not c['id'].endswith('_no_c') and (exp!='E3' or c.get('alpha'))]
        eligible=[c for c in candidates if stats[c['id']]['successes']>=base['successes']-2 and cost_eligible(c['id'],cost)]
        chosen=[]
        if eligible:
            chosen.append(min(eligible,key=lambda c:(cost_rank(c['id'],cost),-stats[c['id']]['successes'],c['id'])))
            if exp=='E1':
                quality=min(eligible,key=lambda c:(-stats[c['id']]['successes'],cost_rank(c['id'],cost),c['id']))
                if quality not in chosen: chosen.append(quality)
        selected+=chosen; decisions[exp]=dict(eligible=[c['id'] for c in eligible],selected=[c['id'] for c in chosen])
    xbase=stats['xiaomi_pure_d2']; e4=[c for c in configs if c['experiment']=='E4' and c.get('mode')!='periodic']
    suppressed={member for eq in cost.get('behavioral_equivalence_classes',[]) for member in eq['members'] if member!=eq['representative']}
    e4=[c for c in e4 if c['id'] not in suppressed]
    quality=[c for c in e4 if stats[c['id']]['successes']>xbase['successes']]
    cost_candidates=[c for c in e4 if stats[c['id']]['successes']>=xbase['successes']-2 and cost_eligible(c['id'],cost)]
    chosen=[]
    if quality: chosen.append(min(quality,key=lambda c:(-stats[c['id']]['successes'],cost_rank(c['id'],cost),c['tau'])))
    if cost_candidates:
        c=min(cost_candidates,key=lambda c:(cost_rank(c['id'],cost),-stats[c['id']]['successes'],c['tau']))
        if c not in chosen: chosen.append(c)
    selected+=chosen; decisions['E4']=dict(quality_eligible=[c['id'] for c in quality],cost_eligible=[c['id'] for c in cost_candidates],selected=[c['id'] for c in chosen],behavioral_equivalence_classes=cost.get('behavioral_equivalence_classes',[]))
    required=[]
    if any(c['model']=='pi0' for c in selected): required+=['pi0_parent','pi0_full_d10','pi0_full_d2','pi0_pure_d2']
    if any(c['model']=='xiaomi' for c in selected): required+=['xiaomi_full_d2','xiaomi_pure_d2']
    if any(c['experiment']=='E1' for c in selected): required+=['E1_pure_d1']
    if any(c['experiment']=='E2' for c in selected):
        required+=['E2_alternate']+[c['id'] for c in configs if c['id'].endswith('_no_c')]
    for c in selected:
        if c['experiment']=='E3': required.append('E3_hard'+('0.125' if c['alpha']==1.25 else '0.15'))
    if chosen: required.append('E4_periodic')
    all_conditions=list(selected)
    for cid in required:
        if lookup[cid] not in all_conditions: all_conditions.append(lookup[cid])
    return freeze('confirm300',all_conditions,expand=True,decision=decisions)


if __name__=='__main__':
    print(first80()); print(promote())
