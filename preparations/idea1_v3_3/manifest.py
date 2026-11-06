"""Freeze scientific grid before GPU work; first80 is a prefix of all300."""
from pathlib import Path
import hashlib
from preparations.idea1_v3.io import atomic,digest
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def conditions():
    result=[]
    def add(id,model='pi0',experiment='baseline',**kw):
        result.append(dict(id=id,model=model,experiment=experiment,**kw))
    add('pi0_parent',tau=.1)
    add('pi0_full_d10',mode='full',full_steps=10)
    add('pi0_full_d2',mode='full',full_steps=2)
    add('pi0_pure_d2',mode='pure',draft_steps=2)
    for angle,change in ((10,.02),(20,.05),(30,.10)):
        for pose in (2,5,10): add(f'E2_a{angle}_b{change}_c{pose}',experiment='E2',angle=angle,change=change,pose=pose,tau=.1)
    add('E2_alternate',experiment='E2',alternate=True,tau=.1)
    add('xiaomi_full_d2',model='xiaomi',mode='full')
    add('xiaomi_pure_d2',model='xiaomi',mode='pure')
    for tau in (.025,.05,.1,.2,.4,.8): add(f'E4_tau{tau}',model='xiaomi',experiment='E4',tau=tau)
    add('E4_periodic',model='xiaomi',experiment='E4',mode='periodic')
    for tau in (.1,.15,.25,.4): add(f'E1_D1_tau{tau}',experiment='E1',draft_steps=1,tau=tau)
    add('E1_pure_d1',experiment='E1',mode='pure',draft_steps=1)
    for alpha in (1.25,1.5): add(f'E3_alpha{alpha}',experiment='E3',alpha=alpha,tau=.1)
    for tau in (.125,.15): add(f'E3_hard{tau}',experiment='E3',tau=tau)
    return result


def cases(model,expand=False):
    start=0 if model=='xiaomi' else 6
    result=[dict(suite=s,task=t,init=i) for t in range(10) for s in SUITES for i in range(start,start+2)]
    if expand:
        result += [dict(suite=s,task=t,init=i) for t in range(10) for s in SUITES for i in range(start+2,start+7)]
        result += [dict(suite=s,task=t,init=start+7) for t in range(0,10,2) for s in SUITES]
    return result


def main():
    root=Path(__file__).resolve().parents[2]; out=root/'results/idea1_v3_3/manifest.json'
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    plan=root/'docs/experiments/idea1/idea1_v3_3_experiment_plan.md'
    m=dict(schema=1,conditions=conditions(),cases={model:cases(model) for model in ('pi0','xiaomi')},source_files=sources,
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),seed='idea1-v22/1701 suite/task/init/replan',video=False,workers_per_gpu=1,batch=1)
    m['manifest_hash']=digest(m)
    if out.exists(): raise FileExistsError(out)
    atomic(out,m); print(out)


if __name__=='__main__': main()
