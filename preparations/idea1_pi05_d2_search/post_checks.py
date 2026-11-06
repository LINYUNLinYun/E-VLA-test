"""Small CPU fixtures for frozen selection, semantic dedup and new200 keys."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from preparations.idea1_pi05_d2_search import advance,cli,summary


def row(name,n,s,cost):
    return dict(condition='pi05D2_'+name,successes=s,policy_ms_per_action=cost,config=dict(mask=name,skip=list(range(3,3+n))))


def check():
    ranks=[row('FullD2',0,35,1),row('largest',12,26,10),row('fastest',4,28,2),row('backup',6,30,3),row('fail',16,25,1)]
    assert [r['config']['mask'] for r in advance.choose(ranks)]==['largest','fastest','backup']
    odd=advance.odd_choice(ranks); assert len(odd['skip'])==13 and odd['extra_layer']==15
    assert advance.odd_choice([row('odd',5,26,1)]) is None
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); source=root/'source'; source.mkdir()
        m=cli.manifest(SimpleNamespace(masks='{"FullD2":[],"n2":[3,9]}',shards=4)); (source/'manifest.json').write_text(json.dumps(m))
        base=SimpleNamespace(command='screen',source=[source],masks='{"renamed":[3,9]}',shards=4)
        try: advance.build(base)
        except ValueError: pass
        else: raise AssertionError('Renamed duplicate semantic screen must fail')
        base.masks='{"n3":[3,9,11]}'
        assert len(advance.build(base)['tasks'])==40
        fake=[]
        for task in m['tasks']:
            # One low suite is allowed: gate is overall26/40, not per suite.
            index=len(fake)%40
            fake.append(dict(task,analysis_stage='E2',success=index>=14,source_attempt=str(source)))
        old=advance.collect; advance.collect=lambda roots:(fake,[(source,m)])
        try:
            expansion=advance.build(SimpleNamespace(command='expand',source=[source],conditions=['pi05D2_FullD2','pi05D2_n2'],shards=4))
        finally: advance.collect=old
        assert len(expansion['tasks'])==320 and {r['init'] for r in expansion['tasks']}=={7,8,9,10}
        assert len({r['key'] for r in expansion['tasks']})==320
        assert not({r['key'] for r in expansion['tasks']}&{r['key'] for r in m['tasks']})
        samples=[]
        for i,(actions,ms,success) in enumerate([(2,10,True),(5,20,False)]):
            ep=root/f'episode{i}'; ep.mkdir()
            (ep/'infer.jsonl').write_text(json.dumps(dict(executed_length=actions,timing=dict(policy_wall_ms=ms),route=dict(denoise_steps=2),round=0))+'\n')
            samples.append(dict(condition='pi05D2_n2',suite='libero_spatial',task=i,init=6,success=success,actions=actions,key=str(i),directory=str(ep),source_attempt='fixture',config=m['conditions'][1]['config'],protocol=dict(D=2,H=10,R=5,sample_kwargs={'num_steps':2},denoise_nodes=[1.,.5,0.])))
        summary.summarize(samples,root/'summary')
        with (root/'summary/overall_quality_cost.csv').open() as stream: result=next(csv.DictReader(stream))
        assert result['successes']=='1' and result['episodes']=='2' and result['overall_ge65_sample']=='False' and float(result['policy_ms_per_actual_action'])==30/7
    print('PASS CPU: overall65 selection/three categories/dedup, odd priority, semantic rescreen blocked, 320 onlynew keys, low suite allowed, actual-action cost')


if __name__=='__main__': check()
