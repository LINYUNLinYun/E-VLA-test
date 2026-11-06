"""Small CPU fixtures for selection, incremental keys, cost and capture."""
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace


def check():
    from preparations.idea1_v3_supplement import advance,bench,capture_worker,e1_cli
    from preparations.idea1_v3_supplement.bench_compare import compare
    from preparations.idea1_v3_supplement.summary import summarize
    rows=[]
    for name,s,cost,period in [('best_quality',38,20,8),('best_speed',33,10,None),('below',31,1,4)]:
        rows.append(dict(condition=name,successes=s,policy_ms_per_action=cost,config=dict(tau=.25,period=period)))
    assert [r['condition'] for r in advance.select_parents(rows)]==['best_quality','best_speed']
    records=[dict(kind='measurement',request_wall_ms=t,execute_length=n,contains_runtime_setup=False) for t,n in [(10,2),(20,5)]]
    result=bench.aggregate(records); assert result['replay_ms_per_executed_action']==30/7 and result['mean_request_ms']==15
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); source=root/'source'; source.mkdir()
        tasks=[dict(experiment='E1',condition='c',config={},checkpoint={},suite='suite',task=t,init=6,key=str(t),owner=0,execution='A') for t in range(40)]
        (source/'manifest.json').write_text(json.dumps(dict(conditions=[dict(id='c',config={})],tasks=tasks,stage='E1',shards=1)))
        saved=advance.collect; advance.collect=lambda roots:([dict(condition='c',init=6) for _ in range(40)],[])
        try: expanded=advance.expansion(source,['c'],2)
        finally: advance.collect=saved
        assert len(expanded['tasks'])==160 and {r['init'] for r in expanded['tasks']}=={7,8,9,10}
        assert len({r['key'] for r in expanded['tasks']})==160
        assert all(len({r['owner'] for r in expanded['tasks'] if r['task']==t})==1 for t in range(40))
        import subprocess
        from preparations.idea1_v3 import cli as base_cli
        old_worker=base_cli.worker; old_gate=e1_cli.require_recovery_pass; old_popen=subprocess.Popen; launches=[]
        base_cli.worker=lambda args:subprocess.Popen(['python','-m','preparations.idea1_v3.server'],env={})
        e1_cli.require_recovery_pass=lambda path:{'passed':True}
        subprocess.Popen=lambda command,*a,**kw:launches.append((command,kw))
        try: e1_cli.worker(SimpleNamespace(attempt=root/'dispatch',shard=0,safety_gate=root/'gate'),dict(shards=1,tasks=[]))
        finally: base_cli.worker=old_worker; e1_cli.require_recovery_pass=old_gate; subprocess.Popen=old_popen
        assert launches[0][0][2]=='preparations.idea1_v3_supplement.e1_server' and 'V3_QK_WORKER_RECEIPT' in launches[0][1]['env']
        # No real RPC, environment, process or GPU: capture the existing calls.
        from preparations.idea1_v3 import episode
        old_run=episode.run; old_rpc=episode.rpc; old_main=e1_cli.main
        dest=root/'capture'; dest.mkdir()
        def fake_run(*args,**kw):
            for i in range(30): episode.rpc(0,dict(op='predict',obs=i,instruction='real',suite='libero_goal',seed=i))
            return {'success':True}
        episode.run=fake_run; episode.rpc=lambda *a:None
        e1_cli.main=lambda:episode.run(dest,0,dict(task=0,init=7,suite='libero_goal',key='key'))
        try: capture_worker.main()
        finally: episode.run=old_run; episode.rpc=old_rpc; e1_cli.main=old_main
        captured=pickle.loads((dest/'real_predict_inputs_first25.pkl').read_bytes())
        assert len(captured)==25 and [c['seed'] for c in captured]==list(range(25)) and all('op' not in c for c in captured)
        samples=[]
        for i,(n,ms,success) in enumerate([(2,10,True),(5,20,False)]):
            ep=root/f'episode{i}'; ep.mkdir()
            (ep/'infer.jsonl').write_text(json.dumps(dict(executed_length=n,timing=dict(policy_wall_ms=ms),route=dict(path='full',full_reason='zero_accept',full_steps=2,accepted_length=0)))+'\n')
            samples.append(dict(experiment='E1',condition='new',suite='libero_spatial',task=i,init=6,success=success,actions=n,directory=str(ep),config=dict(revision='supplement-e1-router-v1'),source_attempt='fixture'))
        summarize(samples,root/'summary')
        import csv
        with (root/'summary/overall_quality_cost.csv').open() as stream: value=next(csv.DictReader(stream))
        assert value['successes']=='1' and value['episodes']=='2' and float(value['policy_ms_per_actual_action'])==30/7
    base=dict(revision='v',input_sha256=['same'],thread_environment={},gpu={'uuid':'g'},config={'experiment':'E1','reference':'FullD10'},measured_requests=2,mean_request_ms=20,replay_ms_per_executed_action=4,replay_executed_actions=10,measured_requests_with_runtime_setup=0)
    candidate=dict(base,config={'experiment':'E1'},mean_request_ms=15,replay_ms_per_executed_action=30/7,replay_executed_actions=7)
    assert compare(base,candidate)['executed_action_ratio']==4/(30/7)
    try: compare(base,dict(candidate,input_sha256=['different']))
    except ValueError: pass
    else: raise AssertionError('Different input banks must reject comparison')
    print('PASS CPU: frozen parent rank/common40, incremental only160, task-adjacent owner, actual-action cost, capture25/no extra calls, same-bank ratios')


if __name__=='__main__': check()
