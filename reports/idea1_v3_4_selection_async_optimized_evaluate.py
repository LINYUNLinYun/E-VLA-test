import argparse,json,time
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_selection_async.common import OUT,ck
from preparations.idea1_v3_4_selection_async.rollout import episode
from preparations.idea1_v3_4_selection_async.audit import episode as audit
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v2_1.runtime_client import rpc


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);a=p.parse_args()
    m=json.loads((OUT/'optimized_manifest.json').read_text());cfg=m['condition'];pool=Fresh()
    order=[c for c in m['cases'] if c['task'] in (0,2)]+[c for c in m['cases'] if c['task'] not in (0,2)]
    try:
        for case in order:
            d=OUT/'optimized_closed_loop'/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            if (d/'complete.json').exists():continue
            if d.exists():raise RuntimeError(f'Interrupted optimized episode: {d}')
            atomic(OUT/'optimized_current.json',dict(case=case,condition=cfg['id'],time=time.time()))
            result=episode(d,a.port,case,cfg,pool,m['manifest_hash'])
            checked=audit(cfg,d);atomic(OUT/'optimized_audits'/f'{ck(case)}.json',checked)
            print(json.dumps(dict(case=case,condition=cfg['id'],success=result['success'],actions=result['actions'],wall_s=result['wall_s'])),flush=True)
        rpc(a.port,dict(op='reset_all'))
    finally:pool.close()
    atomic(OUT/'optimized_execution_complete.json',dict(n=40,time=time.time()))


if __name__=='__main__':main()
