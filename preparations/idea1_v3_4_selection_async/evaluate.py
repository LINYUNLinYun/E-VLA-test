import argparse,json,time
from .common import OUT,directory,ck,verify,eligible,progress
from .audit import episode as audit
from .rollout import episode
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v2_1.runtime_client import rpc


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True)
    p.add_argument('--worker',required=True);a=p.parse_args()
    m=json.loads((OUT/'closed_loop_manifest.json').read_text());verify(m);pool=Fresh()
    try:
        # Cover the small fixed screen first. All 40-case conditions then finish;
        # valid failures are never rerun. Every case remains on one service.
        order=[c for c in m['cases'] if c['task'] in (0,2)]+[c for c in m['cases'] if c['task'] not in (0,2)]
        for ci,case in enumerate(order):
            claim=lock(OUT/'case_claims'/f'{ck(case)}.lock',blocking=False)
            try:claim.__enter__()
            except BlockingIOError:continue
            try:
                configs=[c for c in m['conditions'] if eligible(c,case)]
                if all((directory(c['id'],case)/'complete.json').exists() for c in configs):continue
                # Stable rotated order avoids systematic warm-history advantage.
                offset=ci%len(configs);configs=configs[offset:]+configs[:offset]
                atomic(OUT/'workers'/a.worker/'current.json',dict(case=case,time=time.time()))
                for cfg in configs:
                    d=directory(cfg['id'],case)
                    if (d/'complete.json').exists():continue
                    if d.exists():raise RuntimeError(f'Interrupted output requires bounded technical recovery: {d}')
                    atomic(OUT/'workers'/a.worker/'current.json',dict(case=case,condition=cfg['id'],time=time.time()))
                    r=episode(d,a.port,case,cfg,pool,m['manifest_hash'])
                    checked=audit(cfg,d);atomic(OUT/'episode_audits'/cfg['id']/f'{ck(case)}.json',checked)
                    print(json.dumps(dict(condition=cfg['id'],case=case,success=r['success'],actions=r['actions'],wall_s=r['wall_s'])),flush=True)
                rpc(a.port,dict(op='reset_all'))
                atomic(OUT/'workers'/a.worker/'progress.json',dict(progress=progress(),time=time.time()))
            finally:claim.__exit__(None,None,None)
    finally:pool.close()


if __name__=='__main__':main()
