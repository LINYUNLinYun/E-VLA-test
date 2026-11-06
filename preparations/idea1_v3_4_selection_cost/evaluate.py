"""One environment, fixed official states, all variants in rotated order."""
import argparse
import json
import time
from .common import OUT, directory, ck, progress, verify_frozen
from .finish import audit_episode
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v3_4_verified_paths.rollout import episode
from preparations.idea1_v2_1.runtime_client import rpc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args();manifest=json.loads((OUT/'manifest.json').read_text())
    verify_frozen(manifest);pool=Fresh()
    try:
        for case_index,case in enumerate(manifest['cases']):
            cfgs=manifest['conditions'];offset=case_index%len(cfgs)
            cfgs=cfgs[offset:]+cfgs[:offset]
            for cfg in cfgs:
                output=directory(cfg['id'],case)
                if (output/'complete.json').exists():
                    result=json.loads((output/'result.json').read_text())
                else:
                    atomic(OUT/'current.json',dict(case=case,condition=cfg['id'],time=time.time()))
                    # A valid timeout is final; technical failures stop with their evidence.
                    if output.exists():
                        raise RuntimeError(f'Interrupted output needs explicit technical recovery: {output}')
                    result=episode(output,args.port,case,cfg,pool,manifest['manifest_hash'])
                audited=audit_episode(cfg,output,result)
                atomic(OUT/'episode_audits'/cfg['id']/f'{ck(case)}.json',audited)
                state=progress()
                atomic(OUT/'status.json',dict(status='evaluating',progress=state,
                    completed=sum(row['completed'] for row in state.values()),target=128,time=time.time()))
                print(json.dumps(dict(condition=cfg['id'],case=case,success=result['success'],
                    actions=result['actions'],wall_s=result['wall_s'])),flush=True)
            rpc(args.port,dict(op='reset_all'))
    finally:pool.close()


if __name__=='__main__':main()
