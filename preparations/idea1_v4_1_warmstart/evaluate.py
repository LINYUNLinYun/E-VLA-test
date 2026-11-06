"""Independent closed loops, rotated by case, on tuning or frozen confirmation."""
import argparse
import time
import json
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v4_0.rollout import episode
from .common import OUT, read, directory, key, stage_cases, stage_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--worker',required=True)
    parser.add_argument('--stage',choices=('tune','confirm'),required=True)
    args = parser.parse_args()
    manifest, pool = read(), Fresh()
    try:
        for index, case in enumerate(stage_cases(args.stage,manifest)):
            claim = lock(OUT/'case_claims'/f'{key(case)}.lock',blocking=False)
            try:
                claim.__enter__()
            except BlockingIOError:
                continue
            try:
                configs = stage_configs(args.stage)
                offset = index % len(configs)
                for cfg in configs[offset:]+configs[:offset]:
                    path = directory(cfg['id'],case)
                    if (path/'complete.json').exists():
                        continue
                    if path.exists():
                        raise RuntimeError(f'Incomplete technical output requires recovery: {path}')
                    atomic(OUT/'workers'/args.worker/'current.json',dict(stage=args.stage,
                        case=case,condition=cfg['id'],time=time.time()))
                    result = episode(path,args.port,case,cfg,pool,manifest['manifest_hash'],pace_hz=20)
                    print(json.dumps(dict(stage=args.stage,case=case,condition=cfg['id'],
                        success=result['success'],actions=result['actions'],wall_s=result['wall_s'])),flush=True)
                rpc(args.port,dict(op='reset_all'))
            finally:
                claim.__exit__(None,None,None)
    finally:
        pool.close()


if __name__=='__main__':
    main()
