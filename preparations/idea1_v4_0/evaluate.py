"""Same-case rotated, independent closed loops; every valid failure is retained."""
import argparse
import json
import time
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v2_1.runtime_client import rpc
from .rollout import episode
from .common import OUT, directory, key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    manifest = json.loads((OUT/'manifest.json').read_text())
    pool = Fresh()
    wd = OUT/'workers'/args.worker
    try:
        for index, case in enumerate(manifest['cases']):
            claim = lock(OUT/'case_claims'/f'{key(case)}.lock', blocking=False)
            try:
                claim.__enter__()
            except BlockingIOError:
                continue
            try:
                configs = manifest['conditions']
                offset = index % len(configs)
                ordered = configs[offset:]+configs[:offset]
                for config in ordered:
                    path = directory(config['id'], case)
                    if (path/'complete.json').exists():
                        continue
                    if path.exists():
                        raise RuntimeError(f'Incomplete output needs explicit technical recovery: {path}')
                    atomic(wd/'current.json', dict(case=case, condition=config['id'], time=time.time()))
                    result = episode(path, args.port, case, config, pool, manifest['manifest_hash'],
                                     pace_hz=manifest['pace_hz'])
                    print(json.dumps(dict(case=case, condition=config['id'], success=result['success'],
                        actions=result['actions'], wall_s=result['wall_s'])), flush=True)
                rpc(args.port, dict(op='reset_all'))
            finally:
                claim.__exit__(None, None, None)
    finally:
        pool.close()


if __name__ == '__main__':
    main()
