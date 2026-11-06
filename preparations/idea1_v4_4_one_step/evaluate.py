import argparse
import json
import random
import time
import uuid
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_1.rollout import Environments
from .rollout import episode
from .common import OUT, key, directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    manifest = json.loads((OUT/'manifest.json').read_text())
    for index, case in enumerate(manifest['cases']):
        done = OUT/'case_done'/f'{key(case)}.json'
        if done.exists():
            continue
        try:
            with lock(OUT/'claims'/f'{key(case)}.lock', blocking=False):
                if done.exists():
                    continue
                cfgs = manifest['conditions']
                offset = index % len(cfgs)
                for cfg in cfgs[offset:]+cfgs[:offset]:
                    d = directory(cfg['id'], case)
                    if (d/'complete.json').exists():
                        continue
                    for attempt in range(2):
                        if d.exists():
                            archive = OUT/'attempts'/f'{key(case)}_{cfg["id"]}_{uuid.uuid4().hex}'
                            archive.parent.mkdir(parents=True, exist_ok=True)
                            d.rename(archive)
                        random.seed(7)
                        np.random.seed(7)
                        pool = Environments(reuse=False)
                        atomic(OUT/'workers'/args.worker/'evaluation_state.json', dict(case=case, condition=cfg['id'], time=time.time()))
                        try:
                            result = episode(d, args.port, case, cfg, pool, manifest['manifest_hash'])
                            print(json.dumps(dict(case=case, condition=cfg['id'], success=result['success'], actions=result['actions'])), flush=True)
                            break
                        except Exception:
                            import traceback
                            atomic(d/'technical_error.json', dict(attempt=attempt+1, error=traceback.format_exc()))
                            if attempt:
                                raise
                        finally:
                            pool.close()
                first, identities, raw_first = [], [], []
                for cfg in cfgs:
                    d = directory(cfg['id'], case)
                    first.append(json.loads((d/'infer.jsonl').open().readline())['input_hash'])
                    identities.append(json.loads((d/'identity.json').read_text()))
                    with np.load(d/'arrays.npz') as arrays:
                        raw_first.append(arrays['raw_0'].copy())
                assert np.allclose(raw_first[0], raw_first[1], atol=1e-5, rtol=1e-5), 'Initial Gaussian D1 output differs'
                assert len(set(first)) == 1, 'Paired initial observation differs'
                assert len({r['init_vector_sha256'] for r in identities}) == 1
                assert len({r['protocol']['server_pid'] for r in identities}) == 1
                atomic(done, dict(case=case, worker=args.worker, initial_input_hash=first[0], time=time.time()))
        except BlockingIOError:
            continue


if __name__ == '__main__':
    main()
