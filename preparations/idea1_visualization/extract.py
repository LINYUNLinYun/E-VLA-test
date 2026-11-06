"""Extract selected real requests on one persistent model service."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from preparations.idea1_v3_1.server import rpc
from .common import SUITES, atomic_json, episode_path
from .selection import select_episode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    args=p.parse_args(); root=Path(args.root)
    completed=0; begin=time.perf_counter()
    for suite_index,suite in enumerate(SUITES):
        for task in range(10):
            if (suite_index*10+task)%args.shards!=args.shard:
                continue
            for init in range(3):
                directory=episode_path(root,suite,task,init)
                if not (directory/'complete.json').exists():
                    raise ValueError(f'Rollout not complete: {directory}')
                selection_path=directory/'selection.json'
                samples=(json.loads(selection_path.read_text())['samples'] if selection_path.exists()
                         else select_episode(directory))
                for sample in samples:
                    output=root/'features'/suite/f'task{task:02d}'/f'init{init:02d}'/f"request_{sample['request']:04d}"
                    receipt=rpc(args.port,dict(op='extract',request_path=sample['path'],output=str(output)))
                    if not receipt.get('passed'):
                        raise ValueError(receipt)
                    # Correct early pilot metadata against the actual official
                    # training config. This checkpoint disables state tokens;
                    # generic pi05 defaults do not describe pi05_libero.
                    metadata=json.loads((output/'metadata.json').read_text())
                    metadata.update(language_group='valid instruction prompt tokens; selected pi05_libero disables discrete state input',
                                    discrete_state_input=False)
                    atomic_json(output/'metadata.json',metadata)
                    atomic_json(output/'sample.json',dict(sample,suite=suite,task=task,init=init,
                                episode=f'{suite}/{task}/{init}'))
                    completed+=1
                    atomic_json(root/f'extract_progress_{args.shard}.json',dict(
                        completed=completed,elapsed_s=time.perf_counter()-begin,
                        last=f"{suite}/{task}/{init}/{sample['request']}"))
                print(json.dumps(dict(suite=suite,task=task,init=init,completed=completed,
                                      elapsed_s=time.perf_counter()-begin)),flush=True)
    atomic_json(root/f'extract_done_{args.shard}.json',dict(completed=completed,passed=True))


if __name__=='__main__':
    main()
