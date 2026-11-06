#!/usr/bin/env python3
"""Exercise only simulator recording with a fake policy; NOT model evaluation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.experiment8 import run_submodule_rollout as rollout


class FakePolicy:
    def __init__(self, *args, **kwargs):
        pass

    def infer(self, obs):
        actions = np.zeros((50,7),dtype=np.float32)
        actions[:,-1] = -1
        return {'actions':actions,'diagnostic_metadata':{
            'skip_attn_layers':[],'skip_ffn_layers':[],'denoise_steps':10,
            'action_space':'pi0_libero_unnormalized_7d_policy_command',
            'noise_sha256':'fake_recorder_smoke_not_model_inference'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    rollout.WebsocketClientPolicy=FakePolicy
    sys.argv=['run_submodule_rollout','--suite','libero_10','--task','0','--max-steps','2',
              '--warmup-steps','0','--condition-label','FAKE POLICY / recorder smoke only',
              '--output-dir',str(args.output_dir)]
    rollout.main()
    summary=json.loads((args.output_dir/'summary.json').read_text())
    assert summary['state_count']==summary['video_frame_count']==3
    assert summary['executed_actions']==2
    rows=[json.loads(line) for line in (args.output_dir/'telemetry.jsonl').read_text().splitlines()]
    assert rows[0]['when']=='initial' and rows[-1]['when']=='post_action'
    assert [row['video_frame_index'] for row in rows]==[0,1,2]
    assert np.isfinite(np.load(args.output_dir/'trajectory.npz')['eef_xyz']).all()
    print('Recorder smoke passed: measured initial + two post-action states; no model evaluation.')


if __name__=='__main__':
    main()
