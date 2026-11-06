"""Queue the missing offline Action Expert probe on an idle GPU, checking every 5 minutes."""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_standard_vlm_pruning as runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-existing-observations',action='store_true')
    args=parser.parse_args()
    config=runner.load_config()
    destination=runner.RESULTS/'action_expert_layer_probe'
    destination.mkdir(exist_ok=True)
    status_path=destination/'pipeline_status.json'
    def stage(name,**extra):
        record={'stage':name,'updated_at_unix':time.time(),**extra}
        status_path.write_text(json.dumps(record,indent=2)+'\n')
        runner.log_event({'event':'action_expert_followup_stage',**record})
    gpu=None
    stage('resuming_existing_observations' if args.resume_existing_observations else 'waiting_for_idle_gpu',gpu_check_interval_seconds=300)
    while not args.resume_existing_observations:
        idle=runner.free_gpus(config)
        if idle:
            gpu=idle[0];break
        runner.log_event({'event':'action_expert_probe_gpu_busy','next_check_seconds':300})
        time.sleep(300)
    env=os.environ.copy()
    env.update(MUJOCO_GL='egl',LIBERO_CONFIG_PATH=str(runner.ROOT/'.libero'),OPENPI_DATA_HOME=str(runner.OPENPI_DATA),PYTHONPATH=f'{runner.LIBERO_ROOT}:{runner.ROOT / "src"}:{runner.ROOT}')
    if gpu is not None:
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu))
    def run(command,label):
        stage(label,gpu=gpu)
        subprocess.run([str(p) for p in command],cwd=runner.ROOT,env=env,check=True)
    try:
        smoke=destination/'smoke'
        observations=destination/'observations'
        if args.resume_existing_observations:
            manifest=json.loads((observations/'manifest.json').read_text())
            smoke_summary=json.loads((smoke/'probe/summary.json').read_text())
            if manifest['task_count']!=40 or not all(r['success_matched'] and r['max_eef_abs_error']<=1e-5 for r in manifest['replay_audit']) or not smoke_summary['complete'] or not smoke_summary['checkpoint_unchanged']:
                raise RuntimeError('Existing observations/smoke parity not verified')
        else:
            run([runner.CLIENT_PYTHON,HERE/'collect_action_expert_observations.py','--output-dir',smoke/'observations','--suites','libero_object','--task-limit','1'],'smoke_replay')
            run([runner.SERVER_PYTHON,HERE/'probe_action_expert_layers.py','--observations',smoke/'observations/manifest.json','--converted-root',runner.BASE_DIR.parent,'--output-dir',smoke/'probe','--sample-limit','1','--noise-seeds','12345'],'smoke_probe')
            run([runner.CLIENT_PYTHON,HERE/'collect_action_expert_observations.py','--output-dir',observations],'collecting_real_baseline_observations')
        run([runner.SERVER_PYTHON,HERE/'probe_action_expert_layers.py','--observations',observations/'manifest.json','--converted-root',runner.BASE_DIR.parent,'--output-dir',destination/'probe'],'probing_all_18_action_layers')
        summary=json.loads((destination/'probe/summary.json').read_text())
        if not summary.get('complete') or summary.get('task_count')!=40 or summary.get('noise_seeds')!=[12345,23456,34567]:
            raise RuntimeError('Full probe coverage not verified')
        candidates=summary['closed_loop_candidates']
        if candidates:
            run([runner.SERVER_PYTHON,HERE/'screen_action_expert_candidates.py'],'screening_probe_approved_action_layers')
            stage('complete_with_candidate_screening',gpu=gpu,closed_loop_candidates=candidates,sample_count=summary['sample_count'])
        else:
            stage('complete_no_credible_pruning_candidates',gpu=gpu,closed_loop_candidates=[],sample_count=summary['sample_count'])
        subprocess.run([str(runner.SERVER_PYTHON),str(HERE/'audit_round_completion.py')],cwd=runner.ROOT,check=True)
    except Exception as exc:
        stage('failed',gpu=gpu,error=str(exc))
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
