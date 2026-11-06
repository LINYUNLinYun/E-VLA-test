#!/usr/bin/env python3
"""Audit same-state/noise closed-loop pairs and summarize measured XYZ overlap."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiment8.render_xyz_video import validate_video


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sustained_onset(errors, threshold_m, consecutive_states):
    """First fully observed sustained positional divergence, not causal failure onset."""
    if threshold_m <= 0 or consecutive_states < 1:
        raise ValueError('Positive threshold and consecutive-state count required')
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Expected finite one-dimensional position errors')
    if len(values) < consecutive_states:
        return None
    runs = np.convolve((values > threshold_m).astype(int), np.ones(consecutive_states, dtype=int), mode='valid')
    matches = np.flatnonzero(runs == consecutive_states)
    return int(matches[0]) if len(matches) else None


def quaternion_angle_degrees(reference, candidate):
    reference, candidate = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 4:
        raise ValueError('Expected matching Nx4 quaternions')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('Non-finite quaternions')
    norms = [np.linalg.norm(values, axis=1) for values in (reference, candidate)]
    if any(np.any(norm < 1e-12) for norm in norms):
        raise ValueError('Zero quaternion')
    cosine = np.abs(np.sum((reference / norms[0][:,None]) * (candidate / norms[1][:,None]), axis=1))
    return np.degrees(2 * np.arccos(np.clip(cosine, 0, 1)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--expected-count',type=int,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    parser.add_argument('--divergence-threshold-m',type=float,default=0.02)
    parser.add_argument('--divergence-consecutive-states',type=int,default=5)
    args=parser.parse_args()
    for suffix in ('.csv', '.json'):
        if args.output_prefix.with_suffix(suffix).exists():
            raise FileExistsError('Refusing to overwrite existing paired audit: ' + str(args.output_prefix.with_suffix(suffix)))
    pairs=sorted(path for path in args.root.iterdir() if path.is_dir() and (path/args.variant/'summary.json').is_file())
    if len(pairs)!=args.expected_count:
        raise ValueError(f'Expected {args.expected_count} completed pairs, got {len(pairs)}')
    records=[]
    for path in pairs:
        summaries=[json.loads((path/name/'summary.json').read_text()) for name in ('baseline',args.variant)]
        baseline,variant=summaries
        for key in ('semantic_task_key','init_state_sha256','noise_seed','env_seed','warmup_steps','replan_steps','denoise_steps','control_frequency_hz'):
            if baseline[key]!=variant[key]:
                raise ValueError(f'{path}: pair differs in {key}')
        if baseline['skip_attn_layers'] or baseline['skip_ffn_layers']:
            raise ValueError('Baseline must be complete model')
        metadata=[{row['infer_id']:row['metadata']['noise_sha256'] for row in read_rows(path/name/'inferences.jsonl')}
                  for name in ('baseline',args.variant)]
        shared=metadata[0].keys()&metadata[1].keys()
        if not shared or any(metadata[0][key]!=metadata[1][key] for key in shared):
            raise ValueError(f'{path}: paired diffusion noise mismatch')
        trajectories=[read_rows(path/name/'telemetry.jsonl') for name in ('baseline',args.variant)]
        for name,summary,rows in zip(('baseline',args.variant),summaries,trajectories):
            if len(rows)!=summary['executed_actions']+1 or len(rows)!=summary['video_frame_count']:
                raise ValueError(f'{path/name}: invalid N+1 coverage')
            if [row['step'] for row in rows]!=list(range(len(rows))) or rows[0]['when']!='initial':
                raise ValueError('Missing initial state or non-contiguous steps')
            validate_video(path/name/'rollout.mp4',len(rows))
        overlap=min(map(len,trajectories))
        arrays=[np.asarray([row['eef_pos'] for row in rows],dtype=float) for rows in trajectories]
        if any(array.shape!=(len(rows),3) or not np.isfinite(array).all() for array,rows in zip(arrays,trajectories)):
            raise ValueError('Invalid measured XYZ')
        times=[np.asarray([row['sim_time_s'] for row in rows[:overlap]]) for rows in trajectories]
        if not np.allclose(times[0],times[1],atol=1e-8,rtol=0):
            raise ValueError('Environment-step alignment differs in simulation time')
        errors=np.linalg.norm(arrays[0][:overlap]-arrays[1][:overlap],axis=1)
        outcome=('both_success' if baseline['success'] and variant['success'] else
                 'new_failure' if baseline['success'] else
                 'recovery' if variant['success'] else 'both_failure')
        max_step=int(errors.argmax())
        onset=sustained_onset(errors,args.divergence_threshold_m,args.divergence_consecutive_states)
        rotation_errors=quaternion_angle_degrees(
            [row['eef_quat'] for row in trajectories[0][:overlap]],
            [row['eef_quat'] for row in trajectories[1][:overlap]])
        gripper_errors=np.abs(np.asarray([row['gripper_width_proxy'] for row in trajectories[0][:overlap]])-
                              np.asarray([row['gripper_width_proxy'] for row in trajectories[1][:overlap]]))
        if not np.isfinite(gripper_errors).all():
            raise ValueError('Non-finite measured gripper width proxy')
        rotation_onset=sustained_onset(rotation_errors,5.0,args.divergence_consecutive_states)
        gripper_onset=sustained_onset(gripper_errors,0.01,args.divergence_consecutive_states)
        records.append({'pair_id':path.name,'baseline_success':baseline['success'],'variant_success':variant['success'],
            'outcome':outcome,'paired_noise_blocks':len(shared),'shared_state_count':overlap,
            'baseline_actions':baseline['executed_actions'],'variant_actions':variant['executed_actions'],
            'xyz_mean_m':float(errors.mean()),'xyz_p95_m':float(np.percentile(errors,95)),
            'xyz_max_m':float(errors.max()),'xyz_max_state_index':max_step,
            'xyz_sustained_divergence_state_index':onset,
            'baseline_phase_at_divergence':trajectories[0][onset].get('semantic_phase') if onset is not None else None,
            'variant_phase_at_divergence':trajectories[1][onset].get('semantic_phase') if onset is not None else None,
            'rotation_p95_degrees':float(np.percentile(rotation_errors,95)),
            'rotation_max_degrees':float(rotation_errors.max()),
            'rotation_sustained_divergence_state_index':rotation_onset,
            'gripper_width_proxy_p95_difference':float(np.percentile(gripper_errors,95)),
            'gripper_width_proxy_max_difference':float(gripper_errors.max()),
            'gripper_sustained_divergence_state_index':gripper_onset,
            'baseline_phase_at_max':trajectories[0][max_step].get('semantic_phase'),
            'variant_phase_at_max':trajectories[1][max_step].get('semantic_phase')})
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    with args.output_prefix.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    report={'pair_count':len(records),'variant':args.variant,
        'baseline_successes':sum(row['baseline_success'] for row in records),
        'variant_successes':sum(row['variant_success'] for row in records),
        'outcome_counts':{outcome:sum(row['outcome']==outcome for row in records) for outcome in ('both_success','both_failure','new_failure','recovery')},
        'scope':'Selected tasks, one official state and seed; exploratory paired implementation/intervention diagnostics, not benchmark or noninferiority evidence.',
        'units':'m','coordinate_frame':'MuJoCo world','errors_use_only_common_valid_states':True,
        'divergence_definition':{'threshold_m':args.divergence_threshold_m,'consecutive_states':args.divergence_consecutive_states,
            'rotation_threshold_degrees':5.0,'gripper_width_proxy_threshold':0.01,
            'interpretation':'First sustained measured divergence; heuristic thresholds, not causal failure onset. Gripper width proxy is not a calibrated contact force or grasp success metric.'},'pairs':records}
    args.output_prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='pairs'},indent=2))


if __name__=='__main__':
    main()
