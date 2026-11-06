"""Audit existing Full/V16/V15/V14 telemetry event chronology; no inference or environment."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from scripts.experiment11.protocol import RESULTS
from scripts.experiment10.common import dump_json, sha256, json_lines

VARIANTS = ('Full18', 'V16_A02', 'V15_A08', 'V14_A03')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def rigid_heights(row):
    primary = row.get('primary_objects') or []
    result = {}
    for name in primary:
        obj = (row.get('objects') or {}).get(name, {})
        position = obj.get('pos')
        if obj.get('type') == 'object' and obj.get('joint') is None and position is not None:
            result[name] = float(position[2])
    return result


def chronology(rows):
    """Candidate events, not diagnoses or causal failure labels."""
    rows = sorted((r for r in rows if r.get('when') in ('initial', 'post_action', None)),
                  key=lambda r: int(r['step']))
    if not rows:
        raise RuntimeError('Empty telemetry')
    initial = rigid_heights(rows[0])
    grasp_history, heights_history, events = [], [], []
    had_stable_grasp = False
    first = {}
    previous_command = None
    for index, row in enumerate(rows):
        step = int(row['step'])
        primary = row.get('primary_objects') or []
        grasp_flags = row.get('robot_grasps')
        grasp_known = isinstance(grasp_flags, dict) and bool(primary) and all(n in grasp_flags for n in primary)
        grasp = any(bool(grasp_flags[n]) for n in primary) if grasp_known else None
        contact_flags = row.get('robot_contacts')
        contact = (any(bool(contact_flags.get(n, False)) for n in primary)
                   if isinstance(contact_flags, dict) and primary else None)
        action = row.get('action')
        command = float(action[6]) if action is not None and len(action) >= 7 else None
        prior_stable = len(grasp_history) >= 3 and all(x is True for x in grasp_history[-3:])
        prior_consecutive = index >= 3 and [int(r['step']) for r in rows[index-3:index]] == list(range(step-3, step))
        if contact is True and 'first_contact' not in first:
            first['first_contact'] = step
        if grasp is True and 'first_grasp' not in first:
            first['first_grasp'] = step
        if command is not None:
            sign = int(np.sign(command))
            if previous_command is not None and sign != previous_command:
                events.append({'step': step, 'kind': 'gripper_sign_switch', 'from_sign': previous_command, 'to_sign': sign})
            if command < 0 and prior_stable and prior_consecutive:
                events.append({'step': step, 'kind': 'open_while_previously_stably_grasped',
                               'note': 'may_be_intended_release;not_a_failure_cause'})
                first.setdefault('first_open_while_stably_grasped', step)
            previous_command = sign
        grasp_history.append(grasp)
        if len(grasp_history) >= 3 and all(x is True for x in grasp_history[-3:]):
            last_steps = [int(r['step']) for r in rows[max(0,index-2):index+1]]
            if last_steps == list(range(step-2, step+1)):
                had_stable_grasp = True
                first.setdefault('first_stable_grasp', step-2)
        if (had_stable_grasp and len(grasp_history) >= 3 and all(x is False for x in grasp_history[-3:]) and
                [int(r['step']) for r in rows[max(0,index-2):index+1]] == list(range(step-2,step+1))):
            if 'first_grasp_loss_candidate' not in first:
                first['first_grasp_loss_candidate'] = step-2
                events.append({'step': step-2, 'confirmed_at_step': step, 'kind': 'grasp_loss_candidate',
                               'note': 'not_automatically_unintended_or_causal'})
        current = rigid_heights(row)
        if any(current.get(n, z)-z >= .01 for n,z in initial.items()):
            first.setdefault('first_lift_10mm', step)
        for name, height in current.items():
            if name not in initial:
                continue
            recent = [(s, zs[name]) for s,zs in heights_history if 0 < step-s <= 5 and name in zs]
            candidates = [(s,z) for s,z in recent if z-initial[name] >= .01 and z-height >= .01]
            if candidates and 'first_descent_10mm_candidate' not in first:
                first['first_descent_10mm_candidate'] = step
                events.append({'step': step, 'kind': 'object_descent_candidate', 'object': name,
                               'drop_mm': 1000*(max(z for _,z in candidates)-height),
                               'note': 'includes_intended_placement;not_a_failure_diagnosis'})
        heights_history.append((step,current))
        heights_history = [(s,zs) for s,zs in heights_history if step-s <= 5]
    return {**first, 'rigid_height_applicable': bool(initial),
            'grasp_known_steps': sum(x is not None for x in grasp_history), 'telemetry_steps': len(rows)}, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-id', default='round2_20261011')
    args = parser.parse_args()
    output = RESULTS / 'analysis/next_round' / args.attempt_id / 'failure_events'
    output.mkdir(parents=True, exist_ok=True)
    source = RESULTS / 'analysis/closed_loop_d3/episode_results.csv'
    with source.open(newline='') as stream:
        episodes = [r for r in csv.DictReader(stream) if r['variant_id'] in VARIANTS]
    if len(episodes) != 800:
        raise RuntimeError(f'Expected 800 existing episodes, got {len(episodes)}')
    baseline = {(r['suite'],r['task_id'],r['episode_idx']):r for r in episodes if r['variant_id']=='Full18'}
    deviation_source = RESULTS / 'analysis/closed_loop_d3/trajectory_deviation.csv'
    with deviation_source.open(newline='') as stream:
        deviations = {(r['variant_id'],r['suite'],r['task_id'],r['episode_idx']):r for r in csv.DictReader(stream)}
    hashes = {str(source):sha256(source), str(deviation_source):sha256(deviation_source)}
    summaries, timeline, selected, counts = [], [], [], Counter()
    for number, episode in enumerate(episodes,1):
        key = episode['suite'],episode['task_id'],episode['episode_idx']
        base_success, success = baseline[key]['success']=='1', episode['success']=='1'
        cohort = ('both_success' if base_success and success else 'baseline_only_success' if base_success else
                  'variant_only_success' if success else 'both_failure')
        base = RESULTS / 'closed_loop' / episode['condition'] / episode['suite'] / 'run'
        path = base / episode['telemetry_path']
        source_rows = json_lines(path)
        event_summary, events = chronology(source_rows)
        identity = {k:episode[k] for k in ('variant_id','suite','task_id','episode_idx')}
        d = deviations.get((episode['variant_id'],*key), {})
        row = {**identity, 'success':success, 'full_success':base_success, 'cohort':cohort,
               'furthest_phase':episode['furthest_semantic_phase'], **event_summary,
               'first_persistent_2cm_deviation_step':d.get('first_persistent_2cm_deviation_step'),
               'baseline_first_contact_step':d.get('baseline_first_primary_object_contact_step'),
               'divergence_before_baseline_contact':d.get('divergence_before_baseline_contact'),
               'max_xyz_deviation_m':d.get('max_xyz_deviation_m'),
               'max_deviation_step':d.get('max_deviation_step'),
               'telemetry_path':str(path), 'video_path':str(base/episode['video_path']),
               'source_scope':'actual_closed_loop;not_Full_only_shadow'}
        summaries.append(row)
        timeline.extend({**identity, 'success':success, 'cohort':cohort, **event} for event in events)
        hashes[str(path)] = sha256(path)
        counts[(episode['variant_id'],cohort,episode['furthest_semantic_phase'])] += 1
        if number % 100 == 0:
            print(json.dumps({'episodes_analyzed':number,'expected':800}),flush=True)
    for variant in VARIANTS[1:]:
        for suite in ('libero_object','libero_spatial','libero_goal','libero_10'):
            subset = [r for r in summaries if r['variant_id']==variant and r['suite']==suite]
            risky = [r for r in subset if r.get('first_open_while_stably_grasped') is not None]
            failures = [r for r in subset if r['cohort']=='baseline_only_success' and r.get('max_xyz_deviation_m')]
            ordering = lambda r:(int(r['task_id']),int(r['episode_idx']))
            if risky:
                chosen = min(risky,key=lambda r:(r['first_open_while_stably_grasped'],*ordering(r)))
                selected.append({**chosen,'selection_rule':'earliest_open_while_previously_stably_grasped',
                                 'event_step':chosen['first_open_while_stably_grasped']})
            if failures:
                chosen = min(failures,key=lambda r:(-float(r['max_xyz_deviation_m']),*ordering(r)))
                selected.append({**chosen,'selection_rule':'largest_xyz_deviation_in_paired_new_failure',
                                 'event_step':chosen['max_deviation_step']})
    write_csv(output/'episode_event_summary.csv',summaries)
    write_csv(output/'event_timeline.csv',timeline)
    write_csv(output/'failure_cohort_phase_counts.csv',[{'variant_id':v,'cohort':c,'furthest_phase':p,'count':n}
                                                       for (v,c,p),n in sorted(counts.items())])
    write_csv(output/'diagnostic_video_examples.csv',selected)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ('initial','approach','contact','grasp_or_manipulate','transport','goal_alignment','release_or_final_action')
    fig,ax=plt.subplots(figsize=(9,4))
    left=np.zeros(3)
    for phase in labels:
        values=np.array([sum(n for (v,c,p),n in counts.items() if v==variant and p==phase and c in ('both_failure','baseline_only_success'))
                         for variant in VARIANTS[1:]])
        ax.barh(VARIANTS[1:],values,left=left,label=phase);left+=values
    ax.set_xlabel('Failed episodes; furthest reached phase, not causal onset')
    ax.legend(fontsize=7,bbox_to_anchor=(1.01,1));fig.tight_layout()
    fig.savefig(output/'failure_furthest_phase.png',dpi=180);plt.close(fig)
    dump_json(output/'completion_audit.json',{'complete':True,'episodes':800,'variants':list(VARIANTS),
        'source_hashes':hashes,'event_rows':len(timeline),'diagnostic_examples':len(selected),
        'new_environment_steps':0,'new_model_predictions':0,
        'limitations':['candidate_events_are_not_failure_causes','same_step_not_progress_alignment',
                       'shadow_not_variant_executed_state','known_successes_included'],
        'outputs':{p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='completion_audit.json'}})
    print(json.dumps({'complete':True,'analysis':str(output),'episodes':800,'event_rows':len(timeline)}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
