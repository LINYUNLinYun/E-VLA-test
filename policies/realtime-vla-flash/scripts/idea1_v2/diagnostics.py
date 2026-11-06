"""Offline failure diagnostics and bounded clips from recorded states/videos."""
import json
import subprocess
from pathlib import Path
import numpy as np
from .core import SUITES, rows, sha, write_json
from .kalman import table
from .analysis import completed
from .runner import ALT, PHASE

PHASES = ('initial', 'approach', 'contact', 'grasp_or_manipulate', 'transport',
          'goal_alignment', 'release_or_final_action', 'success')

def detect(states, baseline):
    """Keep final furthest phase separate from first sustained discrepancy."""
    phases = [s.get('semantic_phase', 'unknown') for s in states]
    known = [p for p in phases if p in PHASES]
    furthest = max(known, key=PHASES.index) if known else 'unknown'
    by_step = {int(s.get('step', i)): s for i, s in enumerate(baseline)}
    events = []
    streak = []
    for i, s in enumerate(states):
        step = int(s.get('step', i))
        old = by_step.get(step)
        distance = (float(np.linalg.norm(np.asarray(s['eef_pos']) - old['eef_pos']))
                    if old is not None else None)
        if distance is not None and np.isfinite(distance) and distance >= .02:
            if streak and step != streak[-1] + 1: streak = []
            streak.append(step)
        else: streak = []
        if len(streak) == 3:
            events.append({'event': 'first_sustained_same_step_xyz_2cm', 'event_step': streak[0],
                           'distance_at_confirmation_m': distance})
            break
    # Unknown grasp cannot satisfy either side of the six-state criterion.
    grasps = []
    for s in states:
        names = s.get('primary_objects') or []
        flags = s.get('robot_grasps') or {}
        grasps.append(any(flags[n] for n in names) if names and all(isinstance(flags.get(n), bool) for n in names) else None)
    for i in range(3, len(states)-2):
        if all(g is True for g in grasps[i-3:i]) and all(g is False for g in grasps[i:i+3]):
            events.append({'event': 'grasp_loss_candidate', 'event_step': int(states[i].get('step', i)),
                           'may_be_normal_release': phases[i] in ('goal_alignment', 'release_or_final_action', 'success')})
    return furthest, events

def run(root, source):
    records=[]; counts={}
    for condition in (ALT, PHASE):
        for key, row in sorted(completed(root, condition).items()):
            suite, task, init = key
            old_dir=source/'closed_loop/Full18_D3_R5_FORMAL'/suite/'run'
            old=next(e for e in json.loads((old_dir/'episode_log.json').read_text()) if (e['task_id'],e['episode_idx'])==(task,init))
            directory=Path(row['directory']);states=rows(directory/row['telemetry_path'])
            furthest, events=detect(states, rows(old_dir/old['telemetry_path']))
            count_key=(condition,suite,furthest,bool(row['success']))
            counts[count_key]=counts.get(count_key,0)+1
            if not events:events=[{'event':'no_detected_event','event_step':None}]
            for event in events:
                step=event['event_step']
                records.append(dict(condition=condition,suite=suite,task=task,init=init,success=bool(row['success']),
                                    furthest_phase=furthest,full_video=str(old_dir/old['video_path']),
                                    new_video=str(directory/row['video_path']),
                                    distance_at_confirmation_m=event.get('distance_at_confirmation_m'),
                                    may_be_normal_release=event.get('may_be_normal_release'),
                                    event=event['event'],event_step=step))
    if records:table(root/'analysis/failure_events.csv',records,exclusive=False)
    if counts:table(root/'analysis/phase_counts.csv',[dict(condition=c,suite=s,furthest_phase=p,success=ok,episodes=n)
                                                   for (c,s,p,ok),n in sorted(counts.items())],exclusive=False)
    write_json(root/'analysis/failure_diagnostics.json',{'episodes':sum(counts.values()),'records':records,
               'interpretation':'Association only; furthest phase is not failure onset; grasp loss may be normal release.'})
    return records

def clips(root):
    """At most one event per fixed selected case; no extra case fishing."""
    cases_path=root/'media/cases.json'
    cases=json.loads(cases_path.read_text())['cases'] if cases_path.exists() else []
    records=json.loads((root/'analysis/failure_diagnostics.json').read_text())['records']
    result=[]
    for case in cases:
        matches=[r for r in records if all(r[k]==case[k] for k in ('condition','suite','task','init')) and r['event_step'] is not None]
        if not matches:continue
        event=min(matches,key=lambda r:(r['event_step'],r['event']))
        for role in ('full','new'):
            source=Path(event[role+'_video'])
            probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                                  '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(source)],capture_output=True,text=True,check=True)
            stream=json.loads(probe.stdout)['streams'][0];frames=int(stream['nb_read_frames'])
            num,den=map(int,stream['r_frame_rate'].split('/'));fps=num/den
            if fps!=20:raise ValueError('Event source fps differs from frozen 20Hz')
            start=max(0,event['event_step']-25);end=min(frames-1,event['event_step']+25)
            if start>end:continue # Baseline terminated before event window; never fabricate frames.
            stem='%s_%s_t%02d_i%02d_%s'%(case['condition'],case['suite'],case['task'],case['init'],role)
            output=root/'videos/event_clips'/(stem+'.mp4');marker=output.with_suffix('.json')
            provenance={'source':str(source),'source_sha256':sha(source),'start_step':start,'end_step':end,'event':event['event'],
                        'event_step':event['event_step'],'role':role,'is_full_episode':False}
            output.parent.mkdir(parents=True,exist_ok=True)
            if marker.exists():
                saved=json.loads(marker.read_text())
                if any(saved[k]!=v for k,v in provenance.items()) or saved['sha256']!=sha(output):raise ValueError('Event clip provenance changed')
            else:
                if output.exists():raise FileExistsError('Partial clip preserved: '+str(output))
                subprocess.run(['ffmpeg','-v','error','-i',str(source),'-vf',
                                'select=between(n\\,%d\\,%d),setpts=N/(20*TB)'%(start,end),'-an','-c:v','libx264',
                                '-pix_fmt','yuv420p','-movflags','+faststart',str(output)],capture_output=True,check=True)
                subprocess.run(['ffmpeg','-v','error','-i',str(output),'-f','null','-'],capture_output=True,check=True)
                write_json(marker,dict(provenance,sha256=sha(output)),exclusive=True)
            result.append(dict(provenance,path=str(output.relative_to(root))))
    write_json(root/'media/event_clips.json',{'clips':result,'window':'event ±25 actual steps; no terminal padding'})
    return result
