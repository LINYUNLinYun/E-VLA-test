"""Measured XYZ and event-state paired clips; no model or environment calls."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.experiment11.protocol import RESULTS
from scripts.experiment10.common import dump_json, sha256, json_lines
from scripts.experiment8.render_xyz_video import read_video, read_telemetry, validate_video


def state_label(row):
    names = row.get('primary_objects') or []
    flags = row.get('robot_grasps')
    grasp = str(any(bool(flags[n]) for n in names)) if isinstance(flags, dict) and names and all(n in flags for n in names) else 'unknown'
    action = row.get('action')
    command = 'N/A' if action is None else f'{float(action[6]):+.3f}'
    height = [(n, (row.get('objects') or {}).get(n, {}).get('pos')) for n in names]
    height = ', '.join(f'{n}:z={p[2]:.3f}m' for n,p in height if p is not None) or 'N/A'
    width = row.get('gripper_width_proxy')
    width = f'{width:.4f}m' if width is not None else 'unknown'
    return [f'phase: {row.get("semantic_phase", "unknown")}; grasp: {grasp}',
            f'gripper cmd: {command}; width: {width}', height]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt-id', default='round2_20261011')
    args = parser.parse_args()
    base = RESULTS / 'analysis/next_round' / args.attempt_id / 'failure_events'
    source = base / 'diagnostic_video_examples.csv'
    examples = list(csv.DictReader(source.open()))
    output = base / 'event_clips'
    output.mkdir(parents=True, exist_ok=True)
    if len(examples) > 24:
        raise RuntimeError('Frozen diagnostic case cap exceeded')
    index = []
    for number, case in enumerate(examples):
        run = RESULTS / 'closed_loop/Full18_D3_R5_FORMAL' / case['suite'] / 'run'
        episodes = json.loads((run / 'episode_log.json').read_text())
        baseline = next(e for e in episodes if int(e['task_id']) == int(case['task_id']) and
                        int(e['episode_idx']) == int(case['episode_idx']))
        paths = [run / baseline['telemetry_path'], Path(case['telemetry_path']),
                 run / baseline['video_path'], Path(case['video_path'])]
        hashes = {str(p): sha256(p) for p in paths}
        name = f'{number:02d}_{case["variant_id"]}_{case["suite"]}_task{int(case["task_id"]):02d}_init{int(case["episode_idx"]):02d}'
        video, marker = output / f'{name}.mp4', output / f'{name}.json'
        if marker.exists():
            record = json.loads(marker.read_text())
            if record['source_hashes'] != hashes or sha256(video) != record['video_sha256']:
                raise RuntimeError('Completed clip source/output mismatch')
            validate_video(video, len(record['frame_index']))
            index.append(record)
            continue
        if video.exists():
            raise RuntimeError('Refusing partial video overwrite')
        rows, xyz, frames, rates = [], [], [], []
        for t, v in ((paths[0], paths[2]), (paths[1], paths[3])):
            r, x = read_telemetry(t, 'state-synchronous')
            f, rate = read_video(v)
            if len(r) != len(f):
                raise RuntimeError('Source frame/state count mismatch')
            rows.append(r); xyz.append(x); frames.append(f); rates.append(rate)
        if rates[0] != rates[1]:
            raise RuntimeError('Source rates differ')
        event = int(case['event_step'])
        start, end = max(0,event-25), min(max(map(len,rows))-1,event+25)
        if not start <= event <= end:
            raise RuntimeError('Event outside actual source coverage')
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=100)
        colors, labels = ['#1f77b4','#ff7f0e'], ['Full18',case['variant_id']]
        combined = np.concatenate(xyz)
        traces = []
        for axis, (a,b) in zip(axes, [(0,1),(0,2),(1,2)]):
            for x,c,l,style,mark in zip(xyz,colors,labels,['-','--'],['o','^']):
                line, = axis.plot([],[],color=c,label=l,alpha=.65,linestyle=style,
                                  marker=mark,markersize=2,markevery=max(1,len(x)//30))
                traces.append((line,x,a,b))
            axis.set(xlabel=f'{"XYZ"[a]} (m)',ylabel=f'{"XYZ"[b]} (m)',
                     xlim=(combined[:,a].min()-.01,combined[:,a].max()+.01),
                     ylim=(combined[:,b].min()-.01,combined[:,b].max()+.01))
            axis.grid(alpha=.2); axis.set_aspect('equal',adjustable='box')
        axes[0].legend(fontsize=8)
        fig.tight_layout(rect=(0,0,1,.88))
        frame_index = []
        with imageio.get_writer(video, fps=rates[0], codec='libx264',pixelformat='yuv420p',
             macro_block_size=2,ffmpeg_params=['-crf','20','-movflags','+faststart']) as writer:
            for step in range(start,end+1):
                panels, states = [], []
                for r,f,label,color in zip(rows,frames,labels,[(31,119,180),(255,127,14)]):
                    terminated = step >= len(r)
                    i = min(step,len(r)-1)
                    panel = np.zeros((560,500,3),dtype=np.uint8)
                    panel[100:] = cv2.resize(f[i],(500,460))
                    text = [f'{label}; step {step}' + (' [terminated; held]' if terminated else '')] + state_label(r[i])
                    for k,line in enumerate(text):
                        cv2.putText(panel,line,(8,20+22*k),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
                    panels.append(panel); states.append(None if terminated else int(r[i]['step']))
                for line,x,a,b in traces:
                    n = min(step+1,len(x));line.set_data(x[:n,a],x[:n,b])
                fig.suptitle(f'Event {event}; current step {step}; same-clock, not progress aligned',fontsize=10)
                fig.canvas.draw()
                plot = cv2.resize(np.asarray(fig.canvas.buffer_rgba())[...,:3],(1000,300))
                writer.append_data(np.concatenate([np.concatenate(panels,axis=1),plot],axis=0))
                frame_index.append({'frame':len(frame_index),'source_step':step,
                                    'full_state_step':states[0],'variant_state_step':states[1],
                                    'event_frame':step==event})
        for line,x,a,b in traces:
            line.set_data(x[:,a],x[:,b])
        fig.suptitle(f'{labels[1]} task{case["task_id"]}/init{case["episode_idx"]}; measured XYZ',fontsize=10)
        fig.savefig(output/f'{name}.png',dpi=160);plt.close(fig)
        encoding = validate_video(video,len(frame_index))
        record = {'case':case,'source_hashes':hashes,'video_path':str(video),
                  'video_sha256':sha256(video),'encoding':encoding,'frame_index':frame_index,
                  'xyz_plot':str(output/f'{name}.png'),'selection_is_post_hoc':True,
                  'scope':'actual_closed_loop;not_shadow;not_causal_proof'}
        dump_json(marker,record);index.append(record)
        print(json.dumps({'validated_clips':len(index),'expected':len(examples)}),flush=True)
    dump_json(output/'completion_audit.json',{'complete':True,'clips':len(index),'expected':len(examples),
              'selection_sha256':sha256(source),'new_predictions':0,'environment_steps':0,
              'markers':{str(output/f'{n:02d}_{r["case"]["variant_id"]}_{r["case"]["suite"]}_task{int(r["case"]["task_id"]):02d}_init{int(r["case"]["episode_idx"]):02d}.json'):r['video_sha256'] for n,r in enumerate(index)}})


if __name__ == '__main__':
    main()
