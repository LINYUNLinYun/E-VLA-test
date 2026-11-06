#!/usr/bin/env python3
"""Historical paired RGB video with measured EEF trajectories, not action integration."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_telemetry(path, alignment='pre-action-historical'):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    steps = [int(row['step']) for row in rows]
    if steps != list(range(len(rows))):
        raise ValueError('Historical telemetry must contain contiguous post-action steps from zero')
    xyz = np.asarray([row['eef_pos'] for row in rows], dtype=float)
    if xyz.shape != (len(rows), 3) or not np.isfinite(xyz).all():
        raise ValueError('Missing or invalid measured XYZ')
    expected_when = ['initial'] + ['post_action'] * (len(rows)-1) if alignment == 'state-synchronous' else ['post_action'] * len(rows)
    if [row.get('when') for row in rows] != expected_when:
        raise ValueError('Expected explicitly post-action telemetry')
    return rows, xyz


def read_video(path):
    reader = imageio.get_reader(path)
    try:
        fps = float(reader.get_meta_data()['fps'])
        frames = [frame for frame in reader]
    finally:
        reader.close()
    if not frames or fps <= 0:
        raise ValueError('Empty video or invalid frame rate')
    for frame in frames:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected RGB uint8 video frames')
    return frames, fps


def validate_video(path, expected_frames):
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,pix_fmt,width,height,nb_read_frames,r_frame_rate',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)['streams'][0]
    if stream['codec_name'] != 'h264' or stream['pix_fmt'] != 'yuv420p':
        raise ValueError(f'Unexpected encoding: {stream}')
    if int(stream['nb_read_frames']) != expected_frames:
        raise ValueError('Encoded frame count differs from frame index')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                   check=True, capture_output=True)
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-video', 'variant-video', 'baseline-telemetry', 'variant-telemetry', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--baseline-label', default='Historical Control / D3')
    parser.add_argument('--variant-label', required=True)
    parser.add_argument('--alignment', choices=('pre-action-historical','state-synchronous'), default='pre-action-historical')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline_rows, blue = read_telemetry(args.baseline_telemetry,args.alignment)
    variant_rows, orange = read_telemetry(args.variant_telemetry,args.alignment)
    baseline_frames, fps = read_video(args.baseline_video)
    variant_frames, other_fps = read_video(args.variant_video)
    if not np.isclose(fps, other_fps):
        raise ValueError('Different source frame rates require explicit time resampling')
    if len(baseline_frames) != len(blue) or len(variant_frames) != len(orange):
        raise ValueError('Historical source video must have exactly one pre-action frame per action')
    overlap = min(len(blue), len(orange))
    errors = np.linalg.norm(blue[:overlap] - orange[:overlap], axis=1)
    length = max(len(baseline_frames), len(variant_frames))
    all_xyz = np.concatenate([blue, orange])
    low, high = all_xyz.min(axis=0), all_xyz.max(axis=0)
    padding = np.maximum((high - low) * .08, .01)
    low, high = low - padding, high + padding
    fig = plt.figure(figsize=(10, 3.4), dpi=100)
    axes = [fig.add_subplot(1, 4, index + 1) for index in range(4)]
    projections = [(0, 1), (0, 2), (1, 2)]
    lines = []
    baseline_mark_every = max(1, len(blue) // 30)
    variant_mark_every = max(1, len(orange) // 30)
    for ax, (x, y) in zip(axes, projections):
        a, = ax.plot([], [], color='#1f77b4', linewidth=2, alpha=.62,
                     linestyle='-', marker='o', markersize=2.5, markevery=baseline_mark_every)
        b, = ax.plot([], [], color='#ff7f0e', linewidth=2, alpha=.78,
                     linestyle='--', marker='^', markersize=2.8, markevery=variant_mark_every)
        ax.set(xlim=(low[x], high[x]), ylim=(low[y], high[y]),
               xlabel=f'{"XYZ"[x]} (m)', ylabel=f'{"XYZ"[y]} (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.2)
        lines.append((a, b))
    action_count_offset = 0 if args.alignment == 'state-synchronous' else 1
    axes[3].plot(np.arange(overlap) + action_count_offset, errors * 100, color='#555555')
    cursor = axes[3].axvline(0, color='#d62728')
    axes[3].set(xlabel='Completed action count', ylabel='XYZ deviation (cm)',
                xlim=(0, length), ylim=(0, max(float(errors.max()) * 110, 1)))
    axes[3].grid(alpha=.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_index = []
    with imageio.get_writer(args.output, fps=fps, codec='libx264', pixelformat='yuv420p',
                            macro_block_size=2, ffmpeg_params=['-crf', '20', '-movflags', '+faststart']) as writer:
        for frame_id in range(length):
            scenes = []
            for frames, label, color in ((baseline_frames, args.baseline_label, (31,119,180)),
                                          (variant_frames, args.variant_label, (255,127,14))):
                terminated = frame_id >= len(frames)
                image = cv2.resize(frames[min(frame_id, len(frames) - 1)], (500, 500))
                panel = np.zeros((540, 500, 3), dtype=np.uint8)
                panel[40:] = image
                title = label + (' [terminated; last frame held]' if terminated else '')
                cv2.putText(panel, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                scenes.append(panel)
            # Source frame k is pre-action k; the matching saved state is post-action k-1.
            # Frame zero has no saved initial state: show no fabricated XYZ.
            state_index = frame_id if args.alignment == 'state-synchronous' else frame_id - 1
            visible_states = state_index + 1
            for (a, b), (x, y) in zip(lines, projections):
                a.set_data(blue[:min(visible_states,len(blue)), x], blue[:min(visible_states,len(blue)), y])
                b.set_data(orange[:min(visible_states,len(orange)), x], orange[:min(visible_states,len(orange)), y])
            completed_actions = state_index + action_count_offset
            cursor.set_xdata([completed_actions, completed_actions])
            fig.canvas.draw()
            plot = np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
            plot = cv2.resize(plot, (1000, 340))
            scene = np.concatenate(scenes, axis=1)
            canvas = np.concatenate([scene, plot], axis=0)
            writer.append_data(canvas)
            valid_pair = 0 <= state_index < overlap and frame_id < overlap
            frame_index.append({'frame_index':frame_id, 'video_time_s':frame_id / fps,
                'matching_post_action_step':state_index if state_index >= 0 else None,
                'paired_error_m':float(errors[state_index]) if valid_pair else None,
                'baseline_terminated':frame_id >= len(baseline_frames),
                'variant_terminated':frame_id >= len(variant_frames)})
    plt.close(fig)
    stream = validate_video(args.output, length)
    report = {'baseline_label':args.baseline_label, 'variant_label':args.variant_label,
        'source_fps':fps, 'timing':'Recorded source-video timing; not asserted to be real-time simulator playback',
        'coordinate_frame':'MuJoCo world', 'units':'m', 'encoding_validation':stream,
        'alignment':('State-synchronous N+1 frames and measured states; includes initial and terminal states.' if args.alignment == 'state-synchronous' else 'Source videos pre-action; telemetry post-action. Initial XYZ and terminal video frames unavailable.'),
        'max_overlap_xyz_deviation_m':float(errors.max()),
        'player_validation':'ffprobe and full ffmpeg decode passed; target user player not yet confirmed',
        'frame_index':frame_index}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key:value for key,value in report.items() if key!='frame_index'}, indent=2))


if __name__ == '__main__':
    main()
