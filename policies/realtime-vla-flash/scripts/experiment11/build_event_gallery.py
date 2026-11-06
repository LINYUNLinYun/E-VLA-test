"""Create an audited navigation index and readable measured-XYZ event figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.experiment11.protocol import RESULTS
from scripts.experiment10.common import dump_json, sha256, json_lines


def main():
    root = RESULTS / 'analysis/next_round/round2_20261011/failure_events/event_clips'
    audit = json.loads((root/'completion_audit.json').read_text())
    markers = sorted(root.glob('[0-9][0-9]_*.json'))
    if not audit['complete'] or len(markers) != audit['expected']:
        raise RuntimeError('Video completion not proven')
    lines = ['# 真实闭环事件视频与XYZ轨迹', '',
             '蓝色Full，橙色variant。所有XYZ来自telemetry，单位m；红色事件标记按同环境step定位，不是进度对齐。', '',
             '案例按冻结规则事后选取，不代表总体事件频率，也不证明事件导致失败。视频结束后的停留帧已标terminated，不虚构轨迹延伸。', '',
             '| 候选 / suite / task / init | 选择规则 | 视频 | 轨迹 |', '| --- | --- | --- | --- |']
    hashes = {}
    for marker in markers:
        record = json.loads(marker.read_text()); case = record['case']
        video = Path(record['video_path'])
        if sha256(video) != record['video_sha256']:
            raise RuntimeError('Video changed after decode audit')
        if sum(bool(r['event_frame']) for r in record['frame_index']) != 1:
            raise RuntimeError('Event frame mapping invalid')
        sources = list(record['source_hashes'])
        telemetry = [Path(p) for p in sources if p.endswith('telemetry.jsonl')]
        if len(telemetry) != 2:
            raise RuntimeError('Missing paired telemetry')
        data = []
        for p in telemetry:
            if sha256(p) != record['source_hashes'][str(p)]:
                raise RuntimeError('Telemetry changed')
            rows = json_lines(p)
            data.append(np.asarray([r['eef_pos'] for r in rows]))
        # JSON uses sorted dictionary keys; explicitly put Full first.
        order = sorted(range(2),key=lambda i: 'Full18_D3_R5_FORMAL' not in str(telemetry[i]))
        data = [data[i] for i in order]
        fig, axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
        event = int(case['event_step'])
        for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
            for xyz,color,label,style,mark in zip(data,['#1f77b4','#ff7f0e'],
                    ['Full18',case['variant_id']],['-','--'],['o','^']):
                ax.plot(xyz[:,a],xyz[:,b],color=color,label=label,linestyle=style,
                        marker=mark,markersize=3,markevery=max(1,len(xyz)//30),alpha=.6)
                if event < len(xyz):
                    ax.scatter(xyz[event,a],xyz[event,b],s=60,facecolors='none',edgecolors='red',zorder=5)
            ax.set(xlabel=f'{"XYZ"[a]} (m)',ylabel=f'{"XYZ"[b]} (m)')
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
        axes[0].legend(fontsize=9)
        fig.suptitle(f'{case["variant_id"]} / {case["suite"]} / task{case["task_id"]} init{case["episode_idx"]}; event step {event}',fontsize=12)
        image = root/(marker.stem+'_xyz_v2.png')
        fig.savefig(image,dpi=160);plt.close(fig)
        hashes[str(marker)] = sha256(marker);hashes[str(video)] = sha256(video);hashes[str(image)] = sha256(image)
        label = f'{case["variant_id"]} / {case["suite"]} / {case["task_id"]} / {case["episode_idx"]}'
        lines.append(f'| {label} | {case["selection_rule"]} | [MP4]({video.name}) | [XYZ]({image.name}) |')
    index = root/'index.md'
    index.write_text('\n'.join(lines)+'\n')
    hashes[str(index)] = sha256(index)
    dump_json(root/'gallery_audit.json',{'complete':True,'cases':len(markers),'output_hashes':hashes,
              'video_decode_evidence':str(root/'completion_audit.json'),'new_predictions':0,'environment_steps':0})
    print(json.dumps({'complete':True,'cases':len(markers),'index':str(index)}))


if __name__ == '__main__':
    main()
