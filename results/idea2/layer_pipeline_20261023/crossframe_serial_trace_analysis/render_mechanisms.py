"""Render actual CUPTI kernels and normalized profiling mechanism summaries."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('results/idea2/layer_pipeline_20261023')
CASES=[('lookahead_serial','crossframe_serial_trace_analysis'),
       ('encoder_overlap','crossframe_encoder_trace_analysis'),
       ('combined','crossframe_combined_trace_analysis')]


def main():
    summaries=[]
    fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
    frame=5
    labels=['Vision t','Projector t','VLM t','AE1 t','AE2 t','Vision t+1','Projector t+1']
    colors=['#1f77b4','#ff7f0e','#d62728','#9467bd','#8c564b','#17becf','#bcbd22']
    xmax=0
    for ax,(mode,directory) in zip(axes,CASES):
        d=BASE/directory
        findings=json.loads((d/'crossframe_findings.json').read_text())
        report=json.loads((d/'report.json').read_text())
        stage={(r['stage'],r['iteration']):r for r in findings['stage_mean_per_request']}
        pair=next(iter(report['groups'].values()))['pairs']
        summary=dict(mode=mode,profile_only=True,requests=8,adjacent_opportunities=7,
            ae1_same_vlm_intersection_ms_per_request=findings['per_request_same_frame']['ae1_x_same_vlm_ms']['mean'],
            all_ae_vlm_intersection_ms_per_request=pair['AE_x_VLM']['overlap_ms']/8,
            ae2_next_vision_intersection_ms_per_opportunity=findings['per_adjacent_pair']['ae2_x_next_vision_ms']['mean'],
            all_ae_vision_intersection_ms_whole_window=pair['AE_x_Vision']['overlap_ms'],
            next_vlm_not_before_current_ae=findings['checks']['next_vlm_not_before_current_ae'],
            ae1_busy_ms_per_request=stage['AE',1]['kernel_union_ms'],
            ae2_busy_ms_per_request=stage['AE',2]['kernel_union_ms'],
            ae1_envelope_ms_per_request=stage['AE',1]['kernel_envelope_ms'],
            ae2_envelope_ms_per_request=stage['AE',2]['kernel_envelope_ms'],
            vlm_busy_ms_per_request=stage['VLM',None]['kernel_union_ms'],
            vision_busy_ms_per_request=stage['VisionEncoder',None]['kernel_union_ms'],
            trace=report['source'])
        summaries.append(summary)
        with (d/'kernels.csv').open() as f:
            rows=[r for r in csv.DictReader(f)if r['branch'] and r['frame'] in (str(frame),str(frame+1))]
        origin=min(int(r['start'])for r in rows if r['frame']==str(frame))
        lane_rows=[[]for _ in labels]
        for r in rows:
            current=r['frame']==str(frame);stage=r['stage'];iteration=r['iteration']
            lane=None
            if stage=='VisionEncoder':lane=0 if current else 5
            elif stage=='VisionProjector':lane=1 if current else 6
            elif current and stage=='VLM':lane=2
            elif current and stage=='AE' and iteration in ('1','2'):lane=2+int(iteration)
            if lane is not None:
                start,end=((int(r[k])-origin)/1e6 for k in ('start','end'))
                lane_rows[lane].append((start,end-start));xmax=max(xmax,end)
        for i,values in enumerate(lane_rows):
            ax.broken_barh(values,(i-.32,.64),facecolors=colors[i],linewidth=0)
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set_title(f'{mode}: logical frame {frame} and next Encoder (actual kernels only)',loc='left')
    axes[-1].set_xlabel('GPU kernel timestamps relative to current frame first kernel (ms)')
    axes[-1].set_xlim(0,xmax+.5)
    fig.suptitle('Profiling mechanism comparison; not steady latency or a speedup chart\nMetadata/AE preparation omitted; colored bars are CUPTI kernel intervals',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(BASE/'crossframe_mechanisms.png',dpi=180);plt.close(fig)
    with (BASE/'crossframe_mechanisms.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    (BASE/'crossframe_mechanisms.json').write_text(json.dumps(dict(profile_only=True,frame_plotted=frame,rows=summaries),indent=2))


if __name__=='__main__':main()
