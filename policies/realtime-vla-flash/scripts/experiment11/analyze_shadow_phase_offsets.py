"""All predicted offsets by observed replan phase, not future realized phases."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.experiment11.protocol import RESULTS, ROOT, SUITES
from scripts.experiment10.common import PHASE_MAP, dump_json, sha256, json_lines
from scripts.experiment11.shadow_risk_metrics import differences, task_cluster_summary
from scripts.experiment11.analyze_failure_events import write_csv


def main():
    base=RESULTS/'analysis/next_round/round2_20261011'
    output=base/'shadow_phase_offsets';output.mkdir(exist_ok=True)
    controller=ROOT/'examples/libero/.venv/lib/python3.8/site-packages/robosuite/controllers/config/osc_pose.json'
    config=json.loads(controller.read_text())
    if config['output_max']!=[.05,.05,.05,.5,.5,.5] or not config['control_delta'] or config['input_max']!=1:
        raise RuntimeError('Controller scale changed')
    hashes={str(controller):sha256(controller)}
    variants=('V16_A02','V15_A08','V14_A03','V13_Custom')
    phases=('approach','contact_grasp','transport','alignment_release')
    metrics=('translation_target_difference_mm','rotation_increment_vector_difference_deg','nonzero_gripper_disagreement')
    rows=[];keys_reference=None;coverage=[]
    for variant in variants:
        values,tasks,suites,labels,keys=[],[],[],[],set()
        for suite in SUITES:
            directory=(RESULTS/'shadow_supplement_v14/round2_20261011'/variant/suite if variant=='V14_A03'
                       else RESULTS/'shadow'/variant/suite)
            path=directory/'predictions.jsonl';marker=directory/'validated.json'
            m=json.loads(marker.read_text())
            if not m['complete'] or sha256(path)!=m['predictions_sha256'] or m['environment_steps_executed_by_variant']!=0:
                raise RuntimeError('Unvalidated prediction source')
            hashes[str(path)]=sha256(path);hashes[str(marker)]=sha256(marker)
            for row in json_lines(path):
                key=(suite,int(row['task_id']),int(row['episode_idx']),int(row['replan_id']))
                if key in keys:
                    raise RuntimeError('Duplicate prediction')
                keys.add(key)
                values.append(differences(row['full_actions'],row['variant_actions'])[:,[3,4,8]])
                tasks.append(f'{suite}:{row["task_id"]}');suites.append(suite)
                labels.append(PHASE_MAP.get(row['semantic_phase'],'unknown'))
        if len(keys)!=6979 or (keys_reference is not None and keys!=keys_reference):
            raise RuntimeError('Full trajectory coverage mismatch')
        keys_reference=keys
        values=np.asarray(values);tasks=np.asarray(tasks);suites=np.asarray(suites);labels=np.asarray(labels)
        figure_grid={}
        for scope in ('ALL',*SUITES):
            for phase in (*phases,'unknown'):
                selected=(labels==phase)&(np.ones(len(values),bool) if scope=='ALL' else suites==scope)
                if not selected.any():
                    coverage.append({'variant_id':variant,'suite':scope,'phase':phase,'predictions':0,'tasks':0})
                    continue
                data=values[selected];t=tasks[selected];strata=suites[selected]
                mean,ci,n=task_cluster_summary(data,t,strata=strata if scope=='ALL' else None)
                coverage.append({'variant_id':variant,'suite':scope,'phase':phase,'predictions':len(data),'tasks':n})
                if scope=='ALL':
                    figure_grid[phase]=mean
                for offset in range(50):
                    for mi,metric in enumerate(metrics):
                        rows.append({'variant_id':variant,'suite':scope,'replan_phase':phase,'offset':offset+1,
                            'metric':metric,'task_equal_mean':float(mean[offset,mi]),'tasks':n,'predictions':len(data),
                            'ci_low':float(ci[0,offset,mi]) if n>=3 else None,
                            'ci_high':float(ci[1,offset,mi]) if n>=3 else None,
                            'stage_basis':'observed at replan;not actual future phase','exploratory':True})
        fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
        for mi,(ax,metric,unit) in enumerate(zip(axes,metrics,('mm','deg','%'))):
            factor=100 if mi==2 else 1
            grid=np.asarray([figure_grid[p][:,mi]*factor if p in figure_grid else np.full(50,np.nan) for p in phases])
            image=ax.imshow(np.ma.masked_invalid(grid),aspect='auto',cmap='magma')
            fig.colorbar(image,ax=ax,label=unit)
            ax.set_xticks([0,4,11,19,29,39,49],[1,5,12,20,30,40,50]);ax.set_yticks(range(4),phases)
            ax.axvline(4.5,color='cyan',linewidth=1);ax.set(xlabel='Predicted action offset',title=metric)
        fig.suptitle(f'{variant}: phase at replan × predicted offsets; 6–50 are unexecuted future predictions')
        fig.savefig(output/f'{variant}_phase_offset_controller_units.png',dpi=170);plt.close(fig)
        print(json.dumps({'variant':variant,'paired_records':len(keys)}),flush=True)
    write_csv(output/'phase_offset_controller_metrics.csv',rows)
    write_csv(output/'phase_offset_coverage.csv',coverage)
    dump_json(output/'completion_audit.json',{'complete':True,'variants':list(variants),'paired_keys_per_variant':6979,
         'offsets':list(range(1,51)),'source_hashes':hashes,'new_predictions':0,'environment_steps':0,
         'bootstrap_replicates':2000,'stage_basis':'phase at replan,not realized future phase',
         'warning':'controller target increments,not actual EEF displacement or final pose angle',
         'outputs':{p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='completion_audit.json'}})


if __name__=='__main__':
    main()
