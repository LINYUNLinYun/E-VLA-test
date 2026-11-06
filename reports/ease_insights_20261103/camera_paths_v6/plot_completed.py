"""Scientific discovery figure;requires complete accepted and independently audited data."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.camera_paths_v6 import common as c
REPORT=Path(__file__).resolve().parent


def main():
    receipt=REPORT/'discovery_pipeline_acceptance.json';pipeline=json.loads(receipt.read_text())
    c.require(pipeline['status']=='pass' and pipeline['new_cases']==160 and pipeline['preserved_control_cells']==160,'Need complete accepted discovery')
    c.require(all(c.sha(p)==v for p,v in pipeline['artifacts'].items()),'Accepted data changed')
    analysis=c.FAMILY_ROOT/'discovery/analysis/analysis.json';data=json.loads(analysis.read_text())
    c.require(data['complete'] and data['manifest_hash']==pipeline['manifest_hash'],'Mixed or partial results')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output=REPORT/'figures';output.mkdir(exist_ok=True)
    rows=[r for r in data['condition_summaries'] if r['level']=='suite']
    with (output/'successes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['condition','suite','n','successes','success_rate']);writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in writer.fieldnames})
    ratios=data['fixed_input_cost']['median_full_over_policy_cycle']
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[2,1]})
    labels=['Full (original)','Dense D2 (original)','Central image edges','Checkerboard image edges'];colors=['#5a626b','#279e80','#4779cb','#df8953']
    x=np.arange(4);width=.2
    for index,cid in enumerate(c.CONDITIONS):
        selected=[next(r for r in rows if r['condition']==cid and r['suite']==s) for s in c.SUITES]
        bars=axes[0].bar(x+(index-1.5)*width,[100*r['success_rate'] for r in selected],width,color=colors[index],label=labels[index])
        axes[0].bar_label(bars,labels=[str(r['successes']) for r in selected],fontsize=9,padding=3)
    axes[0].set_xticks(x,['Spatial','Object','Goal','Long']);axes[0].set_ylim(0,112);axes[0].set_ylabel('Success (%)');axes[0].legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.1))
    axes[1].barh(np.arange(4),[ratios[cid] for cid in c.CONDITIONS],color=colors)
    axes[1].set_yticks(np.arange(4),['Full','Dense D2','Central mask','Checker mask']);axes[1].set_xlim(0,max(ratios.values())*1.18)
    for i,cid in enumerate(c.CONDITIONS):axes[1].text(ratios[cid]+.02,i,f'{ratios[cid]:.3f}',va='center',fontsize=9)
    axes[1].set_xlabel('Same-service Full / policy cost');axes[1].axvline(1,color='black',linewidth=.6)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Full-storage image-edge masks: complete exploratory discovery',fontsize=12)
    fig.text(.5,.015,'Bar labels: successes / 20. 160 new candidate rollouts + 160 preserved control cells; 4 completed cost services. Full QK/V shapes retained; cost excludes RPC/environment.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.08,1,.94])
    for suffix in ('png','pdf','svg'):fig.savefig(output/f'discovery.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    c.atomic_json(output/'provenance.json',dict(manifest_hash=pipeline['manifest_hash'],selection_status=pipeline['selection_status'],inputs={str(p):c.sha(p) for p in (receipt,analysis)},source=dict(path=str(Path(__file__).resolve()),sha256=c.sha(__file__)),independent_confirmation=False,visually_inspected=False))
    print(json.dumps(dict(output=str(output),selection_status=pipeline['selection_status'])))


if __name__=='__main__':main()
