"""Export the accepted module discovery without reading the running camera outcomes."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/root/nfs/embodied/benchmark');OUT=Path(__file__).resolve().parent
ANALYSIS=ROOT/'results/ease_insights_20261103/module_tradeoff_v1/discovery/analysis/analysis.json'
PINS={ANALYSIS:'0b9278150c4c64afa33e47f01e1d4a2141e954bf81db9beceb8bb1abf948d4e4',
    ROOT/'reports/ease_insights_20261103/module_tradeoff_v1/discovery_acceptance.json':'f76ce8cc10b5943b909de1e39c28a62d1fd09ddc71b40bd184fa799099019d09',
    ROOT/'reports/ease_insights_20261103/module_tradeoff_v1/independent_statistics.json':'c497114420c981d168c6f85e3b8ab095c27686326fff0b0e7a1a2d558765c6dd'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for path,pin in PINS.items():
        if sha(path)!=pin:raise ValueError('Accepted module input changed: '+str(path))
    m=json.loads(ANALYSIS.read_text())
    if not m['complete'] or m['selection']['status']!='no_candidate':raise ValueError('Need accepted complete negative discovery')
    ids=['full_d10','vlm_c','ae_16','ae_14','ae_12','hybrid_15_15'];labels=['Full*','VLM_c*','AE16','AE14','AE12','Hybrid15/15']
    suites=['libero_spatial','libero_object','libero_goal','libero_10'];names=['Spatial','Object','Goal','Long']
    rows=[next(r for r in m['condition_summaries'] if r['level']=='suite' and r['condition']==c and r['suite']==s) for c in ids for s in suites]
    if any(r['n']!=20 for r in rows):raise ValueError('Unexpected case denominator')
    counts=np.asarray([r['successes'] for r in rows]).reshape(6,4);diff=(counts[2]-counts[1])*5
    with (OUT/'suite_counts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['condition','suite','successes','n','source']);writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in ('condition','suite','successes','n')}|dict(source='preserved_original_reference' if r['condition'] in ('full_d10','vlm_c') else 'new_discovery'))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,(ax,bx,cx)=plt.subplots(1,3,figsize=(13.2,4.8),gridspec_kw={'width_ratios':[1.4,1,1.1]})
    ax.imshow(counts/20,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_xticks(range(4),names);ax.set_yticks(range(6),labels);ax.set_title('Successes / 20 per suite',loc='left',fontweight='bold')
    for i in range(6):
        for j in range(4):ax.text(j,i,f'{counts[i,j]}/20',ha='center',va='center',color='white' if counts[i,j]>=13 else '#172b4d')
    ax.tick_params(length=0);ax.set_xlabel('* Original Full / spread_c references')
    bx.barh(names,diff,color='#2c7fb8',height=.55);bx.invert_yaxis();bx.axvline(0,color='#333333',linewidth=.8)
    bx.set_xlim(-2,18);bx.set_xlabel('AE16 minus VLM_c (percentage points)');bx.set_title('No opposite suite preference',loc='left',fontweight='bold')
    for i,d in enumerate(diff):bx.text(d+.6,i,f'+{d}' if d else '0',va='center')
    bx.spines[['top','right']].set_visible(False);bx.grid(axis='x',alpha=.2);bx.set_axisbelow(True)
    services=m['fixed_input_cost']['services'];med=m['fixed_input_cost']['median_full_over_policy_cycle']
    for i,c in enumerate(ids[1:]):
        xs=[s['full_over_policy_cycle'][c] for s in services]
        cx.scatter(xs,i+np.linspace(-.1,.1,len(xs)),s=18,alpha=.55,color='#5b8db8')
        cx.scatter([med[c]],[i],marker='|',s=160,color='black',zorder=3)
    cx.set_yticks(range(5),labels[1:]);cx.invert_yaxis();cx.axvline(1.15,color='#a44b41',linestyle='--',linewidth=1)
    cx.set_xlabel('Full / periodic native policy cost');cx.set_title('Fixed-input cost only',loc='left',fontweight='bold')
    cx.spines[['top','right']].set_visible(False);cx.grid(axis='x',alpha=.2);cx.set_axisbelow(True)
    fig.suptitle('Module discovery: no qualified complementary pair',fontweight='bold',y=.99)
    fig.text(.03,.02,'320 new episodes + 160 preserved control cells; H50/R5, FullD10 + two FastD2. Discovery only.\n'
        'AE16 and VLM_c are usable, but AE16 is no weaker in every suite. Cost dots: eight completed services; no closed-loop speed claim.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.95))
    for suffix in ('png','pdf','svg'):fig.savefig(OUT/('module_discovery.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig)
    artifacts={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.csv','.png','.pdf','.svg')}
    (OUT/'provenance.json').write_text(json.dumps(dict(analysis_manifest_hash=m['manifest_hash'],inputs={str(p):v for p,v in PINS.items()},
        script_sha256=sha(__file__),artifacts=artifacts,scientific_outcomes_modified=False,current_camera_outcomes_read=False),indent=2)+'\n')
    print(json.dumps(dict(artifacts=list(artifacts),verdict=m['selection']['status'])))


if __name__=='__main__':main()
