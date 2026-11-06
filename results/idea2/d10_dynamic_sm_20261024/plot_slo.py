"""SLO trade-off, normalized within each independently measured run."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parents[3]/'reports/idea2_d10_dynamic_sm_20261024'
rows=list(csv.DictReader((out/'slo_summary.csv').open()))
assert len(rows)==24 and len({r['case'] for r in rows})==6, 'Wait for all six confirmations'
modes=['fresh_serial','lookahead_serial','encoder_overlap','combined']
labels=['F0: serial','F1: Host lookahead','F2: + Encoder overlap','F4: + layer overlap']
colors=['#64748b','#0284c7','#0d9488','#7c3aed']
fig,axes=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
for ax,period in zip(axes,[33,62,71]):
    ax.axvspan(-15,5,color='#dcfce7',alpha=.6,zorder=0)
    ax.axvline(5,color='#166534',ls='--',lw=1)
    ax.axhline(1,color='#94a3b8',ls=':',lw=1)
    for mode,label,color in zip(modes,labels,colors):
        rr=[r for r in rows if r['mode']==mode and float(r['arrival_period_ms'])==period]
        ax.scatter([(float(r['fresh_serial_capture_p95_ratio'])-1)*100 for r in rr],
                   [float(r['fresh_serial_throughput_ratio']) for r in rr],s=48,color=color,label=label,zorder=3)
    ax.set_title(f'Arrival period: {period} ms')
    ax.set_xlabel('Capture-to-action P95 change (%)')
    ax.set_xlim(-15,35)
    ax.set_ylim(.95,1.7)
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Completed request rate / F0')
axes[2].legend(fontsize=8,loc='upper right')
fig.suptitle('D10 / fresh KV: faster throughput is accepted only within the latency budget\nGreen region: P95 regression <= 5%; two independent runs per point',fontsize=12)
fig.savefig(out/'slo_tradeoff.png',dpi=180)
fig.savefig(out/'slo_tradeoff.svg')
print(out/'slo_tradeoff.png')
