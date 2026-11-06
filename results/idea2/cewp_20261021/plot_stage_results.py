import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('results/idea2/cewp_20261021')
r=json.loads((root/'stages_gated/result.json').read_text())
t=json.loads((root/'stage_trace_summary.json').read_text())
phases=['qkv','pure_attention','ffn'];labels=['QKV','Attention','FFN']
fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
for ax,metric,title in zip(axes,['gpu_pair_speedup','ae_concurrent_slowdown','overlap'],['Paired GPU completion speedup','AE concurrent / isolated time','Actual kernel overlap (%)']):
 values=np.array([[t['stage/'+a+'__'+b+'/concurrent']['overlap_fraction']*100 if metric=='overlap' else r['pairs'][a+'__'+b][metric] for b in phases] for a in phases])
 im=ax.imshow(values,cmap='viridis' if metric!='ae_concurrent_slowdown' else 'magma')
 ax.set_xticks(range(3),labels);ax.set_yticks(range(3),labels);ax.set_xlabel('Action Expert phase');ax.set_ylabel('VLM phase');ax.set_title(title)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]<(values.min()+values.max())/2 else 'black')
 fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('π0.5 layer 0 phase screen · 100 alternating pairs · RTX 4090\nSynthetic GPU release gate; phase evidence only, not request latency',fontsize=12)
fig.savefig(root/'stage_pairing.png',dpi=180)
print(root/'stage_pairing.png')
