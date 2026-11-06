"""Re-render fixed CSV values with separate baseline and missing-value colors."""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
output=root/'analysis'
source=output/'conditional_probes.csv'
groups=defaultdict(list)
for row in csv.DictReader(source.open()):
    if row['condition']=='within_task':groups[row['feature']].append(row)
suites=['libero_spatial','libero_object','libero_goal','libero_10']
for key,rows in groups.items():
    layers=sorted({int(row['layer']) for row in rows})
    matrix=np.full((40,len(layers)),np.nan);support={}
    for row in rows:
        index=suites.index(row['suite'])*10+int(row['condition_value'])
        support[index]=(int(row['n_test']),int(row['test_classes']))
        if row['balanced_accuracy'] and int(row['train_classes'])>=2 and int(row['test_classes'])>=2:
            matrix[index,layers.index(int(row['layer']))]=float(row['balanced_accuracy'])-float(row['majority_balanced_accuracy'])
    cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#333333')
    assert not np.allclose(cmap(0.5)[:3],np.asarray([.2,.2,.2]))
    fig,ax=plt.subplots(figsize=(10,15))
    image=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=-1,vmax=1)
    names=[f'{s}/task{t:02d} (n={support[si*10+t][0]}, K={support[si*10+t][1]})' for si,s in enumerate(suites) for t in range(10)]
    ax.set_yticks(range(40),names,fontsize=7)
    ax.set_xticks(range(len(layers)),[f'H{layer}' for layer in layers])
    for border in (9.5,19.5,29.5):ax.axhline(border,color='black',linewidth=.6)
    ax.set_xlabel('Hidden-state index')
    ax.set_title(key+'\nWithin-task phase: held-out balanced accuracy minus training-majority baseline',fontsize=10)
    fig.colorbar(image,ax=ax,label='Score difference; pale grey = baseline, dark = insufficient coverage',shrink=.6)
    fig.tight_layout();fig.savefig(output/'figures'/f"conditional_{key.replace('/','_')}.png",dpi=130);plt.close(fig)
report=output/'report_zh.md'
report.write_text(report.read_text().replace('灰色表示不足两个训练或留出阶段','浅灰色表示接近基线，深灰色表示不足两个训练或留出阶段'))
receipt=dict(passed=True,figures=len(groups),operation='Separate baseline from missing-value colors; fixed conditional CSV, no refitting',source_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(output/'conditional_render_complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
