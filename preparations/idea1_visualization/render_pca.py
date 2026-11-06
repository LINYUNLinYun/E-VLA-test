"""Render PCA PNGs from frozen coordinates without fitting any diagnostic."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .analyze import PHASES,scatter
from .common import SUITES,atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();output=args.root/'analysis'
    source=output/'pca_projections.json'
    payload=source.read_bytes();groups=defaultdict(dict)
    projections=json.loads(payload)
    for row in projections:
        group=groups[row['feature'],row['layer']]
        assert row['suite'] not in group
        group[row['suite']]=row
    for (key,layer),rows in groups.items():
        assert set(rows)==set(SUITES)
        fig,axes=plt.subplots(4,2,figsize=(11,17))
        for index,suite in enumerate(SUITES):
            row=rows[suite];z=np.asarray(row['coordinates'])
            assert z.shape==(len(row['task']),4) and np.isfinite(z).all()
            scatter(axes[index,0],z,np.asarray(row['task']))
            scatter(axes[index,1],z,np.asarray(row['phase']),True)
            for ax in axes[index]:
                ax.set_title(f"{suite} | PC1 {row['variance'][0]:.1%}, PC2 {row['variance'][1]:.1%}",fontsize=9)
                ax.set_xlabel('PC1');ax.set_ylabel('PC2')
        axes[0,0].legend(fontsize=6,ncol=5)
        legend=axes[0,1].legend(fontsize=6,ncol=2)
        assert [item.get_text() for item in legend.get_texts()]==PHASES
        fig.suptitle(f'{key}, H/block {layer} | PCA fit on init0/1; all selected points shown',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.98))
        name=f"pca_{key.replace('/','_')}_layer{layer:02d}.png"
        fig.savefig(output/'figures'/name,dpi=130);plt.close(fig)
    receipt=dict(passed=True,figures=len(groups),projections=len(projections),
        operation='Render frozen PCA coordinates with complete phase legends; no refitting or sample changes',
        projection_sha256=hashlib.sha256(payload).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output/'pca_render_complete.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
