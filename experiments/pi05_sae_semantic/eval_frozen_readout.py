"""Evaluate an earlier frozen readout on the complete pilot, without refitting."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .select_sae_concepts import balanced_weights,concept_label,metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--collection',required=True)
    p.add_argument('--source-selection',required=True)
    p.add_argument('--full-selection',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    collection,source,full,output=map(lambda x:Path(x).resolve(),(args.collection,args.source_selection,args.full_selection,args.output))
    output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((collection/'collection_config.json').read_text())
    rows=json.loads((full/'score_observations.json').read_text())
    if len(rows)!=len(cfg['observations']):raise ValueError('Completed pilot required')
    old_rows=json.loads((source/'score_observations.json').read_text())
    old_summary=json.loads((source/'semantic_selection_summary.json').read_text())
    definitions=copy.deepcopy(json.loads((source/'concept_features.json').read_text()))
    token_pool=old_summary.get('token_pool','all');filename=old_summary.get('readout_file','readouts.npz')
    arrays={f'a{layer}_{kind}':[] for layer in (5,16) for kind in ('all','image','instruction')}
    for row in rows:
        with np.load(collection/'observations'/row['id']/filename,allow_pickle=False) as data:
            for key in arrays:arrays[key].append(data[key])
            ids=({str(layer):data[f'feature_ids{layer}'].tolist() for layer in (5,16)}
                 if old_summary.get('candidate_pool')=='full_dictionary' else cfg['feature_ids'])
    arrays={k:np.stack(v) for k,v in arrays.items()}
    weights=balanced_weights(rows);discovery=np.array([r['split']=='discovery' for r in rows]);calibration=~discovery
    locations={r['id']:i for i,r in enumerate(rows)};source_locations=[locations[r['id']] for r in old_rows]
    old_scores=np.load(source/'semantic_scores.npz',allow_pickle=False)
    score_arrays={};checks=[]
    for name,item in definitions.items():
        labels=np.array([concept_label(r['instruction'],item['category'],item['concept']) for r in rows])
        for layer,entry in item['layers'].items():
            features=entry['features']
            if not features:entry['supported']=False;continue
            positions={f:i for i,f in enumerate(ids[layer])}
            for kind in ('all','image','instruction'):
                a=arrays[f'a{layer}_{kind}'][:,[positions[f['feature_id']] for f in features]]
                u=(a-np.array([f['tau'] for f in features]))/np.array([f['b'] for f in features])
                score=np.clip(u,0,1) @ np.array([f['weight'] for f in features])
                key=f"L{layer}_{item['category']}_{item['concept']}_{kind}"
                if key in old_scores:
                    error=float(np.abs(score[source_locations]-old_scores[key]).max())
                    if not np.allclose(score[source_locations],old_scores[key],atol=1e-12,rtol=1e-12):
                        raise ValueError('Frozen coefficients do not reproduce the original observations')
                    checks.append(dict(key=key,max_abs=error))
                score_arrays[key]=score
            score=score_arrays[f"L{layer}_{item['category']}_{item['concept']}_{token_pool}"]
            theta=entry.get('theta',.8)
            dm=metrics(score[discovery]>=theta,labels[discovery],[r for r,flag in zip(rows,discovery) if flag],weights[discovery])
            cm=metrics(score[calibration]>=theta,labels[calibration],[r for r,flag in zip(rows,calibration) if flag],weights[calibration])
            entry['source_calibration_metrics']=entry.get('calibration_metrics')
            entry['calibration_metrics']=dict(theta=theta,**cm)
            entry['evaluation_discovery_metrics']=dm
            entry['supported']=bool(cm['precision'] is not None and cm['precision']>=.8 and cm['false_positive_rate']<=.2
                                    and cm['positive_episodes']>=2 and dm['precision'] is not None and dm['precision']>=.8)
            entry['reason']='Frozen readout supported on completed pilot' if entry['supported'] else 'Frozen readout still needs evidence'
        item['ready_for_intervention']=all(item['layers'][str(layer)]['supported'] for layer in (5,16))
    (output/'concept_features.json').write_text(json.dumps(definitions,indent=2))
    np.savez_compressed(output/'semantic_scores.npz',**score_arrays)
    (output/'score_observations.json').write_text(json.dumps(rows,indent=2))
    summary=dict(completed_observations=len(rows),expected_observations=len(rows),partial=False,
        candidate_pool=old_summary.get('candidate_pool','released'),readout_file=filename,token_pool=token_pool,
        source_selection=str(source),source_selection_sha256=hashlib.sha256((source/'concept_features.json').read_bytes()).hexdigest(),
        source_discovery_observations=sum(r['split']=='discovery' for r in old_rows),
        source_calibration_observations=sum(r['split']=='calibration' for r in old_rows),
        refitted_features=False,refitted_thresholds=False,original_score_checks=checks,
        ready_concepts=[name for name,item in definitions.items() if item['ready_for_intervention']],
        concepts={name:{layer:{k:v for k,v in value.items() if k not in ['features','feature_count_options','threshold_options']}
                        for layer,value in item['layers'].items()} for name,item in definitions.items()},
        scope='Earlier frozen feature/normalization/threshold snapshot evaluated on completed pilot; no test SR')
    (output/'semantic_selection_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(ready_concepts=summary['ready_concepts'],original_scores_reproduced=True,
        source_discovery=summary['source_discovery_observations'],source_calibration=summary['source_calibration_observations'],
        evaluated_observations=len(rows)),indent=2))


if __name__=='__main__':
    main()
